"""Export the numbers shown on the project website (WEBSITE_PLAN.md §5).

    python -m web.export [--runs runs] [--out web/site/public/data] [--online]

Reads runs/ (log.csv, metrics.json of every non-smoke run), reports/tables/, configs/ and the split
manifests, and writes JSON for the static site:
  results.json        every experiment x unit x label set: seeds, mean, sd, per-relation F1 (mean, sd),
                      plus the not-to-scale subset (7.4) from reports/tables/not_to_scale_eval.csv
  scaling.json        the 7.9 points (analysis.phase7.scaling_points) and the training budget
  pseudo_labels.json  the 7.11 post-hoc pseudo-label precision per filter and relation
  facts.json          site metadata (web/site.yaml), dataset sizes, tolerances, overlay, recipe, compute,
                      backbone architecture, verifier audit, not-to-scale counts, highlight-colour check
Nothing on the page is typed by hand: rerun this after new runs (e.g. the Part B control seeds, Phase 8)
and the page follows. Backbone facts come from the Hugging Face config in the local cache (no download
unless --online); a backbone whose config is unavailable is exported with null facts and a reason.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np
import yaml

from analysis import phase7
from data.queries import vocabulary
from experiments.run import REPO_ROOT, load_config

SITE_YAML = REPO_ROOT / "web" / "site.yaml"
DEFAULT_OUT = REPO_ROOT / "web" / "site" / "public" / "data"
SCHEMA_VERSION = 1


def _r(x, nd: int = 4):
    return None if x is None else round(float(x), nd)


def _stats(v: list[float]) -> dict:
    a = np.asarray(v, float)
    return {"mean": _r(a.mean()), "sd": _r(a.std(ddof=1)) if len(a) > 1 else None, "n_seeds": len(a)}


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def _num(s):
    """CSV cell -> int / float / bool / None / str."""
    if s is None or s == "":
        return None
    if s in ("True", "False"):
        return s == "True"
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return s


def _count_lines(path: Path, header: bool = False) -> int | None:
    if not path.exists():
        return None
    with open(path) as f:
        n = sum(1 for line in f if line.strip())
    return n - 1 if header else n


# ---------------------------------------------------------------- results


def results(runs: Path, tables: Path, vocab: list[str]) -> dict:
    latest = phase7.latest_runs(runs)
    rows = phase7.per_run_table(runs, latest, vocab)
    by = defaultdict(list)
    for r in rows:
        by[(r["experiment"], r["unit"], r["label_set"])].append(r)
    exps: dict[str, dict] = {}
    for (exp, unit, ls), g in by.items():
        g = sorted(g, key=lambda r: r["seed"])
        e = exps.setdefault(exp, {"plan": phase7.EXPERIMENTS.get(exp, ("", None))[0], "seeds": [], "units": {},
                                  "backbone": g[0]["backbone"]})
        e["seeds"] = sorted(set(e["seeds"]) | {r["seed"] for r in g})
        e["units"].setdefault(unit, {})[ls] = {
            **_stats([r["macro_f1"] for r in g]), "seeds": [r["seed"] for r in g],
            "values": [r["macro_f1"] for r in g], "n_queries": g[0]["n"],
            "per_relation": {k: _stats([r[f"f1_{k}"] for r in g]) for k in vocab
                             if all(r[f"f1_{k}"] is not None for r in g)}}
    for e in exps.values():
        e["n_seeds"] = len(e["seeds"])
    order = {k: i for i, k in enumerate(phase7.EXPERIMENTS)}
    exps = dict(sorted(exps.items(), key=lambda kv: order.get(kv[0], 99)))

    nts = defaultdict(list)
    for r in read_csv(tables / "not_to_scale_eval.csv"):
        nts[(r["experiment"], r["label_set"])].append(r)
    not_to_scale = {}
    for (exp, ls), g in nts.items():
        not_to_scale.setdefault(exp, {})[ls] = {**_stats([float(r["macro_f1"]) for r in g]),
                                                "n_queries": _num(g[0]["n_queries"]),
                                                "n_diagrams": _num(g[0]["n_diagrams"])}
    units = {}
    for e in exps.values():
        for u, d in e["units"].items():
            units.setdefault(u, {"n_queries": next(iter(d.values()))["n_queries"]})
    return {"relations": vocab, "label_sets": ["geometric", "asserted"],
            "units": {u: units[u] for u in phase7.UNITS if u in units}, "threshold": 0.5,
            "runs": sorted(latest), "experiments": exps, "not_to_scale": not_to_scale}


def scaling(runs: Path, vocab: list[str], cfg: dict) -> dict:
    latest = phase7.latest_runs(runs)
    rows = phase7.per_run_table(runs, latest, vocab)
    ft = cfg["finetune"]
    return {"points": phase7.scaling_points(rows), "sizes": cfg["scaling"]["sizes"],
            "units": [u for u, _ in phase7.SCALING_UNITS],
            "budget": {"epochs": ft["epochs"], "epoch_images": ft["epoch_images"],
                       "total_images": ft["epochs"] * ft["epoch_images"], "real_fraction": ft["real_fraction"]}}


def pseudo_labels(runs: Path, vocab: list[str]) -> dict:
    latest = phase7.latest_runs(runs)
    rows = phase7.pseudo_label_table(runs, latest, vocab)
    if not rows:
        return {"available": False}
    first = rows[0]
    filters: dict[str, dict] = {}
    for r in rows:
        filters.setdefault(r["kept_by"], {})[r["relation"]] = {
            "pseudo_pos": r["pseudo_pos"], "pos_precision": r["pos_precision"], "pseudo_neg": r["pseudo_neg"],
            "neg_accuracy": r["neg_accuracy"], **({"accuracy": r["accuracy"]} if r["relation"] == "all" else {})}
    return {"available": True, "teacher": first["teacher"], "tau": first["tau"],
            "consistency_negatives": first["consistency_negatives"], "pool_diagrams": first["pool_diagrams"],
            "contradiction_diagrams": first["contradiction_diagrams"],
            "truth_contradiction_rate": first["truth_contradiction_rate"], "filters": filters}


# ---------------------------------------------------------------- facts


def hf_backbone_facts(name: str, long_side: int, online: bool = False) -> dict:
    """Architecture of a hub backbone from its config (built on the meta device: no weights)."""
    try:
        import torch
        from transformers import AutoConfig, AutoModel
        c = AutoConfig.from_pretrained(name, local_files_only=not online)
        with torch.device("meta"):
            m = AutoModel.from_config(c)
        regs = getattr(c, "num_register_tokens", 0) or 0
        side = long_side // c.patch_size
        return {"name": name, "available": True, "class": type(m).__name__, "hidden": c.hidden_size,
                "layers": c.num_hidden_layers, "heads": c.num_attention_heads, "patch": c.patch_size,
                "register_tokens": regs, "params_M": round(sum(p.numel() for p in m.parameters()) / 1e6, 1),
                "grid": side, "patch_tokens": side * side, "tokens": side * side + 1 + regs}
    except Exception as e:  # noqa: BLE001 - gated or not cached: exported as unavailable
        return {"name": name, "available": False, "reason": f"{type(e).__name__}: {str(e)[:160]}"}


def facts(cfg: dict, site: dict, runs: Path, tables: Path, processed: Path, vocab: list[str],
          backbone_facts: Callable[[str, int], dict]) -> dict:
    splits = processed / "splits"
    L = cfg["input"]["long_side"]
    demo_cfg = load_config(REPO_ROOT / site["demo_config"]) if site.get("demo_config") else cfg
    data = {
        "geometry3k_train_figures": _count_lines(splits / "train.txt"),
        "geometry3k_val_figures": _count_lines(splits / "val.txt"),
        "geometry3k_test_figures": _count_lines(splits / "test.txt"),
        "pgdp5k_heldout_figures": _count_lines(splits / "pgdp5k_heldout.txt"),
        "pgps9k_pool_figures": _count_lines(splits / "pgps9k_pool.txt"),
        "synth_train": {str(n): _count_lines(splits / f"synth_train_{n // 1000}k.txt")
                        for n in cfg["synthetic"]["pool_sizes"]},
        "synth_extrap_diagrams": _count_lines(processed / "synth_extrap.jsonl"),
        "train_queries": _count_lines(processed / "index" / "train.jsonl"),
        "max_queries_per_diagram": cfg["data"]["max_queries_per_diagram"],
        "eval_caps": cfg["eval"]["max_queries"],
        "synthetic_points": cfg["synthetic"]["n_points"], "extrap_points": cfg["synthetic"]["extrap_points"],
    }
    tol = cfg["tolerances"]
    tolerances = {
        "parallel_deg": tol["parallel_deg"], "perpendicular_deg": tol["perpendicular_deg"],
        "equal_length_pct": _r(100 * tol["equal_length_rel"], 2), "on_circle_pct": _r(100 * tol["on_circle_rel"], 2),
        "tangent_pct": _r(100 * tol["tangent_rel"], 2), "collinear_pct": _r(100 * tol["collinear_rel"], 2),
        "not_to_scale_angle_deg": cfg["not_to_scale"]["angle_deg"],
        "not_to_scale_length_pct": _r(100 * cfg["not_to_scale"]["rel_length"], 1),
        "calibration": {r["relation"]: {"method": r["method"], "valley_or_knee": _num(r["tolerance"]),
                                        "unit": r["unit"]} for r in read_csv(tables / "tolerance_calibration.csv")},
    }
    ft, au = cfg["finetune"], cfg["augment"]
    recipe = {"lp_queries": ft["lp_queries"], "epochs": ft["epochs"], "epoch_images": ft["epoch_images"],
              "total_images": ft["epochs"] * ft["epoch_images"], "batch_size": ft["batch_size"],
              "head_lr": ft["head_lr"], "backbone_lr": ft["backbone_lr"], "llrd": ft["llrd"],
              "weight_decay": ft["weight_decay"], "ema_decay": ft["ema_decay"], "amp_dtype": ft["amp_dtype"],
              "real_fraction": ft["real_fraction"], "augment": {"rotation_deg": au["rotation_deg"], "hflip": au["hflip"],
                                                               "vflip": au["vflip"], "scale": au["scale"]},
              "probe_epochs": cfg["probe"]["epochs"], "probe_lr": cfg["probe"]["lr"]}
    ov = cfg["overlay"]
    overlay = {"color": ov["color"], "width_px": ov["width_px"], "halo_width_px": ov["halo_width_px"],
               "halo_alpha": ov["halo_alpha"], "point_radius_px": ov["point_radius_px"], "long_side": L}
    bench = json.loads((runs / "benchmark_lab" / "benchmark.json").read_text()) \
        if (runs / "benchmark_lab" / "benchmark.json").exists() else None
    compute = None if bench is None else {
        "gpu": bench["gpu"], "n_gpus": bench["n_gpus"], "train_img_per_s_448": bench["rates"]["448"]["train_img_per_s"],
        "planned_gpu_hours": _r(bench["total_gpu_hours"], 1), "planned_wall_days": _r(bench["wall_hours"] / 24, 1)}
    backbones = {"main": backbone_facts(cfg["model"]["backbone"], L),
                 "demo": backbone_facts(demo_cfg["model"]["backbone"], L)}
    audit = {r["relation"]: {k: _num(v) for k, v in r.items() if k != "relation"}
             for r in read_csv(tables / "audit_geometry3k_figures.csv")}
    nts = {r["set"]: {k: _num(v) for k, v in r.items() if k != "set"} for r in read_csv(tables / "not_to_scale.csv")}
    gold = {f"{r['evaluation']}": {"fp_rate": _num(r["fp_rate"]), "fn_rate": _num(r["fn_rate"]), "n": _num(r["n"])}
            for r in read_csv(tables / "verifier_audit.csv") if r["relation"] == "ALL"}
    colour = {r["test"]: {"images": _num(r["images"]), "images_with": _num(r["images_with"]), "pct": _num(r["pct"]),
                          "min_pixels": _num(r.get("min_pixels")),
                          "radius": _num(r["test"].rpartition("_r")[2]) if r["test"].startswith("near_") else None}
              for r in read_csv(tables / "highlight_colour_check.csv")}
    meta = {k: v for k, v in site.items()}
    return {"meta": meta, "relations": vocab, "data": data, "tolerances": tolerances, "overlay": overlay,
            "input": {"long_side": L, "mean": cfg["input"]["mean"], "std": cfg["input"]["std"]},
            "recipe": recipe, "compute": compute, "backbones": backbones, "audit": audit, "not_to_scale": nts,
            "gold": gold, "colour_check": colour}


# ---------------------------------------------------------------- main


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return None


def export(out: Path, runs: Path, tables: Path, processed: Path, cfg: dict, site: dict,
           backbone_facts: Callable[[str, int], dict]) -> dict[str, dict]:
    vocab = vocabulary(cfg)
    docs = {"results": results(runs, tables, vocab), "scaling": scaling(runs, vocab, cfg),
            "pseudo_labels": pseudo_labels(runs, vocab),
            "facts": facts(cfg, site, runs, tables, processed, vocab, backbone_facts)}
    stamp = {"schema": SCHEMA_VERSION, "commit": _git_commit()}
    out.mkdir(parents=True, exist_ok=True)
    for name, doc in docs.items():
        (out / f"{name}.json").write_text(json.dumps({"_export": stamp, **doc}, indent=1, allow_nan=False) + "\n")
    return docs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default=None, help="runs directory (default: paths.runs of the base config)")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--online", action="store_true", help="allow fetching backbone configs from the hub")
    args = ap.parse_args(argv)
    cfg = load_config(REPO_ROOT / "configs" / "base.yaml")
    site = yaml.safe_load(SITE_YAML.read_text())
    runs = Path(args.runs) if args.runs else REPO_ROOT / cfg["paths"]["runs"]
    docs = export(Path(args.out), runs, REPO_ROOT / cfg["paths"]["tables"], REPO_ROOT / cfg["paths"]["processed"],
                  cfg, site, lambda name, L: hf_backbone_facts(name, L, args.online))
    r = docs["results"]
    print(f"{len(r['runs'])} runs, {len(r['experiments'])} experiments -> {args.out}")
    for e, d in r["experiments"].items():
        print(f"  {e:24s} seeds {d['seeds']}")
    for k, b in docs["facts"]["backbones"].items():
        print(f"  backbone {k}: {b['name']} " + (f"{b['params_M']}M params, {b['tokens']} tokens" if b["available"]
                                                 else f"UNAVAILABLE ({b['reason'][:80]})"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
