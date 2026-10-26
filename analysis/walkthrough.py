"""Plan 10.3: one diagram end to end, for notebooks/walkthrough.ipynb (the notebook only calls this).

    from analysis import walkthrough as W
    run = W.best_checkpoint()                      # best DINOv3 checkpoint by validation macro-F1
    model, thresholds, device = W.load(run)
    ds = W.unit("geometry3k_test")
    diagram_id = W.pick_diagram(ds)                # or any diagram id of the unit
    results = W.read_diagram(model, ds, diagram_id, thresholds, device, run=run)
    W.decision_table(results)                      # every applicable decision: model, ruler, marks
    W.show_query(results[k])                       # original, the model's view, attention map

For each query of the diagram: the overlay is drawn exactly as the square-padded evaluation drew it
(analysis.attention_maps.model_view), the model gives a probability per relation and the decision at
the run's validation-tuned threshold, the verifier re-measures the relation from the annotated
coordinates (verifier.checks.evaluate, the fixed tolerances of configs/base.yaml) and reports its
verdict and margin, and the last-layer CLS attention is mapped onto the patches. The probability is
compared with the stored square-padded evaluation of the same query (`p_eval`), which checks that the
notebook reproduces the numbers in the report.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from analysis import attention_maps as AM
from data.overlay import flatten
from data.queries import vocabulary
from experiments.run import REPO_ROOT, get_device, load_config

CFG_PATH = REPO_ROOT / "configs" / "base.yaml"


def config() -> dict:
    return load_config(CFG_PATH)


def runs_dir(cfg: dict | None = None) -> Path:
    cfg = cfg or config()
    p = Path(cfg["paths"]["runs"])
    return p if p.is_absolute() else REPO_ROOT / p


def best_checkpoint(runs: Path | None = None, backbone: str | None = "config") -> str:
    """The run with a saved checkpoint and the highest validation macro-F1 (the training's own
    selection metric; test sets are not consulted). `backbone="config"`: only runs of the study's
    backbone (DINOv3); None: any backbone (the DINOv2-L ablation would then win)."""
    cfg = config()
    runs = runs or runs_dir(cfg)
    want = cfg["model"]["backbone"] if backbone == "config" else backbone
    best, best_f1 = None, -1.0
    for ck in sorted(runs.glob("*/checkpoints/best.pt")):
        run = ck.parent.parent
        mpath = run / "metrics.json"
        if run.name.startswith("smoke") or not mpath.exists():
            continue
        m = json.loads(mpath.read_text())
        if m.get("status") != "ok" or (want and m.get("backbone_loaded") != want):
            continue
        if float(m.get("val_macro_f1", -1)) > best_f1:
            best, best_f1 = run.name, float(m["val_macro_f1"])
    if best is None:
        raise FileNotFoundError(f"no finished run with checkpoints/best.pt under {runs}")
    return best


def load(run: str, device: str = "auto"):
    """Model with eager attention, the run's validation-tuned thresholds (from its sq_ evaluation when
    there is one, as in the report), and the device."""
    from analysis.phase8 import run_thresholds
    cfg = config()
    runs = runs_dir(cfg)
    dev = get_device(device)
    model, _ = AM.load_model(cfg, runs / run / "checkpoints" / "best.pt", dev)
    src = f"sq_{run}" if (runs / f"sq_{run}" / "predictions" / "val.npz").exists() else run
    return model, run_thresholds(runs, src, vocabulary(cfg)), dev


def unit(name: str = "geometry3k_test"):
    from data.dataset import QueryDataset
    return QueryDataset(config(), name, train=False)


def pick_diagram(ds, min_queries: int = 6, max_queries: int = 24) -> str:
    """A small diagram that shows the most, with lettered points (readable in a notebook): the most
    distinct relations true by the verifier, then the most relations true but unmarked, then the fewest
    queries; ties broken by id. Deterministic."""
    by: dict[str, list[dict]] = {}
    for r in ds.rows:
        by.setdefault(r["diagram_id"], []).append(r)

    def lettered(did: str) -> bool:
        return all(not str(n).startswith("_") for n in ds.diagrams[did].points)

    def score(item):
        did, rows = item
        geo = np.array([r["geo"] for r in rows])
        asr = np.array([r["asr"] for r in rows])
        return (int((geo.sum(0) > 0).sum()), int(((geo == 1) & (asr == 0)).sum()), -len(rows), did)

    cands = [(d, rs) for d, rs in by.items() if min_queries <= len(rs) <= max_queries and lettered(d)]
    return max(cands or list(by.items()), key=score)[0]


def verdict(d, rel: str, refs: list, tol: dict) -> tuple[bool | None, float, float]:
    """The verifier's verdict for one relation of one query, its margin and the margin in units of the
    tolerance (<= 1 means the relation holds)."""
    from verifier.checks import TOL_KEYS, _sin, evaluate
    from verifier.primitives import bbox_diagonal
    ok, m = evaluate(d, rel, refs, tol, scale=bbox_diagonal(d.points.values()))
    t = tol.get(TOL_KEYS[rel])
    if rel in ("parallel", "perpendicular"):
        t = _sin(t)
    ratio = float(m) / float(t) if (t and m == m) else float("nan")
    return ok, float(m), ratio


def read_diagram(model, ds, diagram_id: str, thresholds: np.ndarray, device, run: str | None = None) -> list[dict]:
    """Every query of the diagram through the model, the verifier and the attention map."""
    cfg = config()
    vocab = vocabulary(cfg)
    d = ds.diagrams[diagram_id]
    idx = [i for i, r in enumerate(ds.rows) if r["diagram_id"] == diagram_id]
    stored = None
    if run is not None:
        p = runs_dir(cfg) / f"sq_{run}" / "predictions" / f"{ds.unit}.npz"
        if p.exists():
            z = np.load(p)
            if len(z["qid"]) == len(ds.rows):
                stored = z["probs"]
    out = []
    for i in idx:
        row = ds.rows[i]
        a = AM.analyse_query(model, ds, row, cfg, device)
        decisions = []
        for j, rel in enumerate(vocab):
            if not row["mask"][j]:
                continue
            ok, margin, ratio = verdict(d, rel, row["refs"], cfg["tolerances"])
            p = float(a["probs"][j])
            decisions.append({"relation": rel, "p": round(p, 3), "threshold": round(float(thresholds[j]), 2),
                              "model": bool(p >= thresholds[j]), "verifier": ok, "margin_over_tol": round(ratio, 2),
                              "marked": bool(row["asr"][j]), "stored_geometric": bool(row["geo"][j]),
                              "p_eval": round(float(stored[i, j]), 3) if stored is not None else None})
        out.append({"index": i, "row": row, "diagram": d, "decisions": decisions, **a})
    return out


def decision_table(results: list[dict]):
    """One row per applicable decision: the query, the model, the ruler and the marks."""
    import pandas as pd
    recs = []
    for k, r in enumerate(results):
        q = " ".join("".join(map(str, ref[1:])) if ref[0] != "circle" else f"circle{ref[1]}" for ref in r["row"]["refs"])
        for dec in r["decisions"]:
            recs.append({"query": k, "primitives": q, **{c: dec[c] for c in
                         ("relation", "p", "threshold", "model", "verifier", "margin_over_tol", "marked", "p_eval")},
                         "agrees_with_verifier": dec["model"] == bool(dec["verifier"]),
                         "highlight_attention": round(r["stats_last"]["share_highlight"], 2)})
    return pd.DataFrame(recs)


def original_image(ds, d) -> Image.Image:
    with Image.open(ds.raw / d.image_path) as im:
        return flatten(im).copy()


def show_query(result: dict, ds, title: str = ""):
    """Three panels: the original diagram, the model's view (the highlighted query), the last-layer CLS
    attention over that view. Returns the matplotlib figure."""
    import matplotlib.pyplot as plt
    L = ds.L
    square = Image.new("RGB", (L, L), (255, 255, 255))
    square.paste(result["view"], (0, 0))
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.9))
    axes[0].imshow(original_image(ds, result["diagram"]))
    axes[0].set_title("the diagram", fontsize=9)
    axes[1].imshow(square)
    axes[1].set_title("the model's view: the query highlighted", fontsize=9)
    AM._heat(axes[2], square, result["maps"]["last_layer"], L,
             f"last-layer CLS attention ({result['stats_last']['share_highlight']:.0%} on the highlight)")
    for ax in axes[:2]:
        ax.axis("off")
    lines = [f"{dec['relation']}: model p={dec['p']:.2f} -> {'yes' if dec['model'] else 'no'}; "
             f"ruler {'yes' if dec['verifier'] else 'no'} (margin {dec['margin_over_tol']}x tol); "
             f"marked {'yes' if dec['marked'] else 'no'}" for dec in result["decisions"]]
    head = (title + "\n" if title else "") + "\n".join(lines)
    fig.suptitle(head, fontsize=8.5, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.045 * (head.count("\n") + 1)))
    return fig
