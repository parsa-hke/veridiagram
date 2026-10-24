"""Phase gates as checks on the run records.

    python -m experiments.gate smoke [--config X]    # plan §6.6
    python -m experiments.gate phase7 [--config X]   # smoke of the Phase 7 modes added later (7.8, Part B)
    python -m experiments.gate eval_square [--config X]   # smoke of the square-padded re-evaluation

smoke: the probe and fine-tune smoke runs (runs/smoke_probe_s<seed>, runs/smoke_finetune_s<seed>)
must have status ok, have loaded the PRIMARY backbone (`model.backbone`, not the fallback),
finished in under 10 minutes each (`duration_s`), and written metrics.json, predictions for every
evaluation unit, and for the fine-tune checkpoints/best.pt. Prints one line per check and PASS/FAIL;
exit code 0 only on PASS. phase7: the same checks for runs/smoke_ablation_nooverlay_s<seed> and
runs/smoke_selftrain_consistency_s<seed>, plus pseudo-labels produced by the self-training smoke.
eval_square: runs/smoke_sq_finetune_s<seed> and runs/smoke_sq_probe_s<seed> (configs/eval_square.yaml
on the smoke fine-tune and probe of gate 6.6): the same checks, square padding, no checkpoint of their own.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from data.queries import UNITS
from experiments.run import REPO_ROOT, load_config

LIMIT_S = 600
GATES = {"smoke": ("GATE 6.6 (smoke)", ("probe", "finetune")),
         "phase7": ("GATE 7 (smoke of the 7.8 and Part B modes)", ("ablation_nooverlay", "selftrain_consistency")),
         "eval_square": ("GATE 7 (smoke of the square-padded re-evaluation)", ("sq_finetune", "sq_probe"))}


def check_smoke(cfg: dict, seed: int = 0, names=GATES["smoke"][1]) -> list[tuple[str, bool, str]]:
    runs = Path(cfg["paths"]["runs"])
    runs = runs if runs.is_absolute() else REPO_ROOT / runs
    primary = cfg["model"]["backbone"]
    out = []
    for mode in names:
        d = runs / f"smoke_{mode}_s{seed}"
        mpath = d / "metrics.json"
        if not mpath.exists():
            out.append((f"{mode}: metrics.json", False, f"missing at {mpath}"))
            continue
        m = json.loads(mpath.read_text())
        out.append((f"{mode}: status", m.get("status") == "ok", str(m.get("status"))))
        loaded = m.get("backbone_loaded")
        out.append((f"{mode}: primary backbone", loaded == primary, f"loaded {loaded}"))
        dur = float(m.get("duration_s", 1e9))
        out.append((f"{mode}: under {LIMIT_S // 60} min", dur < LIMIT_S, f"{dur:.0f} s"))
        missing = [u for u in UNITS if u != "train" and not (d / "predictions" / f"{u}.npz").exists()]
        out.append((f"{mode}: predictions", not missing, "all units" if not missing else f"missing {missing}"))
        if mode.startswith("sq_"):
            out.append((f"{mode}: square padding", m.get("pad") == "square", f"pad {m.get('pad')}, source {m.get('source_run')}"))
        elif mode != "probe":
            out.append((f"{mode}: best.pt", (d / "checkpoints" / "best.pt").exists(), str(d / "checkpoints")))
        if mode.startswith("selftrain"):
            n = (m.get("pseudo_labels") or {}).get("train_rows", 0)
            out.append((f"{mode}: pseudo-labels", n > 0, f"{n} pseudo-labelled training queries"))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gate", choices=list(GATES))
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "base.yaml"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    title, names = GATES[args.gate]
    checks = check_smoke(load_config(args.config), args.seed, names)
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    passed = all(ok for _, ok, _ in checks)
    print(f"{title}: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
