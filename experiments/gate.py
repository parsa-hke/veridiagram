"""Phase gates as checks on the run records.

    python -m experiments.gate smoke [--config X]    # plan §6.6

smoke: the probe and fine-tune smoke runs (runs/smoke_probe_s<seed>, runs/smoke_finetune_s<seed>)
must have status ok, have loaded the PRIMARY backbone (`model.backbone`, not the fallback),
finished in under 10 minutes each (`duration_s`), and written metrics.json, predictions for every
evaluation unit, and for the fine-tune checkpoints/best.pt. Prints one line per check and PASS/FAIL;
exit code 0 only on PASS.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from data.queries import UNITS
from experiments.run import REPO_ROOT, load_config

LIMIT_S = 600


def check_smoke(cfg: dict, seed: int = 0) -> list[tuple[str, bool, str]]:
    runs = Path(cfg["paths"]["runs"])
    runs = runs if runs.is_absolute() else REPO_ROOT / runs
    primary = cfg["model"]["backbone"]
    out = []
    for mode in ("probe", "finetune"):
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
        if mode == "finetune":
            out.append(("finetune: best.pt", (d / "checkpoints" / "best.pt").exists(), str(d / "checkpoints")))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gate", choices=["smoke"])
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "base.yaml"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    checks = check_smoke(load_config(args.config), args.seed)
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    passed = all(ok for _, ok, _ in checks)
    print(f"GATE 6.6 (smoke): {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
