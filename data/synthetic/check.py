"""Verifier on synthetic data (plan §4.4).

    python -m data.synthetic.check     (after python -m verifier.label --sets synth_train synth_extrap)

Per pool and relation:
  exact_max_margin   the injected relations measured on the exact, pre-jitter coordinates
                     (meta.exact_points): ~0 if the constructions are correct
  injected_pass      share of injected (asserted) queries the verifier confirms after jitter
  random_pass        share of non-injected candidate queries the verifier calls true (accidental
                     relations; should be small, except where geometry makes them common)
Line relations are counted once per carrier pair, as in the audit. Writes
reports/tables/synthetic_verifier_check.csv and reports/figures/synthetic_examples.png (20 rendered
diagrams: 15 train, 5 extrapolation).
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

from data.schema import Circle, Diagram, read_jsonl  # noqa: E402
from experiments.run import REPO_ROOT, load_config  # noqa: E402
from verifier.audit import grouped  # noqa: E402
from verifier.checks import evaluate  # noqa: E402

POOLS = ("synth_train", "synth_extrap")


def exact_copy(d: Diagram) -> Diagram:
    ex = Diagram.from_dict(d.to_dict())
    ex.points = {n: tuple(v) for n, v in d.meta["exact_points"].items()}
    ex.circles = [Circle(c.center, tuple(e[:2]), e[2], c.points_on) for c, e in zip(d.circles, d.meta["exact_circles"])]
    return ex


def exact_margins(diagrams) -> dict[str, float]:
    worst = defaultdict(float)
    for d in diagrams:
        ex = exact_copy(d)
        for r in d.asserted:
            if r.type in ("midpoint", "perp_bisector", "equal_angle"):
                continue
            _, m = evaluate(ex, r.type, r.args)
            if not math.isnan(m):
                worst[r.type] = max(worst[r.type], m)
    return dict(worst)


def rates(records) -> dict[str, dict]:
    out = defaultdict(lambda: {"injected": 0, "injected_pass": 0, "random": 0, "random_pass": 0})
    for r in records:
        c = out[r["relation"]]
        k = "injected" if r["asserted"] else "random"
        c[k] += 1
        c[k + "_pass"] += int(r["geometric"])
    return out


def examples_figure(cfg, out) -> None:
    raw, proc = REPO_ROOT / cfg["paths"]["raw"], REPO_ROOT / cfg["paths"]["processed"]
    picks = []
    for pool, k in (("synth_train", 15), ("synth_extrap", 5)):
        for i, d in enumerate(read_jsonl(proc / f"{pool}.jsonl")):
            if i >= k:
                break
            picks.append(d)
    fig, axes = plt.subplots(4, 5, figsize=(17, 14))
    for ax, d in zip(axes.flat, picks):
        with Image.open(raw / d.image_path) as im:
            ax.imshow(im.convert("RGB"))
        made = ", ".join(f"{k}×{v}" if v > 1 else k for k, v in sorted(d.meta["constructions"].items())) or "random only"
        ax.set_title(f"{d.id.split(':')[1]}: {len(d.points)} pts\n{made}", fontsize=8)
        ax.axis("off")
    fig.suptitle("Synthetic diagrams (rows 1-3: training pool; row 4: extrapolation pool, 12-20 points). "
                 "Titles list the injected constructions.", fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(out, dpi=95)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "base.yaml"))
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    proc = REPO_ROOT / cfg["paths"]["processed"]
    rows = []
    for pool in POOLS:
        diagrams = list(read_jsonl(proc / f"{pool}.jsonl"))
        worst = exact_margins(diagrams)
        for rel, c in sorted(rates(grouped(proc / "queries" / f"{pool}.csv.gz", cfg["tolerances"])).items()):
            rows.append({"pool": pool, "relation": rel, "exact_max_margin": f"{worst.get(rel, 0.0):.2e}",
                         "injected": c["injected"], "injected_pass": f"{c['injected_pass'] / max(c['injected'], 1):.3f}",
                         "random": c["random"], "random_pass": f"{c['random_pass'] / max(c['random'], 1):.3f}"})
    out = REPO_ROOT / cfg["paths"]["tables"] / "synthetic_verifier_check.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)
    examples_figure(cfg, REPO_ROOT / cfg["paths"]["figures"] / "synthetic_examples.png")
    print(f"-> {out}, reports/figures/synthetic_examples.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
