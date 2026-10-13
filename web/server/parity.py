"""Parity of the website's answers with the run's saved predictions (WEBSITE_PLAN.md §4.1).

    python -m web.server.parity [--n 600] [--device auto] [--modes single,square,batch]

On an evenly spaced sample of validation queries, compares the demo model's probabilities with
runs/<run>/predictions/val.npz under three ways of asking:
  batch   as evaluation did: consecutive rows in batches of eval.batch_size, each padded with
          normalised white to the batch's largest image (data.dataset.pad_batch)
  single  one query at a time, padded only to the patch multiple (what QueryDataset returns)
  square  one query at a time, padded with white to long_side x long_side
The pooled feature averages every patch token, padding included, so an answer can depend on how
much padding the query was given; this measures by how much. Prints, per mode: max and mean |dp| over
applicable entries, decisions (p > 0.5) differing, and macro-F1 against the geometric labels next
to the saved predictions' macro-F1 on the same sample.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from analysis.metrics import evaluate
from data.dataset import QueryDataset
from experiments.run import REPO_ROOT, load_config
from web.server import core
from web.server.core import pad_square as square


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "ablation_dinov2.yaml"))
    ap.add_argument("--run", default="ablation_dinov2_s0")
    ap.add_argument("--n", type=int, default=600, help="queries (evenly spaced over val)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--modes", default="batch,single,square")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    e = core.Engine.from_checkpoint(cfg, REPO_ROOT / "runs" / args.run / "checkpoints" / "best.pt", args.device, args.run)
    e.pad = "patch"                             # the modes below pad the dataset's tensors themselves
    ds = QueryDataset(cfg, "val", train=False)
    z = np.load(REPO_ROOT / "runs" / args.run / "predictions" / "val.npz")
    assert (z["qid"] == np.array([r["qid"] for r in ds.rows])).all(), "val predictions do not match the val rows"
    B = cfg["eval"]["batch_size"]
    if "batch" in args.modes:                   # whole evaluation batches, so the padding is the original one
        nb = max(1, args.n // B)
        batches = np.linspace(0, (len(ds) - 1) // B, nb).astype(int)
        batch_idx = np.concatenate([np.arange(b * B, min((b + 1) * B, len(ds))) for b in batches])
    idx = np.linspace(0, len(ds) - 1, args.n).astype(int)
    vocab = e.vocab
    print(f"{args.run} on {e.device}; saved predictions: {len(z['qid'])} val queries; eval batch {B}")
    for mode in args.modes.split(","):
        t0 = time.time()
        if mode == "batch":
            sel = batch_idx
            e.batch = B
            probs = np.concatenate([e.predict([ds[j][0] for j in range(b * B, min((b + 1) * B, len(ds)))]) for b in batches])
        else:
            sel = idx
            e.batch = 1
            pre = (lambda x: square(x, e.L, e.white)) if mode == "square" else (lambda x: x)
            probs = np.concatenate([e.predict([pre(ds[i][0])]) for i in sel])    # dataset tensors, padded per mode
        saved, m = z["probs"][sel], z["mask"][sel].astype(bool)
        diff = np.abs(probs - saved)[m]
        flips = int(((probs > 0.5) != (saved > 0.5))[m].sum())
        f_new = evaluate(probs, z["geo"][sel], z["mask"][sel], vocab)["macro_f1"]
        f_old = evaluate(saved, z["geo"][sel], z["mask"][sel], vocab)["macro_f1"]
        print(f"  {mode:6s} n={len(sel):4d}  max |dp| {diff.max():.4f}  mean |dp| {diff.mean():.4f}  "
              f"decisions differing {flips} of {int(m.sum())} ({100 * flips / max(int(m.sum()), 1):.2f}%)  "
              f"macro-F1 {f_new:.4f} (saved {f_old:.4f})  [{time.time() - t0:.0f}s]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
