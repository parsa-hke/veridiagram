"""One command from raw archives to model-ready data (plan §10.1).

    python -m data.prepare [--config X] [--download] [--synthetic train extrap unlabelled | none]

Steps, each an existing CLI: (download) -> data.build -> data.overlap -> data.splits ->
data.synthetic.build -> verifier.label (geometric labels in every processed set) ->
data.queries (query indexes for the config's units). Tolerances are read from the config; they are
not re-calibrated here (plan §0.2 #5 fixes them).
"""

from __future__ import annotations

import argparse
import time

from experiments.run import REPO_ROOT


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "base.yaml"))
    ap.add_argument("--download", action="store_true", help="download and verify the raw archives first")
    ap.add_argument("--synthetic", nargs="+", default=["train", "extrap", "unlabelled"],
                    help="synthetic pools to build, or 'none'")
    args = ap.parse_args(argv)
    c = ["--config", args.config]

    from data import build, census, download, overlap, queries, splits
    from data.synthetic import build as synth_build
    from verifier import label

    steps = []
    if args.download:
        steps.append(("download", lambda: download.main(c)))
    steps += [("build", lambda: build.main(c)), ("overlap", lambda: overlap.main(c)),
              ("splits", lambda: splits.main(c)), ("census", lambda: census.main(c))]
    pools = [] if args.synthetic == ["none"] else args.synthetic
    if pools:
        steps.append(("synthetic", lambda: synth_build.main(c + ["--pools", *pools])))
    steps += [("label", lambda: label.main(c)), ("queries", lambda: queries.main(c))]
    for name, fn in steps:
        t0 = time.time()
        print(f"\n=== {name} ===", flush=True)
        fn()
        print(f"=== {name} done in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
