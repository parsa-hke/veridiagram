"""Generate and render the synthetic pools (plan §4.3).

    python -m data.synthetic.build [--pools train extrap unlabelled] [--limit N] [--workers K]

Pools (sizes from config `synthetic`):
  train       max(pool_sizes) = 50k diagrams, n_points/n_segments ranges. The Part A sizes 1k / 10k /
              50k are nested prefixes: data/processed/splits/synth_train_{1k,10k,50k}.txt.
  extrap      `extrap_size` diagrams with extrap_points (12-20) / extrap_segments (8-16):
              1.5-2x the training maximum.
  unlabelled  `unlabelled_size` diagrams. data/processed/synth_unlabelled.jsonl holds images only
              (no points, segments, circles or labels); the full annotation is sealed in
              data/processed/sealed/synth_unlabelled.jsonl with its sha256 in sealed/SEAL.txt, for
              post-hoc analysis in Part B only.

Images: <raw>/synthetic/<pool>/<index>.png, or .jpg (quality 95) for noisy styles (gitignored,
regenerated bit-for-bit from the seed).
Annotations: data/processed/synth_<pool>.jsonl.
"""

from __future__ import annotations

import argparse
import hashlib
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

from data.schema import Diagram, write_jsonl
from data.synthetic.generator import generate_one
from data.synthetic.render import image_suffix, render, sample_style_for
from experiments.run import REPO_ROOT, load_config


def _one(index: int, pool: str, scfg: dict, raw: str, ranges: dict) -> dict:
    d = generate_one(scfg, pool, index, scfg["seed"], **ranges)
    d.image_path = d.image_path.rsplit(".", 1)[0] + image_suffix(sample_style_for(scfg["render"], d.meta["seed"]))
    d.meta["style"] = render(d, Path(raw) / d.image_path, scfg["render"], d.meta["seed"])
    return d.to_dict()


def build_pool(cfg: dict, pool: str, n: int, workers: int) -> list[Diagram]:
    scfg = cfg["synthetic"]
    raw = REPO_ROOT / cfg["paths"]["raw"]
    ranges = ({"n_points": tuple(scfg["extrap_points"]), "n_segments": tuple(scfg["extrap_segments"])}
              if pool == "extrap" else {})
    fn = partial(_one, pool=pool, scfg=scfg, raw=str(raw), ranges=ranges)
    if workers > 1:
        with ProcessPoolExecutor(workers) as ex:
            dicts = list(ex.map(fn, range(n), chunksize=64))
    else:
        dicts = [fn(i) for i in range(n)]
    return [Diagram.from_dict(x) for x in dicts]


def seal(diagrams: list[Diagram], proc: Path) -> str:
    sealed = proc / "sealed" / "synth_unlabelled.jsonl"
    write_jsonl(diagrams, sealed)
    digest = hashlib.sha256(sealed.read_bytes()).hexdigest()
    (proc / "sealed" / "SEAL.txt").write_text(
        f"synth_unlabelled.jsonl sha256 {digest}\n"
        "Sealed annotation of the synth_unlabelled pool. Do not read before Part B's post-hoc analysis.\n")
    public = [Diagram(id=d.id, source=d.source, split=d.split, image_path=d.image_path, width=d.width,
                      height=d.height, flags={"sealed": True}, meta={"style": d.meta.get("style", {})})
              for d in diagrams]
    write_jsonl(public, proc / "synth_unlabelled.jsonl")
    return digest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "base.yaml"))
    ap.add_argument("--pools", nargs="+", default=["train", "extrap", "unlabelled"],
                    choices=["train", "extrap", "unlabelled"])
    ap.add_argument("--limit", type=int, default=None, help="cap every pool at N diagrams (smoke)")
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    scfg = cfg["synthetic"]
    proc = REPO_ROOT / cfg["paths"]["processed"]
    workers = args.workers or scfg["workers"]
    sizes = {"train": max(scfg["pool_sizes"]), "extrap": scfg["extrap_size"], "unlabelled": scfg["unlabelled_size"]}
    for pool in args.pools:
        n = min(sizes[pool], args.limit) if args.limit else sizes[pool]
        t0 = time.time()
        ds = build_pool(cfg, pool, n, workers)
        if pool == "unlabelled":
            digest = seal(ds, proc)
            print(f"unlabelled: {n} diagrams, annotation sealed (sha256 {digest[:16]}...)", end="")
        else:
            write_jsonl(ds, proc / f"synth_{pool}.jsonl")
            print(f"{pool}: {n} diagrams -> {proc / f'synth_{pool}.jsonl'}", end="")
        if pool == "train":
            split_dir = REPO_ROOT / cfg["paths"]["splits"]
            split_dir.mkdir(parents=True, exist_ok=True)
            for size in scfg["pool_sizes"]:
                if size <= n:
                    (split_dir / f"synth_train_{size // 1000}k.txt").write_text("".join(d.id + "\n" for d in ds[:size]))
        print(f" ({time.time() - t0:.0f}s, {workers} workers)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
