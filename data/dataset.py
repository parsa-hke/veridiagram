"""torch Dataset over (image, query) -> (image, label_geometric, label_asserted, meta) (plan §5.6).

Per item: the query's diagram image is flattened on white, resized so its long side is
`input.long_side` (448), and the queried primitives are drawn on it (data.overlay); this
overlaid image is cached as PNG at <cache>/overlays/<settings hash>/<set>/<diagram>/<query id>.png
(the hash covers the overlay settings and the long side, so changing them never reuses stale images). Training
items are then augmented (data.augment) and re-fitted to the long side. Finally the image is padded
on the right/bottom with white to a multiple of `input.pad_to_multiple` (the patch size) and
normalised with `input.mean` / `input.std`. `collate` pads a batch to its largest height and width
with normalised white, so batches are (B, 3, H, W) with H, W <= 448 and multiples of 16; with
`square` (evaluation under `eval.pad: square`, plan §0.2 #10 note) every image is padded to
448 x 448 instead, so its answer does not depend on its batch neighbours.

meta per item: qid, diagram_id, set, split, kind, relation mask (applicability), refs, and with
`model.query_input: tokens` the query as token indices (data.query_tokens, plan §7.8).

`overlay.enabled: false` (plan §7.8 ablation) skips the highlight: every query of a diagram then sees
the same image. `data.extra_train_index` (Part B) appends the rows of another index file (the
pseudo-labelled PGPS9K queries, same row format) to the training unit.

    python -m data.dataset --config configs/smoke.yaml [--examples]   # plan §5.8 gate: build + load one batch
"""

from __future__ import annotations

import argparse
import json
import os
import zlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from data.augment import build_augment
from data.overlay import draw_overlay, flatten
from data.queries import UNITS, build_index, index_path, vocabulary
from data.query_tokens import encode as encode_query
from data.schema import read_jsonl
from experiments.run import REPO_ROOT, load_config


def replace_or_discard(tmp: Path, path: Path) -> None:
    """os.replace(tmp, path); if that fails because another process holds the target (Windows) and the
    target already exists, keep the existing file and drop tmp."""
    try:
        os.replace(tmp, path)
    except OSError:
        if not path.exists():
            raise
        tmp.unlink(missing_ok=True)


def fit_long_side(im: Image.Image, L: int) -> tuple[Image.Image, float]:
    k = L / max(im.size)
    return im.resize((max(1, round(im.width * k)), max(1, round(im.height * k))), Image.Resampling.BICUBIC), k


def pad_to_multiple(im: Image.Image, m: int) -> Image.Image:
    w, h = (-(-im.width // m)) * m, (-(-im.height // m)) * m
    if (w, h) == im.size:
        return im
    out = Image.new("RGB", (w, h), (255, 255, 255))
    out.paste(im, (0, 0))
    return out


def overlay_image(im: Image.Image, d, refs: list, L: int, ocfg: dict, highlight: bool = True) -> Image.Image:
    """The model's view of a query before normalisation: the diagram flattened on white, its long side
    resized to L, the queried primitives drawn on it. Shared by QueryDataset and the website's
    inference server (web/server), so both feed the model the same pixels (WEBSITE_PLAN.md §4.1)."""
    img, k = fit_long_side(flatten(im), L)
    if highlight:
        img, _ = draw_overlay(img, d, refs, ocfg, scale=k)
    return img


def normalise(im: Image.Image, patch: int, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Pad right/bottom with white to a multiple of the patch size, scale to [0, 1], normalise: (3, H, W)."""
    a = torch.from_numpy(np.asarray(pad_to_multiple(im, patch), dtype=np.float32) / 255.0).permute(2, 0, 1)
    return (a - mean) / std


class QueryDataset(torch.utils.data.Dataset):
    def __init__(self, cfg: dict, unit: str, train: bool | None = None, build: bool = False):
        self.cfg = cfg
        self.unit = unit
        self.train = (unit == "train") if train is None else train
        self.vocab = vocabulary(cfg)
        path = index_path(cfg, unit)
        if build or not path.exists():
            build_index(cfg, unit)
        with open(path) as f:
            self.rows = [json.loads(line) for line in f]
        cap = cfg["data"].get("max_queries_per_unit")
        if not self.train:                                   # evaluation cap per unit (plan §0.2 #10)
            ecap = (cfg.get("eval", {}).get("max_queries") or {}).get(unit)
            cap = min(c for c in (cap, ecap) if c) if (cap or ecap) else None
        if cap and len(self.rows) > cap:                     # evenly spaced subsample
            step = len(self.rows) / cap
            self.rows = [self.rows[int(i * step)] for i in range(cap)]
        extra = cfg["data"].get("extra_train_index") if unit == "train" else None
        if extra:                                            # Part B: pseudo-labelled pool queries
            with open(extra) as f:
                self.rows += [json.loads(line) for line in f]
        proc = REPO_ROOT / cfg["paths"]["processed"]
        need: dict[str, set] = {}
        for r in self.rows:
            need.setdefault(r["set"], set()).add(r["diagram_id"])
        self.diagrams = {}
        for s, ids in need.items():
            for d in read_jsonl(proc / f"{s}.jsonl"):
                if d.id in ids:
                    self.diagrams[d.id] = d
        self.raw = REPO_ROOT / cfg["paths"]["raw"]
        key = json.dumps({"overlay": cfg["overlay"], "L": cfg["input"]["long_side"]}, sort_keys=True)
        self.cache_tag = format(zlib.crc32(key.encode()), "08x")      # new overlay settings -> new cache dir
        self.cache = (REPO_ROOT / cfg["paths"]["cache"] / "overlays" / self.cache_tag
                      if cfg["data"].get("cache_overlays", True) else None)
        self.L = cfg["input"]["long_side"]
        self.highlight = cfg["overlay"].get("enabled", True)
        self.tokens = cfg["model"].get("query_input", "overlay") == "tokens"
        self.patch = cfg["input"]["pad_to_multiple"]
        self.mean = torch.tensor(cfg["input"]["mean"]).view(3, 1, 1)
        self.std = torch.tensor(cfg["input"]["std"]).view(3, 1, 1)
        self.augment = build_augment(cfg) if self.train else None
        self.seed = cfg.get("seed", 0)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.rows)

    def overlaid(self, r: dict) -> Image.Image:
        """Resized, overlaid image of a query (cached)."""
        d = self.diagrams[r["diagram_id"]]
        path = None
        if self.cache is not None:
            path = self.cache / r["set"] / d.id.replace(":", "_") / f"{r['qid']}.png"
            if path.exists():
                return Image.open(path).convert("RGB")
        with Image.open(self.raw / d.image_path) as im:
            img = overlay_image(im, d, r["refs"], self.L, self.cfg["overlay"], self.highlight)
        if path is not None:                                  # atomic: parallel runs may share the cache
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(f".{path.stem}.{os.getpid()}.png")
            img.save(tmp)
            replace_or_discard(tmp, path)
        return img

    def to_tensor(self, im: Image.Image) -> torch.Tensor:
        return normalise(im, self.patch, self.mean, self.std)

    def __getitem__(self, i: int):
        r = self.rows[i]
        im = self.overlaid(r)
        if self.augment is not None:
            rng = np.random.default_rng([self.seed, self.epoch, i, zlib.crc32(r["qid"].encode())])
            im = self.augment(im, rng)
            if max(im.size) > self.L:
                im, _ = fit_long_side(im, self.L)
        meta = {"qid": r["qid"], "diagram_id": r["diagram_id"], "set": r["set"], "split": r["split"],
                "kind": r["kind"], "mask": torch.tensor(r["mask"], dtype=torch.float32), "refs": r["refs"]}
        if self.tokens:
            d = self.diagrams[r["diagram_id"]]
            meta["qtok"] = encode_query(r["kind"], r["refs"], {i: c.center for i, c in enumerate(d.circles)})
        return (self.to_tensor(im), torch.tensor(r["geo"], dtype=torch.float32),
                torch.tensor(r["asr"], dtype=torch.float32), meta)

    def auxiliary(self) -> np.ndarray:
        """Per row: True for queries outside the labelled real sources of the unit (synthetic or
        pseudo-labelled); `finetune.real_fraction` balances the two groups."""
        real = {s for s, _ in UNITS.get(self.unit, [])}
        return np.array([r["set"] not in real for r in self.rows], dtype=bool)

    def white(self) -> torch.Tensor:
        return ((torch.ones(3, 1, 1) - self.mean) / self.std).view(3)


def pad_batch(imgs, white: torch.Tensor, square: int | None = None) -> torch.Tensor:
    """Stack (3, H, W) tensors, padding each to the batch's largest height and width with normalised
    white (the batches of evaluation and of the website's server), or with `square` to square x square."""
    H = max(x.shape[1] for x in imgs)
    W = max(x.shape[2] for x in imgs)
    if square:
        if max(H, W) > square:
            raise ValueError(f"image {H}x{W} larger than the square {square}")
        H = W = square
    out = white.view(1, 3, 1, 1).repeat(len(imgs), 1, H, W)
    for k, x in enumerate(imgs):
        out[k, :, : x.shape[1], : x.shape[2]] = x
    return out


class Collate:
    """Pads a batch to its largest height/width (or to `square` x `square`) with normalised white. A
    class, not a closure, so it can be pickled into DataLoader worker processes on Windows (spawn)."""

    def __init__(self, white: torch.Tensor, square: int | None = None):
        self.white = white
        self.square = square

    def __call__(self, batch):
        imgs, geo, asr, metas = zip(*batch)
        out = pad_batch(imgs, self.white, self.square)
        meta = {"mask": torch.stack([m["mask"] for m in metas]),
                **{key: [m[key] for m in metas] for key in ("qid", "diagram_id", "set", "split", "kind", "refs")}}
        if "qtok" in metas[0]:
            meta["qtok"] = torch.stack([m["qtok"] for m in metas])
        return out, torch.stack(geo), torch.stack(asr), meta


def make_collate(white: torch.Tensor, square: int | None = None) -> Collate:
    return Collate(white, square)


def eval_square(cfg: dict) -> int | None:
    """`eval.pad`: batch (pad to the batch's largest image; every Phase 7 run) or square (every query
    to long_side x long_side). Returns the square side, or None for batch padding."""
    pad = cfg.get("eval", {}).get("pad", "batch")
    if pad not in ("batch", "square"):
        raise ValueError(f"eval.pad must be batch | square, got {pad!r}")
    return cfg["input"]["long_side"] if pad == "square" else None


def loader(ds: QueryDataset, batch_size: int, shuffle: bool, num_workers: int = 0,
           generator: torch.Generator | None = None, sampler=None,
           square: int | None = None) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle and sampler is None,
                                       num_workers=num_workers, collate_fn=make_collate(ds.white(), square),
                                       persistent_workers=num_workers > 0, generator=generator, sampler=sampler)


def check_disjoint(cfg: dict, units=tuple(UNITS)) -> None:
    """No diagram id appears in two dataset units (plan §5.7)."""
    seen: dict[str, str] = {}
    for u in units:
        with open(index_path(cfg, u)) as f:
            for line in f:
                did = json.loads(line)["diagram_id"]
                if seen.setdefault(did, u) != u:
                    raise AssertionError(f"{did} in both {seen[did]} and {u}")


def examples_figure(cfg: dict, out: Path, n: int = 4) -> None:
    """Model inputs: n queries per evaluation unit (no augmentation) and 2n augmented training ones."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vocab = vocabulary(cfg)
    rows = [("train", True), ("train", True), ("geometry3k_test", False), ("pgdp5k_heldout", False), ("synth_extrap", False)]
    fig, axes = plt.subplots(len(rows), n, figsize=(3.3 * n, 3.4 * len(rows)))
    for (unit, aug), ax_row, k in zip(rows, axes, range(len(rows))):
        ds = QueryDataset(cfg, unit)
        pos = [i for i, r in enumerate(ds.rows) if any(r["geo"])]
        rng = np.random.default_rng(k)
        for ax, i in zip(ax_row, rng.choice(pos, n, replace=False)):
            x, geo, _, meta = ds[int(i)]
            img = (x * ds.std + ds.mean).clamp(0, 1).permute(1, 2, 0).numpy()
            ax.imshow(img)
            lab = ", ".join(v for v, g in zip(vocab, geo.tolist()) if g) or "none"
            ax.set_title(f"{unit}{' (augmented)' if aug else ''}\n{meta['kind']}: {lab}", fontsize=8)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=90)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "smoke.yaml"))
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--examples", action="store_true", help="also write reports/figures/dataset_examples.png")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    bs = args.batch_size or cfg["finetune"]["batch_size"]
    R = len(vocabulary(cfg))
    for unit in UNITS:
        ds = QueryDataset(cfg, unit, build=True)
        x, geo, asr, meta = next(iter(loader(ds, bs, shuffle=unit == "train", num_workers=0)))
        B = min(bs, len(ds))
        assert x.shape[0] == B and x.shape[1] == 3, x.shape
        assert x.shape[2] % cfg["input"]["pad_to_multiple"] == 0 and x.shape[3] % cfg["input"]["pad_to_multiple"] == 0
        assert max(x.shape[2:]) <= cfg["input"]["long_side"] and geo.shape == asr.shape == meta["mask"].shape == (B, R)
        print(f"{unit:16s} {len(ds):5d} queries  batch x {tuple(x.shape)}  geo {tuple(geo.shape)}  asr {tuple(asr.shape)}  "
              f"mask {tuple(meta['mask'].shape)}  x range [{x.min():.2f}, {x.max():.2f}]  positives in batch {int(geo.sum())}")
    check_disjoint(cfg)
    print("GATE 5.8: dataset built on the smoke config and one batch per unit loaded with correct shapes; units disjoint -> PASS")
    if args.examples:
        out = REPO_ROOT / cfg["paths"]["figures"] / "dataset_examples.png"
        examples_figure(cfg, out)
        print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
