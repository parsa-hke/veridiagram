"""torch Dataset over (image, query) -> (image, label_geometric, label_asserted, meta) (plan §5.6).

Per item: the query's diagram image is flattened on white, resized so its long side is
`input.long_side` (448), and the queried primitives are drawn on it (data.overlay); this
overlaid image is cached as PNG at <cache>/overlays/<settings hash>/<set>/<diagram>/<query id>.png
(the hash covers the overlay settings and the long side, so changing them never reuses stale images). Training
items are then augmented (data.augment) and re-fitted to the long side. Finally the image is padded
on the right/bottom with white to a multiple of `input.pad_to_multiple` (the patch size) and
normalised with `input.mean` / `input.std`. `collate` pads a batch to its largest height and width
with normalised white, so batches are (B, 3, H, W) with H, W <= 448 and multiples of 16.

meta per item: qid, diagram_id, set, split, kind, relation mask (applicability), refs.

    python -m data.dataset --config configs/smoke.yaml [--examples]   # plan §5.8 gate: build + load one batch
"""

from __future__ import annotations

import argparse
import json
import zlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from data.augment import build_augment
from data.overlay import draw_overlay, flatten
from data.queries import UNITS, build_index, index_path, vocabulary
from data.schema import read_jsonl
from experiments.run import REPO_ROOT, load_config


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
            img, k = fit_long_side(flatten(im), self.L)
        img, _ = draw_overlay(img, d, r["refs"], self.cfg["overlay"], scale=k)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            img.save(path)
        return img

    def to_tensor(self, im: Image.Image) -> torch.Tensor:
        a = torch.from_numpy(np.asarray(pad_to_multiple(im, self.patch), dtype=np.float32) / 255.0).permute(2, 0, 1)
        return (a - self.mean) / self.std

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
        return (self.to_tensor(im), torch.tensor(r["geo"], dtype=torch.float32),
                torch.tensor(r["asr"], dtype=torch.float32), meta)

    def white(self) -> torch.Tensor:
        return ((torch.ones(3, 1, 1) - self.mean) / self.std).view(3)


def make_collate(white: torch.Tensor):
    def collate(batch):
        imgs, geo, asr, metas = zip(*batch)
        H = max(x.shape[1] for x in imgs)
        W = max(x.shape[2] for x in imgs)
        out = white.view(1, 3, 1, 1).repeat(len(imgs), 1, H, W)
        for k, x in enumerate(imgs):
            out[k, :, : x.shape[1], : x.shape[2]] = x
        return out, torch.stack(geo), torch.stack(asr), {
            "mask": torch.stack([m["mask"] for m in metas]),
            **{key: [m[key] for m in metas] for key in ("qid", "diagram_id", "set", "split", "kind", "refs")}}
    return collate


def loader(ds: QueryDataset, batch_size: int, shuffle: bool, num_workers: int = 0) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                                       collate_fn=make_collate(ds.white()), persistent_workers=num_workers > 0)


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
