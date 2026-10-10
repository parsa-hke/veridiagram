"""Download and extract the three datasets into <raw> (plan §2.1; one-command setup for §10.1).

    python -m data.download [--config X]

Archives are checked against the sha256 sums recorded on 2026-10-10 (CHANGELOG, data/README.md);
a mismatch stops with an error. Extraction refuses absolute paths and '..' entries. Existing,
verified archives are not downloaded again.
"""

from __future__ import annotations

import argparse
import hashlib
import urllib.request
import zipfile
from pathlib import Path

from experiments.run import REPO_ROOT, load_config

G3K = "https://raw.githubusercontent.com/lupantech/InterGPS/main/data/geometry3k/{}.zip"
ARCHIVES = [
    ("geometry3k", "train.zip", G3K.format("train"), "e12ee4d2ad85117a4d4279ce46917ac5f2a3aaa2266d4413118b6087a151a95b"),
    ("geometry3k", "val.zip", G3K.format("val"), "67b1b0a9b424cb5e6634d20a6494a554c529156a6cfd0492c636e047e13f314a"),
    ("geometry3k", "test.zip", G3K.format("test"), "a0d5dece634686303eb6cabca0a7306136ca8ade5f562a4af4c6561ecaf6c9b3"),
    ("geometry3k", "logic_forms.zip", G3K.format("logic_forms"), "264cd7316056d7ee57171b57397e3555c97e40d648a4e901db7baec59423665d"),
    ("geometry3k", "symbols.zip", G3K.format("symbols"), "90966955e3d8a81b6d171ad820ffbb8223e3a6a3aa8f54307c94ef4ea6ef65bb"),
    ("pgdp5k", "PGDP5K.zip", "https://huggingface.co/datasets/PeijieWang/PGDP5K/resolve/main/PGDP5K.zip",
     "2fe9c0c361f231591e3727360a69fa7a639a67ee8b1c13da549c12af0b80f60f"),
    ("pgps9k", "PGPS9K.zip", "https://huggingface.co/datasets/PeijieWang/PGPS9K/resolve/main/PGPS9K.zip",
     "a74b2b819ed4dff91e870b17a33b7860de5f118b254c1f48e8be0fe10ab9bc21"),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_extract(zpath: Path, dest: Path) -> int:
    with zipfile.ZipFile(zpath) as z:
        names = [n for n in z.namelist() if not n.startswith("__MACOSX/")]
        bad = [n for n in names if n.startswith("/") or ".." in Path(n).parts]
        if bad:
            raise RuntimeError(f"{zpath.name}: unsafe paths {bad[:3]}")
        for n in names:
            z.extract(n, dest)
    return len(names)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "base.yaml"))
    args = ap.parse_args(argv)
    raw = REPO_ROOT / load_config(args.config)["paths"]["raw"]
    for src, name, url, digest in ARCHIVES:
        zdir = raw / src / "_zips"
        zdir.mkdir(parents=True, exist_ok=True)
        z = zdir / name
        if not (z.exists() and sha256(z) == digest):
            print(f"downloading {src}/{name} ...", flush=True)
            urllib.request.urlretrieve(url, z)
        got = sha256(z)
        if got != digest:
            raise SystemExit(f"{src}/{name}: sha256 {got} != expected {digest}")
        n = safe_extract(z, raw / src)
        print(f"{src}/{name}: verified, {n} entries extracted", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
