"""Plan 10.5: archive the experiment record into the repository, so it can be cited without runs/.

runs/ is gitignored (3.5 GB: predictions and checkpoints). This copies the small, citable part of it
into archive/ and writes a manifest with SHA-256 digests:

  archive/runs/log.csv                      every run, one line each (config hash, seed, status, metrics)
  archive/runs/<run>/{config.yaml, metrics.json, history.jsonl, effective_config.yaml}
  archive/runs/benchmark_lab/benchmark.json the lab throughput benchmark
  archive/MANIFEST.csv                      path, bytes, sha256 of every archived file, plus the gold set
                                            (verifier/gold/, already committed) and the kept checkpoints
                                            (runs/*/checkpoints/best.pt, digest only; not copied)

    python -m scripts.archive_runs [--check]   # --check: exit 1 if the archive differs from runs/
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = ROOT / "archive"
FILES = ("config.yaml", "effective_config.yaml", "metrics.json", "history.jsonl")
GOLD = ROOT / "verifier" / "gold"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def plan(runs: Path = RUNS) -> list[tuple[Path, str]]:
    """(source, archive-relative destination) for every file to copy."""
    out = [(runs / "log.csv", "runs/log.csv")]
    for d in sorted(p for p in runs.iterdir() if p.is_dir()):
        for name in FILES + ("benchmark.json",):
            if (d / name).exists():
                out.append((d / name, f"runs/{d.name}/{name}"))
    return out


def build(runs: Path = RUNS, out: Path = OUT, gold: Path = GOLD, root: Path = ROOT) -> list[dict]:
    if out.exists():
        shutil.rmtree(out)
    rows = []
    for src, rel in plan(runs):
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        rows.append({"path": f"archive/{rel}", "kind": "copied", "bytes": dst.stat().st_size, "sha256": sha256(dst)})
    for p in sorted(gold.glob("*.csv")):
        rows.append({"path": str(p.relative_to(root)), "kind": "gold set (committed)", "bytes": p.stat().st_size, "sha256": sha256(p)})
    for p in sorted(runs.glob("*/checkpoints/best.pt")):
        if p.parent.parent.name.startswith("smoke"):
            continue
        rows.append({"path": str(p.relative_to(root)), "kind": "checkpoint (not copied)", "bytes": p.stat().st_size, "sha256": sha256(p)})
    with open(out / "MANIFEST.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "kind", "bytes", "sha256"])
        w.writeheader()
        w.writerows(rows)
    return rows


def check(runs: Path = RUNS, out: Path = OUT) -> list[str]:
    """Archived copies that are missing or differ from runs/."""
    bad = []
    for src, rel in plan(runs):
        dst = out / rel
        if not dst.exists() or dst.read_bytes() != src.read_bytes():
            bad.append(rel)
    return bad


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    if not (RUNS / "log.csv").exists():
        print("no runs/log.csv: nothing to archive (the committed archive/ is the record)")
        return 0
    if args.check:
        bad = check()
        print("archive is", "up to date" if not bad else f"STALE: {len(bad)} files, e.g. {bad[:3]}")
        return 1 if bad else 0
    rows = build()
    n_runs = len({r["path"].split("/")[2] for r in rows if r["kind"] == "copied" and r["path"].count("/") >= 3})
    print(f"archived {sum(r['kind'] == 'copied' for r in rows)} files from {n_runs} runs, "
          f"{sum(r['bytes'] for r in rows if r['kind'] == 'copied') / 2**20:.1f} MB; manifest archive/MANIFEST.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
