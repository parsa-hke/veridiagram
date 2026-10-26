"""scripts/archive_runs.py on a fake runs/ directory."""

import csv

from scripts import archive_runs as A


def _fake(tmp):
    runs, gold = tmp / "runs", tmp / "gold"
    (runs / "a_s0" / "checkpoints").mkdir(parents=True)
    (runs / "a_s0" / "metrics.json").write_text('{"x": 1}')
    (runs / "a_s0" / "config.yaml").write_text("name: a\n")
    (runs / "a_s0" / "predictions").mkdir()
    (runs / "a_s0" / "predictions" / "val.npz").write_bytes(b"big")
    (runs / "a_s0" / "checkpoints" / "best.pt").write_bytes(b"weights")
    (runs / "smoke_s0" / "checkpoints").mkdir(parents=True)
    (runs / "smoke_s0" / "checkpoints" / "best.pt").write_bytes(b"w")
    (runs / "benchmark_lab").mkdir()
    (runs / "benchmark_lab" / "benchmark.json").write_text("{}")
    (runs / "log.csv").write_text("name\na_s0\n")
    gold.mkdir()
    (gold / "gold.csv").write_text("id\n1\n")
    return runs, gold


def test_build_copies_records_not_predictions_and_digests_checkpoints(tmp_path):
    runs, gold = _fake(tmp_path)
    out = tmp_path / "archive"
    rows = A.build(runs, out, gold, tmp_path)
    assert (out / "runs" / "log.csv").exists() and (out / "runs" / "a_s0" / "metrics.json").exists()
    assert (out / "runs" / "benchmark_lab" / "benchmark.json").exists()
    assert not (out / "runs" / "a_s0" / "predictions").exists()
    kinds = {r["path"]: r["kind"] for r in rows}
    assert kinds["runs/a_s0/checkpoints/best.pt"] == "checkpoint (not copied)"
    assert "runs/smoke_s0/checkpoints/best.pt" not in kinds
    assert kinds["gold/gold.csv"] == "gold set (committed)"
    man = list(csv.DictReader(open(out / "MANIFEST.csv")))
    assert len(man) == len(rows) and all(len(r["sha256"]) == 64 for r in man)
    assert A.check(runs, out) == []
    (runs / "a_s0" / "metrics.json").write_text('{"x": 2}')
    assert A.check(runs, out) == ["runs/a_s0/metrics.json"]
