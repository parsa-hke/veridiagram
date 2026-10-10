"""experiments/gate.py: smoke gate checks on run records."""

import json

from experiments import gate
from experiments.run import REPO_ROOT, load_config

UNITS = ["val", "geometry3k_test", "pgdp5k_heldout", "synth_extrap"]


def _run(root, mode, backbone, dur, status="ok"):
    d = root / f"smoke_{mode}_s0"
    (d / "predictions").mkdir(parents=True)
    for u in UNITS:
        (d / "predictions" / f"{u}.npz").write_bytes(b"x")
    if mode == "finetune":
        (d / "checkpoints").mkdir()
        (d / "checkpoints" / "best.pt").write_bytes(b"x")
    (d / "metrics.json").write_text(json.dumps({"status": status, "backbone_loaded": backbone, "duration_s": dur}))


def test_smoke_gate(tmp_path):
    cfg = load_config(REPO_ROOT / "configs/base.yaml")
    cfg["paths"]["runs"] = str(tmp_path)
    primary = cfg["model"]["backbone"]
    _run(tmp_path, "probe", primary, 106)
    _run(tmp_path, "finetune", primary, 300)
    assert all(ok for _, ok, _ in gate.check_smoke(cfg))
    (tmp_path / "smoke_finetune_s0/metrics.json").write_text(json.dumps(
        {"status": "ok", "backbone_loaded": cfg["model"]["backbone_fallback"], "duration_s": 300}))
    failed = [n for n, ok, _ in gate.check_smoke(cfg) if not ok]
    assert failed == ["finetune: primary backbone"]
