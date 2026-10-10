"""experiments/run.py: config resolution, overrides, seeding, run-dir + log writing."""

import csv
import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments import run as R

OVERLAYS = {"colab.yaml"}  # partial configs, only valid on top of another via --overlay
CONFIGS = sorted(p for p in (R.REPO_ROOT / "configs").glob("*.yaml") if p.name not in OVERLAYS)


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_every_config_resolves(path):
    cfg = R.resolve_config(path)
    assert cfg["mode"] in R.DISPATCH
    for section in ("paths", "data", "relations", "tolerances", "model", "input",
                    "augment", "probe", "finetune", "eval"):
        assert section in cfg, f"{path.name} missing {section}"


def test_base_has_plan_values():
    cfg = R.load_config(R.REPO_ROOT / "configs/base.yaml")
    assert cfg["input"]["long_side"] == 448
    assert cfg["finetune"]["llrd"] == 0.8
    assert cfg["relations"]["tier1"][-1] == "none"
    tol = cfg["tolerances"]  # calibrated in Phase 3.4
    assert 0.5 < tol["parallel_deg"] < 5 and 0.5 < tol["perpendicular_deg"] < 5
    assert 0 < tol["equal_length_rel"] <= 0.1 and tol["tangent_rel"] == tol["on_circle_rel"]
    assert cfg["not_to_scale"] == {"angle_deg": 10.0, "rel_length": 0.15}


def test_inherit_and_smoke_overlay():
    cfg = R.resolve_config(R.REPO_ROOT / "configs/probe.yaml", smoke=True)
    assert cfg["mode"] == "probe"
    assert cfg["name"] == "smoke_probe"
    assert cfg["data"]["max_diagrams"] == 50
    assert cfg["probe"]["epochs"] == 1
    assert cfg["finetune"]["batch_size"] == 16  # untouched base value survives the merge


def test_overrides():
    cfg = R.resolve_config(R.REPO_ROOT / "configs/finetune.yaml",
                           overrides=["seed=2", "finetune.backbone_lr=1e-5", "data.test_sets=[a, b]"])
    assert cfg["seed"] == 2
    assert cfg["finetune"]["backbone_lr"] == pytest.approx(1e-5)
    assert cfg["data"]["test_sets"] == ["a", "b"]
    with pytest.raises(KeyError):
        R.resolve_config(R.REPO_ROOT / "configs/finetune.yaml", overrides=["finetune.nope=1"])


def test_seed_is_reproducible():
    def draw():
        return random.random(), np.random.rand(), torch.rand(1).item()
    R.set_seed(123)
    a = draw()
    R.set_seed(123)
    assert draw() == a


def test_config_hash_ignores_seed():
    cfg = R.resolve_config(R.REPO_ROOT / "configs/probe.yaml")
    assert R.config_hash(cfg) == R.config_hash({**cfg, "seed": 7})
    assert R.config_hash(cfg) != R.config_hash({**cfg, "name": "other"})


def test_dry_run_writes_metrics_and_log(tmp_path):
    cfg = R.resolve_config(R.REPO_ROOT / "configs/probe.yaml", smoke=True,
                           overrides=[f"paths.runs={tmp_path}", f"paths.log_csv={tmp_path / 'log.csv'}"])
    R.run(cfg, dry_run=True)
    metrics = json.loads((tmp_path / "smoke_probe_s0" / "metrics.json").read_text())
    assert metrics["status"] == "dry_run" and metrics["seed"] == 0
    rows = list(csv.DictReader(open(tmp_path / "log.csv")))
    assert len(rows) == 1 and rows[0]["name"] == "smoke_probe_s0"


def test_failed_run_is_logged(tmp_path):
    cfg = R.resolve_config(R.REPO_ROOT / "configs/finetune.yaml", smoke=True,
                           overrides=[f"paths.runs={tmp_path}", f"paths.log_csv={tmp_path / 'log.csv'}"])
    with pytest.raises(NotImplementedError):
        R.run(cfg)  # finetune is a stub until Phase 6
    rows = list(csv.DictReader(open(tmp_path / "log.csv")))
    assert rows[0]["status"] == "failed:NotImplementedError"


def test_device_info_cli(capsys):
    assert R.main(["--device-info"]) == 0
    assert "device" in capsys.readouterr().out


def test_colab_overlay_keeps_mode_and_name():
    cfg = R.resolve_config(R.REPO_ROOT / "configs/finetune.yaml", smoke=True,
                           overlays=[R.REPO_ROOT / "configs/colab.yaml"], overrides=["seed=1"])
    assert cfg["mode"] == "finetune" and cfg["name"] == "smoke_finetune"
    assert cfg["paths"]["runs"].startswith("/content/drive/")
    assert cfg["paths"]["figures"] == "reports/figures"  # untouched keys keep base values
    assert cfg["data"]["max_diagrams"] == 50 and cfg["seed"] == 1
