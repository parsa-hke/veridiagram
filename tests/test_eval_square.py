"""Square-padded re-evaluation (plan §0.2 #10 note): data.dataset padding, experiments/evaluate.py's
source-run config, probe feature paths and analysis.phase7's padding table. Tiny random ViT, no data."""

import csv
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from analysis import phase7
from data.dataset import Collate, eval_square, pad_batch
from experiments import evaluate
from experiments.run import REPO_ROOT, resolve_config
from models.backbone import tiny_backbone
from models.finetune import Classifier
from models.probe import feature_path

EVAL_SQUARE = REPO_ROOT / "configs" / "eval_square.yaml"
WHITE = torch.full((3,), 2.0)


def _cfg(source="finetune_synth10k", seed=2, smoke=False, extra=()):
    return resolve_config(EVAL_SQUARE, smoke=smoke, overrides=[
        f"seed={seed}", f"name=sq_{source}", f"eval.source_config=configs/{source}.yaml", *extra])


def test_pad_batch_square():
    a, b = torch.zeros(3, 32, 48), torch.ones(3, 48, 16)
    assert pad_batch([a, b], WHITE).shape == (2, 3, 48, 48)
    out = pad_batch([a, b], WHITE, square=64)
    assert out.shape == (2, 3, 64, 64)
    assert (out[0, :, :32, :48] == 0).all() and (out[0, :, 32:] == 2).all() and (out[0, :, :, 48:] == 2).all()
    assert (out[1, :, :48, :16] == 1).all() and (out[1, :, 48:] == 2).all()
    with pytest.raises(ValueError):
        pad_batch([torch.zeros(3, 80, 16)], WHITE, square=64)
    assert eval_square({"eval": {"pad": "square"}, "input": {"long_side": 448}}) == 448
    assert eval_square({"eval": {}, "input": {"long_side": 448}}) is None
    with pytest.raises(ValueError):
        eval_square({"eval": {"pad": "round"}, "input": {"long_side": 448}})


def test_square_padding_removes_the_batch_neighbour_effect():
    """The pooled feature averages padding tokens: batch padding makes an answer depend on its batch
    neighbours, a fixed square does not."""
    torch.manual_seed(0)
    model = Classifier(tiny_backbone("dinov3"), 6).eval()
    small, big = torch.randn(3, 32, 32), torch.randn(3, 64, 48)
    meta = {"qid": "q", "diagram_id": "d", "set": "s", "split": "x", "kind": "seg_pair", "refs": [],
            "mask": torch.ones(6)}
    item = lambda x: (x, torch.zeros(6), torch.zeros(6), meta)  # noqa: E731

    def answer(batch, square):
        x, *_ = Collate(WHITE, square)([item(t) for t in batch])
        with torch.no_grad():
            return model(x)[0]
    assert not torch.allclose(answer([small], None), answer([small, big], None), atol=1e-4)
    assert torch.allclose(answer([small], 64), answer([small, big], 64), atol=1e-5)


def test_source_config_and_checkpoint():
    eff = evaluate.source_config(_cfg())
    assert eff["name"] == "sq_finetune_synth10k" and eff["mode"] == "evaluate" and eff["seed"] == 2
    assert eff["eval"]["source_run"] == "finetune_synth10k_s2" and eff["source_mode"] == "finetune"
    assert eff["eval"]["pad"] == "square" and eff["data"]["synthetic_train_size"] == 10000
    assert evaluate.checkpoint_of(eff).parts[-3:] == ("finetune_synth10k_s2", "checkpoints", "best.pt")
    probe = evaluate.source_config(_cfg("probe", 1))
    assert evaluate.checkpoint_of(probe).parts[-2:] == ("probe_s1", "probe_head.pt")
    dino = evaluate.source_config(_cfg("ablation_dinov2", 0))
    assert dino["model"]["backbone"] == "facebook/dinov2-large"
    nooverlay = evaluate.source_config(_cfg("ablation_nooverlay", 0))
    assert nooverlay["overlay"]["enabled"] is False and nooverlay["model"]["query_input"] == "tokens"
    smoke = evaluate.source_config(_cfg("finetune", 0, smoke=True))
    assert smoke["eval"]["source_run"] == "smoke_finetune_s0" and smoke["data"]["max_diagrams"] == 50


def test_source_mismatches():
    eff = evaluate.source_config(_cfg())
    saved = yaml.safe_load(yaml.safe_dump(eff))
    assert evaluate.source_mismatches(eff, saved) == []
    del saved["model"]["query_input"]                      # a key added after the run, at its default
    assert evaluate.source_mismatches(eff, saved) == []
    saved["model"]["backbone"] = "facebook/dinov2-large"
    saved["data"]["synthetic_train_size"] = 0
    bad = evaluate.source_mismatches(eff, saved)
    assert len(bad) == 2 and bad[0].startswith("model.backbone")


def test_evaluate_refuses_a_mismatched_or_missing_source(tmp_path):
    cfg = _cfg(extra=[f"paths.runs={tmp_path}"])
    with pytest.raises(FileNotFoundError):
        evaluate.run(cfg, tmp_path / "sq_finetune_synth10k_s2")
    src = tmp_path / "finetune_synth10k_s2"
    (src / "checkpoints").mkdir(parents=True)
    (src / "checkpoints" / "best.pt").write_bytes(b"x")
    saved = evaluate.source_config(cfg)
    saved["model"]["backbone"] = "facebook/dinov2-large"
    (src / "config.yaml").write_text(yaml.safe_dump(saved))
    with pytest.raises(ValueError, match="model.backbone"):
        evaluate.run(cfg, tmp_path / "sq_finetune_synth10k_s2")


def test_probe_features_cached_per_padding():
    cfg = resolve_config(REPO_ROOT / "configs" / "probe.yaml")
    sq = resolve_config(REPO_ROOT / "configs" / "probe.yaml", overrides=["eval.pad=square"])
    val, train = SimpleNamespace(unit="val", cache_tag="abcd1234"), SimpleNamespace(unit="train", cache_tag="abcd1234")
    assert feature_path(cfg, "b", val) != feature_path(sq, "b", val)
    assert feature_path(sq, "b", val).parent.name == "abcd1234_square"
    assert feature_path(cfg, "b", train) == feature_path(sq, "b", train)        # training features keep batch padding


def test_padding_effect_table(tmp_path):
    vocab = ["parallel", "perpendicular"]
    runs = tmp_path / "runs"
    log = []
    for run, probs, score in (("finetune_s0", [[0.9, 0.2], [0.4, 0.6]], 0.8), ("sq_finetune_s0", [[0.8, 0.6], [0.4, 0.7]], 0.85)):
        d = runs / run / "predictions"
        d.mkdir(parents=True)
        res = {ls: {"macro_f1": score, "f1": {r: score for r in vocab}} for ls in ("geometric", "asserted")}
        (runs / run / "metrics.json").write_text(json.dumps({"status": "ok", "results": {"val": {**res, "n": 2}}}))
        np.savez(d / "val.npz", qid=np.array(["a", "b"]), probs=np.array(probs), geo=np.zeros((2, 2)),
                 asr=np.zeros((2, 2)), mask=np.array([[1, 1], [1, 0]]))
        log.append({"name": run, "status": "ok"})
    with open(runs / "log.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name", "status"])
        w.writeheader()
        w.writerows(log)
    rows, problems = phase7.padding_effect(runs, phase7.latest_runs(runs), vocab)
    assert not problems and len(rows) == 1
    r = rows[0]
    assert (r["experiment"], r["unit"], r["n_decisions"], r["decisions_changed"]) == ("finetune", "val", 3, 1)
    assert r["geometric_change"] == pytest.approx(0.05) and r["max_abs_dp"] == pytest.approx(0.4)
    assert "+0.050 (33.3%)" in phase7.padding_markdown(rows)


def test_final_queue():
    from experiments.queue import expand
    jobs = expand(REPO_ROOT / "configs" / "queue_phase7_final.yaml")
    assert len(jobs) == 29 and jobs[0]["run"] == "selftrain_control_s1"
    sq = [j for j in jobs if j["mode"] == "evaluate"]
    assert len(sq) == 24 and all(j["run"].startswith("sq_") for j in sq)
    for j in sq:
        cfg = resolve_config(j["config"], overrides=[f"seed={j['seed']}", *j["set"]])
        eff = evaluate.source_config(cfg)
        assert j["run"] == f"sq_{eff['eval']['source_run']}" and eff["eval"]["pad"] == "square"
    assert [j["run"] for j in jobs].index("finetune_colab_s2") < [j["run"] for j in jobs].index("sq_finetune_colab_s0")
