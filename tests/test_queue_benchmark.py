"""experiments/queue.py and experiments/benchmark.py."""

import copy

import pytest

from experiments import benchmark, queue
from experiments.run import REPO_ROOT, load_config  # noqa: F401


def test_expand_status_and_serial(tmp_path):
    q = tmp_path / "q.yaml"
    q.write_text("jobs:\n"
                 "  - {config: configs/probe.yaml, seeds: [0, 1], serial: true}\n"
                 "  - {config: configs/finetune.yaml, seeds: [0], set: [finetune.epochs=2]}\n")
    jobs = queue.expand(q)
    assert [j["run"] for j in jobs] == ["probe_s0", "probe_s1", "finetune_s0"]
    assert jobs[2]["set"] == ["finetune.epochs=2"] and "--set" in queue.command(jobs[2])
    assert queue.next_job(jobs, [jobs[0]]) is jobs[2]                    # probe_s1 waits for probe_s0
    assert queue.next_job(jobs[1:], []) is jobs[1]
    (tmp_path / "log.csv").write_text("timestamp,name,mode,seed,config_hash,status,best_epoch,duration_s,metrics\n"
                                      "t,probe_s0,probe,0,h,ok,1,1,{}\nt,finetune_s0,finetune,0,h,failed:X,,1,{}\n")
    jobs = [dict(j, log_csv=str(tmp_path / "log.csv")) for j in jobs]
    assert queue.job_status(jobs) == {"probe_s0": "ok", "probe_s1": "not started", "finetune_s0": "failed:X"}
    with pytest.raises(ValueError):
        dup = tmp_path / "dup.yaml"
        dup.write_text("jobs:\n  - {config: configs/probe.yaml, seeds: [0, 0]}\n")
        queue.expand(dup)


def test_queue_file_expands():
    jobs = queue.expand(REPO_ROOT / "configs/queue_phase7.yaml")
    names = [j["run"] for j in jobs]
    assert names[0] == "probe_s0" and "finetune_synth10k_s2" in names and len(names) == len(set(names))


def test_estimates_and_makespan():
    rates = {448: {"train_img_per_s": 50.0, "eval_img_per_s": 150.0}, 224: {"train_img_per_s": 200.0, "eval_img_per_s": 600.0}}
    sizes = {"train": 37846, "val": 6227, "geometry3k_test": 11319, "pgdp5k_heldout": 25272, "synth_extrap": 60000}
    job = {"run": "finetune_s0", "config": str(REPO_ROOT / "configs/finetune.yaml"), "seed": 0, "set": [], "mode": "finetune"}
    e = benchmark.estimate_job(job, rates, 37846, sizes)
    expected = (30 * (37846 / 50 + 6227 / 150) + (6227 + 11319 + 25272 + 60000) / 150 + 120) / 3600
    assert e["hours"] == pytest.approx(expected, abs=0.01)
    j224 = dict(job, config=str(REPO_ROOT / "configs/ablation_res224.yaml"), run="ablation_res224_s0")
    assert benchmark.estimate_job(j224, rates, 37846, sizes)["hours"] < e["hours"] / 3
    assert benchmark.makespan([4, 3, 3, 2], 2) == 6 and benchmark.makespan([5, 1], 4) == 5


@pytest.mark.skipif(not (REPO_ROOT / "data/processed/geometry3k_figures.jsonl").exists(), reason="no data")
def test_measure_tiny(tmp_path):
    from models.backbone import tiny_backbone
    cfg = copy.deepcopy(load_config(REPO_ROOT / "configs/smoke.yaml"))
    cfg["device"] = "cpu"
    cfg["data"].update(max_diagrams=2, synthetic_train_size=0, max_queries_per_unit=24)
    cfg["paths"]["cache"] = str(tmp_path / "cache")
    cfg["finetune"].update(batch_size=4, amp=False)
    cfg["eval"]["batch_size"] = 4
    r = benchmark.measure(cfg, tiny_backbone("dinov3"), steps=2)
    assert r["train_img_per_s"] > 0 and r["eval_img_per_s"] > 0
