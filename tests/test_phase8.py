"""Phase 8.1 / 8.2 on synthetic predictions (no data, no GPU): ECE, thresholds, bootstrap CIs, the
phase8 tables and the headline table. Runs in a few seconds."""

import csv
import json

import numpy as np
import pytest

from analysis import headline_table, metrics as M, phase8

VOCAB = ["parallel", "perpendicular", "equal_length", "on_circle", "tangent", "collinear"]
R = len(VOCAB)


def test_ece_calibrated_vs_inverted():
    rng = np.random.default_rng(0)
    p = rng.random((4000, R))
    y = (rng.random((4000, R)) < p).astype(float)
    m = np.ones_like(p)
    assert M.ece(p, y, m) < 0.03                           # sampled from its own probabilities: calibrated
    assert M.ece(1 - p, y, m) > 0.4                        # inverted: badly calibrated
    assert np.isnan(M.ece(p, y, np.zeros_like(m)))         # nothing applicable
    per = M.ece_per_relation(p, y, m, VOCAB)
    assert set(per) == set(VOCAB) and all(v < 0.05 for v in per.values())


def test_ece_known_value():
    # two entries in one bin: probs 0.9, 0.9 with labels 1, 0 -> |0.5 - 0.9| = 0.4
    p = np.array([[0.9], [0.9]])
    assert M.ece(p, np.array([[1.0], [0.0]]), np.ones((2, 1))) == pytest.approx(0.4)


def test_tune_thresholds_recovers_cut_and_ignores_mask():
    rng = np.random.default_rng(1)
    p = rng.random((3000, R))
    y = (p >= 0.3).astype(float)                           # the truth is "p >= 0.3"
    m = np.ones_like(p)
    m[:, 4] = 0                                            # tangent never applicable -> stays at 0.5
    th = M.tune_thresholds(p, y, m, VOCAB)
    assert list(th[:4]) == [0.3] * 4 and th[4] == 0.5 and th[5] == 0.3
    y[:, 0] = 0                                            # no positives -> stays at 0.5
    assert M.tune_thresholds(p, y, m, VOCAB)[0] == 0.5


def test_bootstrap_ci_brackets_point_and_shrinks():
    rng = np.random.default_rng(2)

    def data(n_diag, per=10):
        p = rng.random((n_diag * per, R))
        y = (rng.random((n_diag * per, R)) < 0.7).astype(float)
        pred = (rng.random(p.shape) < 0.85) == (y > 0)     # 85% correct decisions
        return pred, y, np.ones_like(p), np.repeat(np.arange(n_diag), per)

    pred, y, m, d = data(100)
    b = M.bootstrap_macro_f1([(pred, y, m)], d, VOCAB, n_resamples=500)
    point = M.evaluate(pred, y, m, VOCAB)["macro_f1"]
    assert b["macro_f1"] == pytest.approx(point) and b["lo"] < point < b["hi"] and b["n_diagrams"] == 100
    assert all(v["lo"] <= v["f1"] <= v["hi"] for v in b["per_relation"].values())
    pred2, y2, m2, d2 = data(1000)
    b2 = M.bootstrap_macro_f1([(pred2, y2, m2)], d2, VOCAB, n_resamples=500)
    assert (b2["hi"] - b2["lo"]) < (b["hi"] - b["lo"]) / 2          # ten times the diagrams: narrower CI
    # two identical seeds: the seed-mean equals the single run
    b3 = M.bootstrap_macro_f1([(pred, y, m), (pred, y, m)], d, VOCAB, n_resamples=500)
    assert b3["macro_f1"] == pytest.approx(point) and b3["points"] == [pytest.approx(point)] * 2
    assert b3["lo"] == pytest.approx(b["lo"]) and b3["hi"] == pytest.approx(b["hi"])
    # reproducible
    assert M.bootstrap_macro_f1([(pred, y, m)], d, VOCAB, n_resamples=500)["lo"] == b["lo"]


def test_bootstrap_relation_without_positives_is_excluded():
    pred = np.ones((20, R), bool)
    y = np.ones((20, R))
    y[:, 2] = 0                                            # equal_length has no positives: F1 undefined
    b = M.bootstrap_macro_f1([(pred, y, np.ones((20, R)))], np.arange(20), VOCAB, n_resamples=50)
    assert b["macro_f1"] == pytest.approx(1.0) and b["per_relation"]["equal_length"]["f1"] is None


# ------------------------------------------------------------------------------- fake runs on disk

def _fake_runs(tmp_path, quality: dict[str, float], n_diag=40, per=8, seed=0):
    """runs/ with val + the three test units for each run; a query's probability is right with
    probability `quality[experiment]`. Returns (runs dir, rows_of)."""
    rng = np.random.default_rng(seed)
    runs = tmp_path / "runs"
    runs.mkdir()
    rows_of = {}
    for u in phase8.UNITS:
        n = n_diag * per
        rows_of[u] = {"qid": np.array([f"{u}_{i}" for i in range(n)]),
                      "diagram": np.array([f"{u}:d{i // per}" for i in range(n)])}
    truth = {u: (rng.random((n_diag * per, R)) < 0.4).astype(np.float32) for u in phase8.UNITS}
    mask = {u: (rng.random((n_diag * per, R)) < 0.8).astype(np.float32) for u in phase8.UNITS}
    log = []
    for exp, q in quality.items():
        for s in (0, 1):
            run = f"{exp}_s{s}"
            d = runs / run / "predictions"
            d.mkdir(parents=True)
            for u in phase8.UNITS:
                right = rng.random(truth[u].shape) < q
                probs = np.where(right == (truth[u] > 0), rng.uniform(0.6, 1.0, truth[u].shape),
                                 rng.uniform(0.0, 0.4, truth[u].shape)).astype(np.float32)
                np.savez(d / f"{u}.npz", probs=probs, geo=truth[u], asr=truth[u], mask=mask[u], qid=rows_of[u]["qid"])
            (runs / run / "metrics.json").write_text(json.dumps({"status": "ok", "results": {}}))
            log.append({"timestamp": "t", "name": run, "mode": "x", "seed": s, "config_hash": "h", "status": "ok",
                        "best_epoch": 1, "duration_s": 1, "metrics": "{}"})
    with open(runs / "log.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=log[0].keys())
        w.writeheader()
        w.writerows(log)
    return runs, rows_of


def test_phase8_tables(tmp_path):
    runs, rows_of = _fake_runs(tmp_path, {"sq_probe": 0.7, "sq_finetune_colab": 0.95})
    latest = phase8.latest_runs(runs)
    per_run, problems = phase8.per_run_metrics(runs, latest, rows_of, VOCAB, n_resamples=100)
    assert not problems and len(per_run) == 4 * 3 * 2 * 2          # runs x units x label sets x operating points
    ft = [r for r in per_run if r["run"] == "sq_finetune_colab_s0" and r["unit"] == "geometry3k_test"
          and r["label_set"] == "geometric" and r["operating"] == "tuned"][0]
    assert ft["ci_lo"] <= ft["macro_f1"] <= ft["ci_hi"] and 0 <= ft["ece"] <= 1 and ft["n_diagrams"] == 40
    assert all(0.05 <= ft[f"threshold_{v}"] <= 0.95 for v in VOCAB)
    summary = phase8.per_experiment_metrics(runs, latest, rows_of, VOCAB, n_resamples=100)
    assert len(summary) == 2 * 3 * 2 * 2
    s = [r for r in summary if r["experiment"] == "sq_finetune_colab" and r["unit"] == "pgdp5k_heldout"
         and r["label_set"] == "geometric" and r["operating"] == "tuned"][0]
    p = [r for r in summary if r["experiment"] == "sq_probe" and r["unit"] == "pgdp5k_heldout"
         and r["label_set"] == "geometric" and r["operating"] == "tuned"][0]
    assert s["n_seeds"] == 2 and s["sd"] is not None and s["ci_lo"] <= s["mean"] <= s["ci_hi"]
    assert s["mean"] > p["ci_hi"]                                   # the better model is separated by the CI
    md = phase8.markdown(summary, VOCAB)
    assert "| sq_finetune_colab |" in md and "| sq_probe |" in md and "Per-relation F1 on synth_extrap" in md
    # a stale prediction file is a reported problem, not a silent mismatch
    z = dict(np.load(runs / "sq_probe_s0/predictions/val.npz"))
    z["qid"] = z["qid"][::-1]
    np.savez(runs / "sq_probe_s0/predictions/val.npz", **z)
    _, problems = phase8.per_run_metrics(runs, latest, rows_of, VOCAB, n_resamples=10)
    assert problems and "sq_probe_s0/val" in problems[0]


def test_headline_table(tmp_path):
    runs, rows_of = _fake_runs(tmp_path, {"sq_probe": 0.7, "sq_finetune_colab": 0.95})
    latest = phase8.latest_runs(runs)
    rows = headline_table.headline(runs, latest, "sq_probe", "sq_finetune_colab", rows_of, VOCAB)
    q = [r for r in rows if r["seed"] == "mean" and r["level"] == "query"]
    assert len(q) == 3 and all(r["n"] == 320 for r in q)
    for r in q:
        assert sum(r[c] for c in headline_table.CATEGORIES) == pytest.approx(r["n"])
        assert r["finetune_only"] > r["probe_only"]
        assert sum(r[f"share_{c}"] for c in headline_table.CATEGORIES) == pytest.approx(1.0, abs=1e-3)
    dec = [r for r in rows if r["seed"] == 0 and r["level"] == "decision" and r["unit"] == "geometry3k_test"]
    assert {r["relation"] for r in dec} == {"all", *VOCAB}
    assert sum(r["n"] for r in dec if r["relation"] != "all") == next(r["n"] for r in dec if r["relation"] == "all")
    md = headline_table.markdown(rows, VOCAB)
    assert "| geometry3k_test | 320 |" in md
    with pytest.raises(SystemExit):
        headline_table.headline(runs, latest, "sq_probe", "missing", rows_of, VOCAB)


def test_four_way_and_decisions():
    fw = headline_table.four_way([1, 1, 0, 0], [1, 0, 1, 0])
    assert fw == {"both": 1, "probe_only": 1, "finetune_only": 1, "neither": 1}
    preds = {"probs": np.array([[0.9, 0.2]]), "geo": np.array([[1.0, 1.0]]), "mask": np.array([[1.0, 0.0]])}
    assert headline_table.decisions(preds, np.array([0.5, 0.5])).tolist() == [[True, True]]   # masked entry counts as right
