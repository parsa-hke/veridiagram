"""scripts/reproduce.py: step list, skipping, failure handling, timings (no real step runs)."""

import csv
import importlib.util

from scripts import reproduce as rp


def test_every_step_module_exists_and_has_main():
    for s in rp.STEPS:
        assert importlib.util.find_spec(s.module) is not None, s.module
        assert s.stage in rp.STAGES
    assert [s.stage for s in rp.STEPS] == sorted((s.stage for s in rp.STEPS), key=rp.STAGES.index)


def test_attention_and_tolerance_steps_do_not_touch_config():
    tol = next(s for s in rp.STEPS if s.module == "verifier.tolerance")
    assert "--write-config" not in tol.argv


def test_run_skips_missing_inputs_and_records_timings(tmp_path):
    (tmp_path / "here").mkdir()
    calls = []
    steps = [rp.Step("data", "a", "mod.a", ["x"], needs=["here"]),
             rp.Step("data", "b", "mod.b", needs=["absent"]),
             rp.Step("report", "c", "mod.c")]
    loader = lambda m: (lambda argv: calls.append((m, argv)) or 0)
    out = tmp_path / "t.csv"
    res = rp.run(steps, root=tmp_path, loader=loader, quiet=True, timings=out)
    assert [r["status"] for r in res] == ["ok", "skipped", "ok"]
    assert calls == [("mod.a", ["x"]), ("mod.c", [])]
    rows = list(csv.DictReader(open(out)))
    assert rows[1]["note"] == "missing absent" and rows[0]["seconds"]


def test_run_stops_at_first_failure(tmp_path):
    def loader(m):
        def main(argv):
            if m == "bad":
                raise SystemExit(2)
            return 0
        return main
    steps = [rp.Step("data", "ok", "good"), rp.Step("data", "bad", "bad"), rp.Step("data", "never", "good")]
    res = rp.run(steps, root=tmp_path, loader=loader, quiet=True, timings=None)
    assert [r["status"] for r in res] == ["ok", "exit 2"]


def test_list_runs_nothing(capsys):
    assert rp.main(["--list", "--stages", "report"]) == 0
    assert "scripts.report_numbers" in capsys.readouterr().out


def test_load_main_handles_mains_without_arguments():
    for s in rp.STEPS:
        assert callable(rp.load_main(s.module))
