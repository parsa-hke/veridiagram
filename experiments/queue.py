"""Run a list of experiments across the GPUs of one machine (plan §0.6, LAB_GUIDE.md).

    python -m experiments.queue configs/queue_phase7.yaml            # run everything not yet done
    python -m experiments.queue configs/queue_phase7.yaml --status   # what is done / pending / failed
    python -m experiments.queue configs/queue_phase7.yaml --dry-run  # print the plan only
    options: --gpus 0,1,2   --only finetune ablation_res224   --set finetune.epochs=10 (applied to all)

Queue file: `jobs:` list of {config: <path>, seeds: [..], set: [k=v, ...] (optional), serial: true
(optional: the seeds of this entry run one after another, e.g. probes that share cached features)}. Each
(config, seed) becomes one `python -m experiments.run` process with CUDA_VISIBLE_DEVICES set to one
GPU; as soon as a GPU is free the next pending job starts there, in file order. A job whose run name
(<name>_s<seed>) already has status "ok" in runs/log.csv is skipped, so the same command can be
restarted any time; an interrupted fine-tune resumes from its last.pt. Output of each run goes to
runs/queue_logs/<name>_s<seed>.log. Ctrl+C stops the queue and its running jobs (they resume on
the next start).
"""

from __future__ import annotations

import argparse
import csv
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import yaml

from experiments.run import REPO_ROOT, apply_overrides, load_config


def expand(queue_file: str | Path, extra: list[str] | None = None) -> list[dict]:
    spec = yaml.safe_load(Path(queue_file).read_text())
    jobs = []
    for j in spec["jobs"]:
        cfg_path = REPO_ROOT / j["config"] if not Path(j["config"]).is_absolute() else Path(j["config"])
        sets = list(j.get("set", [])) + list(extra or [])
        cfg = apply_overrides(load_config(cfg_path), sets)
        for seed in j.get("seeds", [0]):
            log = Path(cfg["paths"]["log_csv"])
            jobs.append({"run": f"{cfg['name']}_s{seed}", "config": str(cfg_path), "seed": seed, "set": sets,
                         "mode": cfg["mode"], "group": cfg["name"] if j.get("serial") else None,
                         "log_csv": str(log if log.is_absolute() else REPO_ROOT / log)})
    names = [j["run"] for j in jobs]
    dup = {n for n in names if names.count(n) > 1}
    if dup:
        raise ValueError(f"duplicate run names in queue: {sorted(dup)}")
    return jobs


def log_status(log_csv: str | Path) -> dict[str, str]:
    """Latest status per run name in a runs/log.csv."""
    out = {}
    if Path(log_csv).exists():
        with open(log_csv) as f:
            for r in csv.DictReader(f):
                out[r["name"]] = r["status"]
    return out


def job_status(jobs: list[dict]) -> dict[str, str]:
    """Status of each job from the log.csv its own config writes to."""
    logs = {j["log_csv"]: log_status(j["log_csv"]) for j in jobs}
    return {j["run"]: logs[j["log_csv"]].get(j["run"], "not started") for j in jobs}


def command(job: dict) -> list[str]:
    cmd = [sys.executable, "-m", "experiments.run", "--config", job["config"], "--set", f"seed={job['seed']}"]
    for s in job["set"]:
        cmd += ["--set", s]
    return cmd


def next_job(pending: list[dict], running: list[dict]) -> dict | None:
    """First pending job whose serial group (if any) has no job running."""
    busy = {j["group"] for j in running if j.get("group")}
    return next((j for j in pending if not j.get("group") or j["group"] not in busy), None)


def available_gpus(arg: str | None) -> list[str]:
    if arg:
        return [g.strip() for g in arg.split(",") if g.strip()]
    try:
        import torch
        n = torch.cuda.device_count()
    except Exception:  # noqa: BLE001
        n = 0
    return [str(i) for i in range(n)] or ["cpu"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("queue")
    ap.add_argument("--gpus", default=None, help="comma-separated GPU ids (default: all visible)")
    ap.add_argument("--only", nargs="+", default=None, help="run names or config names to include")
    ap.add_argument("--set", dest="extra", action="append", default=[], help="override applied to every job")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    jobs = expand(args.queue, args.extra)
    if args.only:
        jobs = [j for j in jobs if j["run"] in args.only or j["run"].rsplit("_s", 1)[0] in args.only]
    status = job_status(jobs)
    done = [j for j in jobs if status.get(j["run"]) == "ok"]
    pending = [j for j in jobs if status.get(j["run"]) != "ok"]

    if args.status or args.dry_run:
        for j in jobs:
            print(f"  {j['run']:32s} {j['mode']:9s} {status[j['run']]}")
        print(f"{len(done)} done, {len(pending)} pending of {len(jobs)}")
        if args.dry_run:
            gpus = available_gpus(args.gpus)
            print(f"would run {len(pending)} jobs on GPUs {gpus}:")
            for j in pending:
                print("   ", " ".join(command(j)))
        return 0

    gpus = available_gpus(args.gpus)
    logs = REPO_ROOT / "runs" / "queue_logs"
    logs.mkdir(parents=True, exist_ok=True)
    print(f"queue: {len(done)} done, {len(pending)} to run on GPUs {gpus}", flush=True)
    running: dict[str, tuple[subprocess.Popen, dict, float]] = {}
    results: dict[str, str] = {}

    def stop(*_):
        for p, j, _ in running.values():
            p.send_signal(signal.SIGINT)
        print("\nqueue stopped; running jobs were interrupted and will resume on the next start", flush=True)
        sys.exit(130)

    signal.signal(signal.SIGINT, stop)
    while pending or running:
        for gpu in gpus:
            job = next_job(pending, [j for _, j, _ in running.values()]) if gpu not in running else None
            if job is not None:
                pending.remove(job)
                env = dict(os.environ)
                if gpu != "cpu":
                    env["CUDA_VISIBLE_DEVICES"] = gpu
                logf = open(logs / f"{job['run']}.log", "a")
                p = subprocess.Popen(command(job), cwd=REPO_ROOT, env=env, stdout=logf, stderr=subprocess.STDOUT)
                running[gpu] = (p, job, time.time())
                print(f"[{time.strftime('%H:%M:%S')}] start  {job['run']} on GPU {gpu}", flush=True)
        time.sleep(5)
        for gpu, (p, job, t0) in list(running.items()):
            if p.poll() is not None:
                ok = p.returncode == 0
                results[job["run"]] = "ok" if ok else f"failed (exit {p.returncode})"
                print(f"[{time.strftime('%H:%M:%S')}] {'done  ' if ok else 'FAILED'} {job['run']} on GPU {gpu} "
                      f"after {(time.time() - t0) / 60:.1f} min" + ("" if ok else f"; see runs/queue_logs/{job['run']}.log"),
                      flush=True)
                del running[gpu]
    failed = [r for r, s in results.items() if s != "ok"]
    print(f"queue finished: {len(results) - len(failed)} ok, {len(failed)} failed {failed if failed else ''}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
