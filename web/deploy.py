"""Deploy the website to a Hugging Face Space (WEBSITE_PLAN.md §4.3, §6.6). Run by the user, with their
own credentials; nothing here stores or prints a token.

    python -m web.deploy space   [--out build/space]          the Space's two files (README card + Dockerfile)
    python -m web.deploy publish --space USER/NAME --model USER/NAME-model [--skip-checkpoint]
    python -m web.deploy check   --url https://USER-NAME.hf.space [--n 3]

The Space holds only a README and a Dockerfile: the Dockerfile clones this (private) GitHub repository
at a pinned commit with a read-only token given to the build as the Space secret GITHUB_TOKEN, builds
the site and the server exactly as web/server/Dockerfile does, and the server downloads best.pt from
a private model repo with the Space secret HF_TOKEN. Visitors see the website, not the code.

`publish` uses the Hugging Face login of this machine (`hf auth login`, a token with write access) and
reads two tokens for the Space from the environment, never from the command line:
  SPACE_GITHUB_TOKEN   fine-grained GitHub token, this repository only, Contents: read-only
  SPACE_HF_TOKEN       Hugging Face token that can read the model repo (fine-grained, read)
It creates (or reuses) the private model repo and uploads the checkpoint, creates (or reuses) the
public Docker Space on free CPU, sets its secrets and variables, and uploads the two files, which
starts the build. The pinned commit is the current HEAD, which must already be pushed.

`check` waits for the Space to be ready, asks it three validation questions as uploads and compares
its probabilities with the local engine (same checkpoint, CPU), then times single questions and a
chunked whole-figure read. Writes reports/tables/web_deploy_check.csv; `python -m web.export` puts
the measured latency in the colophon.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path

import yaml

from experiments.run import REPO_ROOT, load_config
from web.export import SITE_YAML

DEFAULT_REPO = "parsa-hke/veridiagram"
CHECK_CSV = REPO_ROOT / "reports" / "tables" / "web_deploy_check.csv"

README = """---
title: Colour, not Letters
emoji: 📐
colorFrom: red
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
short_description: A vision transformer reads the geometry in textbook diagrams
---

# Colour, not Letters

Teaching a vision transformer to read the geometry in textbook diagrams: draw a figure, point at two
of its parts, and three witnesses answer (the Ruler, the Eye, and the picture the Eye was shown).

{author}, {course}, {institution}, {year}. The image is built from the project's repository at
commit `{ref}`; uploaded images are processed in memory and never stored.
"""

DOCKERFILE = """# syntax=docker/dockerfile:1
# Built by `python -m web.deploy space` from web/server/Dockerfile: the same image, with the project's
# code cloned from a private repository at a pinned commit (build secret GITHUB_TOKEN, read-only).

FROM node:24-slim AS src
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates && rm -rf /var/lib/apt/lists/*
ARG REPO={repo}
ARG REF={ref}
RUN --mount=type=secret,id=GITHUB_TOKEN,mode=0444,required=true \\
    git init -q /src && cd /src \\
 && git fetch -q --depth 1 "https://x-access-token:$(cat /run/secrets/GITHUB_TOKEN)@github.com/$REPO.git" "$REF" \\
 && git checkout -q FETCH_HEAD && rm -rf /src/.git
WORKDIR /src/web/site
RUN npm ci --no-audit --no-fund && npm run build

FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \\
    HOME=/home/user HF_HOME=/home/user/.cache/huggingface MPLCONFIGDIR=/tmp/matplotlib
RUN useradd -m -u 1000 user
WORKDIR /home/user/app
COPY --from=src /src/web/server/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.14.1 \\
 && pip install --no-cache-dir -r /tmp/requirements.txt
COPY --from=src --chown=user /src/configs ./configs
COPY --from=src --chown=user /src/data ./data
COPY --from=src --chown=user /src/verifier ./verifier
COPY --from=src --chown=user /src/models ./models
COPY --from=src --chown=user /src/experiments ./experiments
COPY --from=src --chown=user /src/analysis ./analysis
COPY --from=src --chown=user /src/web/__init__.py ./web/__init__.py
COPY --from=src --chown=user /src/web/server ./web/server
COPY --from=src --chown=user /src/web/site/dist ./web/site/dist
USER user
RUN python -c "from transformers import AutoConfig; AutoConfig.from_pretrained('facebook/dinov2-large')"
EXPOSE 7860
CMD ["python", "-m", "web.server.app", "--host", "0.0.0.0", "--port", "7860", "--static", "web/site/dist"]
"""


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()


def pinned_commit() -> str:
    """HEAD, which must be pushed: the Space clones it from GitHub."""
    head = git("rev-parse", "HEAD")
    git("fetch", "-q", "origin")
    if subprocess.run(["git", "merge-base", "--is-ancestor", head, "origin/main"], cwd=REPO_ROOT).returncode:
        raise SystemExit(f"HEAD {head[:8]} is not on origin/main: push it first (git push origin main)")
    if git("status", "--porcelain", "--untracked-files=no"):
        print("note: there are uncommitted changes; the Space builds the pushed commit, without them")
    return head


def space_files(out: Path, repo: str, ref: str, site: dict) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    (out / "README.md").write_text(README.format(ref=ref[:12], **{k: site.get(k, "") for k in ("author", "course", "institution", "year")}))
    (out / "Dockerfile").write_text(DOCKERFILE.format(repo=repo, ref=ref))
    return [out / "README.md", out / "Dockerfile"]


def publish(args, site: dict) -> int:
    from huggingface_hub import HfApi

    gh, hf_read = os.environ.get("SPACE_GITHUB_TOKEN"), os.environ.get("SPACE_HF_TOKEN")
    if not gh or not hf_read:
        raise SystemExit("set SPACE_GITHUB_TOKEN and SPACE_HF_TOKEN in this shell first (see the guide)")
    api = HfApi()
    who = api.whoami()["name"]
    print(f"logged in to Hugging Face as {who}")
    ref = pinned_commit()
    ckpt = REPO_ROOT / "runs" / site["demo_run"] / "checkpoints" / "best.pt"
    api.create_repo(args.model, repo_type="model", private=True, exist_ok=True)
    print(f"model repo {args.model} (private) ready")
    if not args.skip_checkpoint:
        if not ckpt.exists():
            raise SystemExit(f"{ckpt} not found")
        print(f"uploading {ckpt.name} ({ckpt.stat().st_size / 1e6:.0f} MB) to {args.model} ...")
        api.upload_file(path_or_fileobj=str(ckpt), path_in_repo="best.pt", repo_id=args.model, repo_type="model",
                        commit_message=f"{site['demo_run']} best.pt")
    api.create_repo(args.space, repo_type="space", space_sdk="docker", private=False, exist_ok=True)
    print(f"Space {args.space} (public, Docker) ready")
    api.add_space_secret(args.space, "GITHUB_TOKEN", gh, description="read-only: clone the project at build time")
    api.add_space_secret(args.space, "HF_TOKEN", hf_read, description="read the private model repo")
    for k, v in {"HF_MODEL_REPO": args.model, "RUN_NAME": site["demo_run"], "READ_CAP": str(site.get("read_cap_cpu", 30))}.items():
        api.add_space_variable(args.space, k, v)
    out = REPO_ROOT / "build" / "space"
    space_files(out, args.repo, ref, site)
    api.upload_folder(repo_id=args.space, repo_type="space", folder_path=str(out),
                      commit_message=f"Build {args.repo}@{ref[:12]}")
    sub = args.space.replace("/", "-").replace("_", "-").lower()
    print(f"uploaded; the Space is building (about 10 minutes): https://huggingface.co/spaces/{args.space}")
    print(f"the website will be at https://{sub}.hf.space")
    return 0


# ------------------------------------------------------------------ check the deployed Space

def post(url: str, body: dict, timeout: float = 900) -> tuple[dict, float]:
    req = urllib.request.Request(url, json.dumps(body).encode(), {"content-type": "application/json"})
    t = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r), time.perf_counter() - t


def wait_ready(url: str, minutes: float = 15) -> dict:
    t0 = time.time()
    while time.time() - t0 < 60 * minutes:
        try:
            with urllib.request.urlopen(f"{url}/api/health", timeout=30) as r:
                h = json.load(r)
            if h.get("ready"):
                return h
            if not h.get("ok"):
                raise SystemExit(f"the Space cannot load the model: {h}")
            print("warming up ...")
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            print("waiting for the Space ...")
        time.sleep(15)
    raise SystemExit("the Space did not become ready")


EXAMPLE = {"points": {"A": [120, 120], "B": [480, 128], "C": [530, 330], "D": [70, 322]},
           "segments": [["A", "B"], ["B", "C"], ["C", "D"], ["D", "A"], ["A", "C"]]}


def check(args, site: dict) -> int:
    import numpy as np

    from data.dataset import QueryDataset
    from web.server import core

    url = args.url.rstrip("/")
    h = wait_ready(url)
    print(f"ready: {h['model']} on {h['device']}")
    cfg = load_config(REPO_ROOT / site["demo_config"])
    ds = QueryDataset(cfg, "val", train=False)
    local = core.Engine.from_checkpoint(cfg, REPO_ROOT / "runs" / site["demo_run"] / "checkpoints" / "best.pt", "cpu",
                                        site["demo_run"])
    raw = REPO_ROOT / cfg["paths"]["raw"]
    diffs, flips, asked = [], 0, 0
    for i in np.linspace(0, len(ds) - 1, args.n).astype(int):
        row = ds.rows[i]
        d = ds.diagrams[row["diagram_id"]]
        prims = {"points": {k: list(v) for k, v in d.points.items()}, "segments": [list(s) for s in d.segments],
                 "circles": [{"center": None, "center_xy": list(c.center_xy), "radius": c.radius} for c in d.circles
                             if c.center_xy is not None and c.radius]}
        b64 = base64.b64encode((raw / d.image_path).read_bytes()).decode()
        remote, _ = post(f"{url}/api/ask", {"image": b64, "primitives": prims, "queries": [{"kind": row["kind"], "refs": row["refs"]}]})
        mine = local.ask(core.from_upload(b64, prims), [{"kind": row["kind"], "refs": row["refs"]}], images=False)[0]
        rp = remote["results"][0]["probs"]
        for rel, p in mine["probs"].items():
            diffs.append(abs(p - rp[rel]))
            flips += (p > 0.5) != (rp[rel] > 0.5)
            asked += 1
    print(f"parity on {args.n} validation questions: max |dp| {max(diffs):.4f}, decisions differing {flips} of {asked}")
    one = []
    q = [{"kind": "seg_pair", "refs": [["seg", "A", "B"], ["seg", "C", "D"]]}]
    for _ in range(args.repeats):
        _, s = post(f"{url}/api/ask", {"construction": EXAMPLE, "queries": q})
        one.append(s)
    total, n_read, t = 0, 0, time.perf_counter()
    off = 0
    while True:
        r, _ = post(f"{url}/api/read", {"construction": EXAMPLE, "offset": off, "limit": 4})
        n_read += len(r["results"])
        total = r["queries_asked"]
        off += 4
        if off >= total:
            break
    read_s = time.perf_counter() - t
    row = {"date": date.today().isoformat(), "url": url, "model": h["model"], "device": h["device"],
           "parity_n": args.n, "parity_max_dp": round(max(diffs), 4), "parity_decisions_differing": flips,
           "parity_answers": asked, "ask_median_s": round(statistics.median(one), 2), "ask_runs": len(one),
           "read_queries": n_read, "read_s": round(read_s, 1), "read_per_query_s": round(read_s / max(n_read, 1), 2)}
    CHECK_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(CHECK_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        w.writeheader()
        w.writerow(row)
    print(json.dumps(row, indent=1))
    print(f"-> {CHECK_CSV}; now run python -m web.export so the colophon shows it")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("space", help="write the Space's README and Dockerfile")
    s.add_argument("--out", default=str(REPO_ROOT / "build" / "space"))
    s.add_argument("--repo", default=DEFAULT_REPO)
    s.add_argument("--ref", default=None, help="commit to build (default: HEAD, which must be pushed)")
    p = sub.add_parser("publish", help="create the repos, set the Space's secrets, upload, build")
    p.add_argument("--space", required=True, help="USER/NAME of the Space")
    p.add_argument("--model", required=True, help="USER/NAME of the private model repo for best.pt")
    p.add_argument("--repo", default=DEFAULT_REPO)
    p.add_argument("--skip-checkpoint", action="store_true", help="the checkpoint is already uploaded")
    c = sub.add_parser("check", help="deployed parity and latency")
    c.add_argument("--url", required=True)
    c.add_argument("--n", type=int, default=3)
    c.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args(argv)
    site = yaml.safe_load(SITE_YAML.read_text())
    if args.cmd == "space":
        ref = args.ref or pinned_commit()
        for f in space_files(Path(args.out), args.repo, ref, site):
            print(f"-> {f}")
        return 0
    if args.cmd == "publish":
        return publish(args, site)
    return check(args, site)


if __name__ == "__main__":
    sys.exit(main())
