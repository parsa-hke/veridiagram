"""The website's HTTP API (WEBSITE_PLAN.md §4.2), FastAPI over web.server.core.

    POST /api/ask    {"construction": {...}} or {"image": <base64>, "primitives": {...}}, plus "queries"
    POST /api/read   the same without "queries": every query of the figure (capped), relation sets,
                     contradictions
    GET  /api/health {"ok", "ready", "model", "device"}; the site pings it on load (cold start)

    python -m web.server.app [--checkpoint runs/ablation_dinov2_s0/checkpoints/best.pt] [--port 7860]

Environment (for the Space): CHECKPOINT (a local path; if it does not exist and HF_MODEL_REPO is set,
best.pt, or HF_MODEL_FILE, is downloaded from that model repo with HF_TOKEN), CONFIG (default configs/ablation_dinov2.yaml), RUN_NAME,
DEVICE (auto), ALLOW_ORIGINS (comma-separated, default *), RATE_PER_MIN (30), READ_CAP (60),
STATIC_DIR (the built site, served at /). The model loads in a background thread, so /api/health
answers at once and reports ready=false while it warms up. Inputs are processed in memory and
never stored or logged.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Callable

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict

from experiments.run import REPO_ROOT, load_config
from web.server import core

MAX_BODY = 8 * 1024 * 1024


class Figure(BaseModel):
    model_config = ConfigDict(extra="forbid")
    points: dict[str, list[float]]
    segments: list[list[str]] = []
    circles: list[dict] = []


class ReadIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    construction: Figure | None = None
    image: str | None = None
    primitives: Figure | None = None


class AskIn(ReadIn):
    queries: list[dict]


class State:
    def __init__(self):
        self.engine: core.Engine | None = None
        self.error: str | None = None
        self.started = time.time()


class RateLimit:
    """At most `per_min` requests per client address in any 60 s window (in memory, per process)."""

    def __init__(self, per_min: int):
        self.per_min = per_min
        self.hits: dict[str, deque] = defaultdict(deque)

    def allow(self, who: str) -> bool:
        now = time.monotonic()
        q = self.hits[who]
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= self.per_min:
            return False
        q.append(now)
        return True


def default_loader() -> core.Engine:
    cfg = load_config(Path(os.environ.get("CONFIG", REPO_ROOT / "configs" / "ablation_dinov2.yaml")))
    run = os.environ.get("RUN_NAME", "ablation_dinov2_s0")
    ckpt = os.environ.get("CHECKPOINT", str(REPO_ROOT / "runs" / run / "checkpoints" / "best.pt"))
    if not Path(ckpt).exists() and os.environ.get("HF_MODEL_REPO"):     # the Space: fetch from a model repo
        from huggingface_hub import hf_hub_download
        ckpt = hf_hub_download(os.environ["HF_MODEL_REPO"], os.environ.get("HF_MODEL_FILE", "best.pt"),
                               token=os.environ.get("HF_TOKEN"))
    return core.Engine.from_checkpoint(cfg, ckpt, os.environ.get("DEVICE", "auto"), run)


def create_app(loader: Callable[[], core.Engine] = default_loader, static_dir: str | None = None,
               background: bool = True) -> FastAPI:
    app = FastAPI(title="Colour, not Letters: inference", docs_url=None, redoc_url=None)
    state = State()
    limit = RateLimit(int(os.environ.get("RATE_PER_MIN", "30")))
    read_cap = int(os.environ.get("READ_CAP", str(core.MAX_QUERIES)))
    origins = [o.strip() for o in os.environ.get("ALLOW_ORIGINS", "*").split(",") if o.strip()]
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"], allow_headers=["content-type"])

    def load() -> None:
        try:
            state.engine = loader()
        except Exception as e:  # noqa: BLE001 - logged here, reported generically by /api/health
            print(f"model load failed: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
            state.error = "the model could not be loaded"

    if background:
        threading.Thread(target=load, daemon=True).start()
    else:
        load()

    @app.middleware("http")
    async def guard(request: Request, call_next):
        if request.url.path.startswith("/api/") and request.method == "POST":
            if int(request.headers.get("content-length") or 0) > MAX_BODY:
                return JSONResponse({"error": "request larger than 8 MB"}, status_code=413)
            who = request.client.host if request.client else "?"
            if not limit.allow(who):
                return JSONResponse({"error": "too many questions; wait a minute"}, status_code=429)
        return await call_next(request)

    def prepared(body: ReadIn):
        e = state.engine
        if e is None:
            return None, JSONResponse({"error": "the instrument is warming up" if not state.error else state.error,
                                       "ready": False}, status_code=503)
        try:
            if (body.construction is None) == (body.image is None):
                raise core.InputError("send either a construction or an image with its primitives")
            if body.construction is not None:
                return core.from_construction(body.construction.model_dump(), e.cfg), None
            if body.primitives is None:
                raise core.InputError("an uploaded image needs its primitives (points, segments, circles)")
            return core.from_upload(body.image, body.primitives.model_dump()), None
        except core.InputError as err:
            return None, JSONResponse({"error": str(err)}, status_code=422)

    @app.get("/api/health")
    def health():
        e = state.engine
        return {"ok": state.error is None, "ready": e is not None, "model": e.name if e else None,
                "device": str(e.device) if e else None, "error": state.error,
                "uptime_s": round(time.time() - state.started)}

    @app.post("/api/ask")
    def ask(body: AskIn):
        p, err = prepared(body)
        if err:
            return err
        try:
            results, ms = core.timed(state.engine.ask, p, body.queries)
        except core.InputError as e:
            return JSONResponse({"error": str(e)}, status_code=422)
        return {"results": results, "model": state.engine.name, "ms": ms}

    @app.post("/api/read")
    def read(body: ReadIn):
        p, err = prepared(body)
        if err:
            return err
        try:
            out, ms = core.timed(state.engine.read, p, read_cap)
        except core.InputError as e:
            return JSONResponse({"error": str(e)}, status_code=422)
        return {**out, "model": state.engine.name, "ms": ms}

    static = static_dir or os.environ.get("STATIC_DIR")
    if static and Path(static).is_dir():
        app.mount("/", StaticFiles(directory=static, html=True), name="site")
    return app


def main(argv: list[str] | None = None) -> int:
    import argparse

    import uvicorn

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--static", default=None, help="serve the built site (web/site/dist) at /")
    args = ap.parse_args(argv)
    for k, v in (("CHECKPOINT", args.checkpoint), ("CONFIG", args.config), ("DEVICE", args.device)):
        if v:
            os.environ[k] = v
    uvicorn.run(create_app(static_dir=args.static), host=args.host, port=args.port, log_level="warning", access_log=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
