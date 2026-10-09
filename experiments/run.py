"""Single entry point for every experiment.

    python -m experiments.run --config configs/probe.yaml [--smoke] [--overlay configs/colab.yaml] [--set seed=1 ...]
    python -m experiments.run --device-info

Loads the config (with `inherit:` chains), sets all seeds, then dispatches on
`mode` to probe / finetune / evaluate / scaling / selftrain. Every run writes
runs/<name>_s<seed>/{config.yaml,metrics.json} and appends one line to runs/log.csv.
"""

from __future__ import annotations

import argparse
import csv
import copy
import hashlib
import importlib
import json
import os
import platform
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE_CONFIG = REPO_ROOT / "configs" / "smoke.yaml"

# mode -> "module:function"; each target has signature run(cfg, run_dir) -> metrics dict.
DISPATCH: dict[str, str] = {
    "probe": "models.probe:run",
    "finetune": "models.finetune:run",
    "evaluate": "experiments.evaluate:run",
    "scaling": "experiments.scaling:run",
    "selftrain": "experiments.selftrain:run",
}

LOG_FIELDS = [
    "timestamp", "name", "mode", "seed", "config_hash",
    "status", "best_epoch", "duration_s", "metrics",
]


# --------------------------------------------------------------------------- config

def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge `override` into a copy of `base`; override wins."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: str | Path) -> dict:
    """Load a YAML config, resolving `inherit:` (path relative to the file) recursively."""
    path = Path(path).resolve()
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    parent = cfg.pop("inherit", None)
    if parent is None:
        return cfg
    return deep_merge(load_config(path.parent / parent), cfg)


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """Apply `a.b.c=value` overrides; values are parsed as YAML (so 1e-3, true, [1,2] work)."""
    cfg = copy.deepcopy(cfg)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override must be key=value, got {item!r}")
        dotted, raw = item.split("=", 1)
        keys = dotted.split(".")
        node = cfg
        for k in keys[:-1]:
            if not isinstance(node.get(k), dict):
                raise KeyError(f"override {dotted!r}: {k!r} is not a config section")
            node = node[k]
        if keys[-1] not in node:
            raise KeyError(f"override {dotted!r}: unknown key {keys[-1]!r}")
        node[keys[-1]] = _parse_value(raw)
    return cfg


def _parse_value(raw: str) -> Any:
    value = yaml.safe_load(raw)
    if isinstance(value, str):
        # PyYAML follows YAML 1.1, which reads `1e-5` (no dot) as a string.
        try:
            return float(value)
        except ValueError:
            pass
    return value


def apply_overlay(cfg: dict, path: str | Path) -> dict:
    """Layer only the overlay file's own keys (not its `inherit` base), so mode and name survive."""
    with open(path) as f:
        layer = yaml.safe_load(f) or {}
    layer.pop("inherit", None)
    layer.pop("name", None)
    return deep_merge(cfg, layer)


def resolve_config(config: str | Path, smoke: bool = False, overrides: list[str] | None = None,
                   overlays: list[str | Path] | None = None) -> dict:
    """config -> smoke overlay -> other overlays (e.g. configs/colab.yaml) -> --set overrides."""
    cfg = load_config(config)
    if smoke:
        name = cfg.get("name", "run")
        cfg = apply_overlay(cfg, SMOKE_CONFIG)
        cfg["name"] = name if name.startswith("smoke") else f"smoke_{name}"
    for path in overlays or []:
        cfg = apply_overlay(cfg, path)
    cfg = apply_overrides(cfg, overrides or [])
    if cfg.get("mode") not in DISPATCH:
        raise ValueError(f"unknown mode {cfg.get('mode')!r}; expected one of {sorted(DISPATCH)}")
    return cfg


def config_hash(cfg: dict) -> str:
    """Short stable hash of the resolved config, ignoring the seed (so seeds group together)."""
    payload = {k: v for k, v in cfg.items() if k != "seed"}
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:10]


# --------------------------------------------------------------------------- environment

def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed python, numpy and torch (all devices); optionally force deterministic kernels."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds CUDA and MPS generators
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # warn_only: some MPS/CUDA ops have no deterministic kernel; warn rather than crash.
        torch.use_deterministic_algorithms(True, warn_only=True)


def get_device(preference: str = "auto") -> torch.device:
    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_info() -> dict[str, Any]:
    """Device, memory and CUDA version (plan task 1.3)."""
    dev = get_device()
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "device": str(dev),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "mps_available": torch.backends.mps.is_available(),
    }
    if dev.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        info["gpu_name"] = props.name
        info["vram_gb"] = round(props.total_memory / 1024**3, 1)
        info["cudnn_version"] = torch.backends.cudnn.version()
    elif dev.type == "mps":
        # Apple silicon: GPU shares unified memory with the CPU; there is no separate VRAM.
        try:
            import subprocess
            mem = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]).strip())
            chip = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
            info["gpu_name"] = f"{chip} (integrated, Metal/MPS)"
            info["unified_memory_gb"] = round(mem / 1024**3, 1)
            info["mps_recommended_max_gb"] = round(torch.mps.recommended_max_memory() / 1024**3, 1)
        except Exception:  # pragma: no cover - informational only
            pass
    return info


# --------------------------------------------------------------------------- logging

def append_log(log_csv: Path, row: dict[str, Any]) -> None:
    log_csv.parent.mkdir(parents=True, exist_ok=True)
    new = not log_csv.exists()
    with open(log_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if new:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in LOG_FIELDS})


def resolve_target(mode: str) -> Callable[[dict, Path], dict]:
    module_name, fn_name = DISPATCH[mode].split(":")
    return getattr(importlib.import_module(module_name), fn_name)


# --------------------------------------------------------------------------- main

def run(cfg: dict, dry_run: bool = False) -> dict:
    """Seed, create the run dir, dispatch, and record metrics + log line. Returns metrics."""
    seed = int(cfg["seed"])
    set_seed(seed, cfg.get("deterministic", True))

    runs_root = REPO_ROOT / cfg["paths"]["runs"]
    run_name = f"{cfg['name']}_s{seed}"
    run_dir = runs_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = {**cfg, "run_dir": str(run_dir), "device_resolved": str(get_device(cfg.get("device", "auto")))}
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    chash = config_hash({k: v for k, v in cfg.items() if k not in ("run_dir", "device_resolved")})
    started = time.time()
    status, metrics = "ok", {}
    try:
        if dry_run:
            status = "dry_run"
        else:
            metrics = resolve_target(cfg["mode"])(cfg, run_dir) or {}
    except BaseException as exc:
        status = f"failed:{type(exc).__name__}"
        metrics = {"error": str(exc), "traceback": traceback.format_exc()}
        raise
    finally:
        duration = round(time.time() - started, 2)
        record = {
            "name": run_name, "mode": cfg["mode"], "seed": seed, "config_hash": chash,
            "status": status, "duration_s": duration, **metrics,
        }
        with open(run_dir / "metrics.json", "w") as f:
            json.dump(record, f, indent=2, default=str)
        log_metrics = {k: v for k, v in metrics.items() if k not in ("traceback", "best_epoch")}
        append_log(REPO_ROOT / cfg["paths"]["log_csv"], {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "name": run_name, "mode": cfg["mode"], "seed": seed, "config_hash": chash,
            "status": status, "best_epoch": metrics.get("best_epoch", ""),
            "duration_s": duration, "metrics": json.dumps(log_metrics, default=str),
        })
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", help="path to a YAML config in configs/")
    parser.add_argument("--smoke", action="store_true", help="overlay configs/smoke.yaml (50 diagrams, 1 epoch)")
    parser.add_argument("--overlay", dest="overlays", action="append", default=[], metavar="YAML",
                        help="layer another config's keys on top, e.g. --overlay configs/colab.yaml")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                        help="override a config value, e.g. --set seed=1 --set finetune.epochs=5")
    parser.add_argument("--dry-run", action="store_true", help="resolve config, seed, write run dir; skip training")
    parser.add_argument("--device-info", action="store_true", help="print device / memory / CUDA info and exit")
    args = parser.parse_args(argv)

    if args.device_info:
        for k, v in device_info().items():
            print(f"{k:24s} {v}")
        return 0
    if not args.config:
        parser.error("--config is required (or use --device-info)")

    cfg = resolve_config(args.config, smoke=args.smoke, overrides=args.overrides, overlays=args.overlays)
    record = run(cfg, dry_run=args.dry_run)
    print(json.dumps({k: v for k, v in record.items() if k != "traceback"}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
