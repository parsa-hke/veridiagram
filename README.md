# geo-relations: geometry diagram → formal relations

MAAI 901 Assessment 2 (Applied Vision Transformer Report). A fine-tuned ViT reads formal
geometric relations (parallel, perpendicular, equal length, on-circle, tangent, …) from
textbook diagrams; correctness is judged by a coordinate-based verifier. The full plan is in
[GEOMETRY_RELATIONS_PLAN.md](GEOMETRY_RELATIONS_PLAN.md); progress in [CHANGELOG.md](CHANGELOG.md).

## Setup

Python 3.11 (pinned in `requirements.txt` / `pyproject.toml`). With [uv](https://docs.astral.sh/uv/):

```bash
uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -r requirements.txt
```

or with any Python 3.11: `python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt`.

## Usage

Every experiment goes through one entry point and one config:

```bash
.venv/bin/python -m experiments.run --config configs/probe.yaml            # full run
.venv/bin/python -m experiments.run --config configs/probe.yaml --smoke    # 50 diagrams, 1 epoch
.venv/bin/python -m experiments.run --config configs/finetune.yaml --set seed=1 --set finetune.epochs=5
.venv/bin/python -m experiments.run --device-info
```

`configs/base.yaml` holds every tunable; other configs override parts of it via `inherit:`.
Each run writes `runs/<name>_s<seed>/{config.yaml,metrics.json}` and appends one line to
`runs/log.csv` (timestamp, name, mode, seed, config hash, status, best epoch, duration, metrics).

Tests: `.venv/bin/python -m pytest`.

## Running on Google Colab

Heavy runs (fine-tunes, scaling, self-training) run on Colab Pro. Open
[notebooks/colab_launcher.ipynb](notebooks/colab_launcher.ipynb) in Colab, pick an A100 runtime
(L4 is fine for probes and smoke runs; avoid T4, which lacks bf16), add the `GH_TOKEN` (while the
repo is private) and `HF_TOKEN` secrets, and run the cells. The notebook mounts Drive, clones
`main`, installs the pins and calls the same entry point with the Colab overlay:

```bash
python -m experiments.run --config configs/finetune.yaml --overlay configs/colab.yaml --set seed=0
```

`configs/colab.yaml` puts raw/processed data and `runs/` under `MyDrive/geo-relations/`, and keeps
the overlay PNG cache on the VM's local disk.

## Hardware

Recorded 2026-10-10 with `python -m experiments.run --device-info`:

| Item | Value |
|------|-------|
| Machine | Apple M5, macOS 27.0.1 (arm64) |
| Device | `mps` (Metal Performance Shaders, integrated GPU) |
| VRAM | No dedicated VRAM: 24.0 GB unified memory, MPS recommended max 17.8 GB |
| CUDA | Not available (`torch.version.cuda = None`) |
| Python / torch | 3.11.17 / 2.14.1 |

Runtimes per step will be added in Phase 10.
