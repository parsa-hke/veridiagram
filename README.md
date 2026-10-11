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

## One-command data setup

```bash
.venv/bin/python -m data.prepare --download     # download + verify archives, build, splits, synthetic pools, labels, query indexes
```

## Models (Phase 6)

```bash
.venv/bin/python -m experiments.run --config configs/probe.yaml --smoke      # frozen DINOv3 features + linear head
.venv/bin/python -m experiments.run --config configs/finetune.yaml --smoke   # full fine-tune (LLRD, cosine, bf16, EMA)
.venv/bin/python -m experiments.run --config configs/finetune.yaml --set seed=1
```

DINOv3 is gated: run `.venv/bin/hf auth login` once (or set `HF_TOKEN`); without access the
DINOv2-L fallback is used and recorded in `metrics.json` (`backbone_loaded`). Each run writes
`config.yaml`, `metrics.json`, `history.jsonl`, `predictions/<unit>.npz` and, for fine-tunes,
`checkpoints/{best,last}.pt`; an interrupted fine-tune resumes from `last.pt`.

## Data

Download and layout: [data/README.md](data/README.md). After the raw files are in `data/raw/`:

```bash
.venv/bin/python -m data.build && .venv/bin/python -m data.overlap && .venv/bin/python -m data.splits && .venv/bin/python -m data.census && .venv/bin/python -m data.sanity
```

## Verifier

```bash
.venv/bin/python -m verifier.tolerance   # margin histograms + calibration table (--write-config to overwrite configs/base.yaml)
.venv/bin/python -m verifier.label       # geometric labels + not-to-scale flags for every set
.venv/bin/python -m verifier.audit       # asserted vs geometric tables and example figures (Result 1)
.venv/bin/python -m verifier.gold sample # 100 gold queries + labelling page verifier/gold/label.html
.venv/bin/python -m verifier.gold score  # after labelling: FP/FN rates -> reports/tables/verifier_audit.csv
```

## Queries and dataset

```bash
.venv/bin/python -m data.queries                                   # query indexes for every unit (base config)
.venv/bin/python -m data.dataset --config configs/smoke.yaml --examples   # Phase 5 gate + reports/figures/dataset_examples.png
```

## Synthetic data

```bash
.venv/bin/python -m data.synthetic.build                                   # 50k train + 1k extrap + 2k unlabelled (~10 min, 8 workers)
.venv/bin/python -m verifier.label --sets synth_train synth_extrap         # geometric labels
.venv/bin/python -m data.synthetic.check                                   # injected vs random pass rates, example figure
```

## Running on the UOWD AI Lab GPUs (Phase 7)

Step by step: [LAB_GUIDE.md](LAB_GUIDE.md). In short: `bash scripts/lab_setup.sh`, then
`python -m experiments.benchmark` and `python -m experiments.queue configs/queue_phase7.yaml`
(one run per GPU; restartable: finished runs are skipped, interrupted fine-tunes resume).

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

Recorded 2026-10-10 with `python -m experiments.run --device-info`.

| Item | Local (development, smoke) | Google Colab Pro (heavy runs) |
|------|------|------|
| Machine | Apple M5, macOS 27.0.1 (arm64) | Linux x86_64, glibc 2.39 |
| Device | `mps` (Metal, integrated GPU) | `cuda`, NVIDIA L4 (A100 for full fine-tunes) |
| VRAM | none dedicated: 24.0 GB unified, MPS max 17.8 GB | 22.0 GB (L4) |
| CUDA / cuDNN / driver | not available | CUDA 13.0 / cuDNN 9.24 / 580.82.07 |
| Python / torch | 3.11.17 / 2.14.1 | 3.13.15 / 2.14.1+cu130 |

Runtimes per step will be added in Phase 10.
