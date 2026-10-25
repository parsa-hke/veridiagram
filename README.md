# geo-relations: geometry diagram → formal relations

MAAI 901 Assessment 2 (Applied Vision Transformer Report). A fine-tuned ViT reads formal
geometric relations (parallel, perpendicular, equal length, on-circle, tangent, …) from
textbook diagrams; correctness is judged by a coordinate-based verifier. The full plan is in
[GEOMETRY_RELATIONS_PLAN.md](GEOMETRY_RELATIONS_PLAN.md); progress in [CHANGELOG.md](CHANGELOG.md).

## Reproducing the project in four commands

From a fresh clone (Python 3.11 and [uv](https://docs.astral.sh/uv/) installed; the Hugging Face
login is only needed for DINOv3, which is gated):

```bash
uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -r requirements.txt   # 1. setup
.venv/bin/python -m data.prepare --download                                                    # 2. data
for q in queue_phase7 queue_phase7_colab queue_phase7_final; do .venv/bin/python -m experiments.queue configs/$q.yaml || break; done   # 3. training (GPU)
.venv/bin/python -m scripts.reproduce                                                          # 4. every table and figure
```

Without uv: `python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt`. Step 2
downloads and checks the three datasets (SHA-256), builds the unified records, the figure-level
splits, the synthetic pools, the verifier labels and the query indexes. Step 3 is the only one that
needs a GPU. The three queues are every Phase 7 run and must run in this order: the first trains the
probe, 7.2, 7.3 and two ablations (the lab ran it), the second Part A, Part B and the other ablations
(it needs 7.3 seed 2 as the self-training teacher), the third the last control seeds, the retrained
7.2 and the square-padded re-evaluation of every run that kept its checkpoint (Colab ran both). Each
queue runs one experiment per GPU, skips finished runs and resumes interrupted ones; on Colab set
`GEO_OVERLAYS=configs/colab.yaml` first (Drive paths), as the launcher notebook and
[COLAB_GUIDE.md](COLAB_GUIDE.md) do. Step 4 regenerates everything in `reports/tables/` and `reports/figures/`, the
generated numbers of both LaTeX reports, their figure copies and Overleaf zips, and checks the
report's word count; `--list` shows each step and whether its inputs exist, `--stages data` (or
`analysis`, `report`) runs part of it. A step whose inputs are missing is skipped with the reason,
so on a clone without `runs/` the data and report stages still run.

Then `.venv/bin/python -m pytest` (the whole suite runs in under 30 s; the LaTeX compile test
is skipped without `tectonic`).

### Fresh-clone check (plan 10.2)

On 2026-10-25 the repository was cloned from GitHub into an empty directory (commit `da29eea`) and
set up with a new virtualenv from `requirements.txt` (plain `venv` + `pip`); the datasets were
downloaded again and verified by SHA-256, `data.prepare` rebuilt everything, the test suite ran
(226 passed, 5 skipped, 0 failed) and both smoke runs passed the gate
(`python -m experiments.gate smoke`: DINOv3 loaded, probe 187 s, fine-tune 399 s, predictions for
every unit, checkpoint written). The 33 files the clone built under `data/processed/` (records,
splits, labels, query indexes) are byte-identical to the ones the experiments used. Times on the M5
laptop: pip install 61 s, download 34 s, data build 393 s (synthetic pools 203 s, labels 91 s), tests
91 s. The Hugging Face cache (DINOv3 weights) and the pip cache were shared with the existing
checkout, so model download time is not included.

### Expected runtime per step

| Step | Command | Time | Measured on |
|---|---|---|---|
| Setup | `uv venv … && uv pip install -r requirements.txt` | 61 s with a warm pip cache (`venv` + `pip`) | M5 laptop |
| Data: download | `python -m data.prepare --download` (download part) | 34 s (network-bound) | M5 laptop |
| Data: build, overlap, splits, census | (inside `data.prepare`) | about 1 min (build 12 s, overlap 36 s) | M5 laptop |
| Data: synthetic pools | (inside `data.prepare`) | about 4 min (8 workers) | M5 laptop |
| Data: verifier labels, query indexes | (inside `data.prepare`) | about 1.5 min (real 32 s, synthetic 57 s, queries 7 s) | M5 laptop |
| Smoke runs (probe + fine-tune), then `python -m experiments.gate smoke` | `python -m experiments.run --config configs/{probe,finetune}.yaml --smoke` | 97 s each / 187 s + 399 s | Colab L4 / M5 laptop (MPS) |
| Linear probe, one seed | `python -m experiments.run --config configs/probe.yaml` | 7 min | lab RTX 5000 Ada |
| Fine-tune, one seed (LP-FT, 80k images) | `python -m experiments.run --config configs/finetune.yaml` | 40-45 min / 75 min | Colab A100 / lab RTX 5000 Ada |
| Square-padded re-evaluation, one run | `python -m experiments.run --config configs/eval_square.yaml …` | 7 min | Colab A100 |
| All Phase 7 runs (53 runs incl. re-evaluations) | the three queues | 23.9 GPU-hours in total | lab + Colab |
| Every table and figure | `python -m scripts.reproduce` | 2.8 min | M5 laptop (MPS for the attention maps) |
| Test suite | `python -m pytest` | 27 s | M5 laptop |

The reproduction time is measured by the script itself on every run and written to
`reports/tables/reproduce_timings.csv` (per step: the slowest are the source-overlap clustering,
36 s, the synthetic verifier check, 38 s, attention maps, 30 s, and the bootstrap CIs, 28 s). The
training rows are the means of the `duration_s` column of `runs/log.csv`.

### What is not in the repository

`data/raw/`, `data/processed/`, `data/cache/` and `runs/` are gitignored (3.1 GB, 0.4 GB, 0.2 GB and
3.5 GB). `runs/` holds every run's `config.yaml`, `metrics.json`, `history.jsonl` and predictions; the
only checkpoint kept is `runs/finetune_synth10k_s2/checkpoints/best.pt` (the Part B teacher, used for
the attention maps and the website). Every committed table can be regenerated from those directories
with step 4; without them the committed tables are the record.

## Setup (details)

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

## Phase 7 experiments

```bash
python -m experiments.run --config configs/ablation_nooverlay.yaml        # 7.8: query as point-name tokens, no highlight
python -m experiments.run --config configs/selftrain_consistency.yaml     # 7.11: Part B student (also _confidence, _control)
python -m experiments.gate phase7                                         # smoke gate of those modes
python -m analysis.phase7                                                 # tables, not-to-scale subset, scaling.png, padding effect
python -m analysis.phase8                                                 # 8.1: bootstrap CIs, ECE, val-tuned thresholds, per-relation F1
python -m analysis.headline_table                                         # 8.2: queries solved by probe only / fine-tune only / both / neither
python -m analysis.error_analysis                                         # 8.4: F1 by points, segment length, source, not-to-scale; accuracy vs verifier margin
python -m analysis.attention_maps                                         # 8.5: attention maps of 5 correct + 5 wrong decisions, attention on the highlight (needs a checkpoint)
python -m analysis.asserted_vs_geometric                                  # 8.3: both label sets side by side, label disagreement cells, audit examples with the model's answer
python -m analysis.figures                                                # 8.6: per-relation F1 bars, headline figure, reports/figures/README.md (the figure manifest)
```

Part B (`experiments/selftrain.py`) uses the best 7.3 run as teacher (`selftrain.teacher_run`),
pseudo-labels the PGPS9K pool figures that are not training figures, filters by confidence or by
confidence plus deductive consistency (`verifier/consistency.py`) and fine-tunes a student from the
teacher's weights; the post-hoc pseudo-label precision against the sealed verifier labels is in its
metrics.json. The lab ran the first batch (`configs/queue_phase7.yaml`); the rest runs on Colab
(`configs/queue_phase7_colab.yaml`, see below).

## Running on the UOWD AI Lab GPUs (Phase 7)

Step by step: [LAB_GUIDE.md](LAB_GUIDE.md). In short: `bash scripts/lab_setup.sh`, then
`python -m experiments.benchmark` and `python -m experiments.queue configs/queue_phase7.yaml`
(one run per GPU; restartable: finished runs are skipped, interrupted fine-tunes resume).

## Running on Google Colab

Heavy runs (fine-tunes, scaling, self-training) run on Colab Pro; step by step for the Phase 7
queue: [COLAB_GUIDE.md](COLAB_GUIDE.md). Open
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

Recorded 2026-10-10 with `python -m experiments.run --device-info` (local and Colab) and
2026-10-12 with `python -m experiments.benchmark` (lab).

| Item | Local (development, data, analysis) | UOWD AI Lab (first Phase 7 batch) | Google Colab Pro (remaining runs) |
|------|------|------|------|
| Machine | Apple M5, macOS 27.0.1 (arm64) | shared classroom PC, 2-3 h per visit | Linux x86_64, glibc 2.39 |
| Device | `mps` (Metal, integrated GPU) | 4x NVIDIA RTX 5000 Ada Generation, one run per GPU | `cuda`, NVIDIA A100 (L4 for probes and smoke runs) |
| Memory | 24.0 GB unified, MPS max 17.8 GB | a ViT-L fine-tune peaks at 5.5 GB | 22.0 GB (L4) |
| Throughput (ViT-L, 448 px) | ~25 s per fine-tune step (development only) | 22.5 train / 98.5 eval images per second | about 2x the lab |
| CUDA / cuDNN / driver | not available | not recorded | CUDA 13.0 / cuDNN 9.24 / 580.82.07 |
| Python / torch | 3.11.17 / 2.14.1 | the same `requirements.txt` pins (`scripts/lab_setup.sh`) | 3.13.15 / 2.14.1+cu130 |
