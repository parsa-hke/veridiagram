# Changelog

## 2026-10-10 — Phase 1: environment and skeleton

Done
- 1.1 Repo initialised (git) at the project root with the §0.4 layout; `.gitignore` covers
  `data/raw`, `data/processed`, `data/cache`, `runs/`, checkpoint files, `.venv`, `wandb/`.
  Every module in §0.4 exists as a stub whose `NotImplementedError` names the phase that fills it.
- 1.2 Python 3.11.17 venv (`.venv`, via uv; Homebrew was unavailable because the Xcode licence
  is not accepted). Pins in `requirements.txt` and `pyproject.toml`: torch 2.14.1,
  torchvision 0.29.1, transformers 5.19.0, timm 1.0.30, numpy 2.4.6, pandas 3.0.6,
  matplotlib 3.11.2, pyyaml 6.0.3, scikit-learn 1.9.1, pytest 9.1.1, tqdm 4.70.1, wandb 0.30.0 (optional).
- 1.3 Device check recorded in README: Apple M5, MPS backend, 24 GB unified memory, no CUDA.
- 1.4 `configs/base.yaml` with every tunable; Phase 3 tolerances and not-to-scale thresholds left
  null. Thin override configs: smoke, probe, finetune, scaling_{1k,10k,50k}, selftrain_{confidence,consistency}.
- 1.5 `experiments/run.py`: `inherit:` resolution, `--smoke` overlay, `--set key=value` overrides,
  seeding (python, numpy, torch, PYTHONHASHSEED, cuDNN deterministic, deterministic algorithms with
  warn_only), dispatch to probe/finetune/evaluate/scaling/selftrain, writes metrics.json and a
  log.csv line for every run, including failed ones.
- 1.6 `tests/test_skeleton.py` (imports every module, checks no unlisted modules, dispatch targets,
  layout dirs) and `tests/test_run.py`. **Gate: `pytest` green, 59 passed in 2.2 s.**

Found
- **No CUDA GPU on this machine.** Training will run on MPS (Apple M5, 24 GB unified memory). A full
  fine-tune of ViT-L at 448 px (Phase 6.4, 7.x) will be much slower than on a CUDA card, and bf16
  autocast and some deterministic kernels behave differently on MPS. This needs a decision before
  Phase 6: cloud GPU for full runs vs scaled-down runs locally. §0.2 is unchanged for now.
- PyYAML reads `1e-5` (no decimal point) as a string under YAML 1.1. Configs use `1.0e-5` style;
  `--set` overrides coerce numeric strings. Covered by a test.
- Repo root is `veridiagram/` rather than the `geo-relations/` folder name in §0.4; contents match the layout.

## 2026-10-10 — Compute on Colab; branch renamed

Done
- Git branch renamed `master` → `main` (no commits existed yet).
- Heavy compute moves to Google Colab Pro; recorded as plan §0.6 (§0.2 unchanged). Local M5 keeps
  development, tests, data build, verifier and smoke runs.
- `experiments/run.py --overlay <yaml>` (repeatable) layers a partial config on top of any config;
  `--smoke` now uses the same mechanism. Order: config → smoke → overlays → `--set`.
- `configs/colab.yaml`: data and runs on `/content/drive/MyDrive/geo-relations/`, overlay cache on
  local `/content/cache`.
- `notebooks/colab_launcher.ipynb`: GPU check, Drive mount, clone/pull `main` (GH_TOKEN secret),
  install pins, HF cache on Drive + HF_TOKEN secret, smoke runs, 3-seed full run. Setup only, no logic.
- `pyproject.toml` `requires-python` relaxed to `>=3.11`: Colab ships its own Python version; pins
  are verified on 3.11 locally only.
- Tests: 60 passed in 2.2 s (new: Colab overlay keeps mode/name and only replaces paths).

Found / open
- The launcher has not been run on Colab yet. The GitHub repo `parsa-hke/veridiagram` does not exist
  yet and nothing is committed, so the clone cell will fail until the repo is pushed.
- Pinned wheels have not been checked against Colab's Python version (likely 3.12+); the install cell
  will show it.
- Phase 6 trainers must resume from the last epoch checkpoint (Colab disconnects).
