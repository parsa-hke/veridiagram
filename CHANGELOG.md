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

## 2026-10-10 — Repo published; Phase 2 pre-checks

Done
- Initial commit `3c7eefd` pushed to private repo https://github.com/parsa-hke/veridiagram (default branch `main`).

Found (pre-checks for Phase 2 and 6, from public pages; not yet confirmed by downloading)
- Backbone: `facebook/dinov3-vitl16-pretrain-lvd1689m` exists on the HF hub, `transformers`
  library, access **gated (manual approval)**. Fallback `facebook/dinov2-large` is not gated.
- Geometry3K: Inter-GPS repo (lupantech/InterGPS), Google Drive folder linked from its README; repo MIT.
- PGPS9K: HF dataset `PeijieWang/PGPS9K` (~133 MB, MIT); the PGPS README says the zip uses a password,
  which is given in that README.
- PGDP5K: official page http://www.nlpr.ia.ac.cn/databases/CASIA-PGDP5K/ ; HF copy `PeijieWang/PGDP5K`
  (~85 MB, MIT); code at mingliangzhang2018/PGDP.
- **Overlap risk for §0.2 #9:** the PGDP5K paper says 1,813 of its 5,000 images come from Geometry3K.
  Using PGDP5K test as a "held-out source" needs de-duplication against Geometry3K (all splits),
  otherwise the held-out result is contaminated. To resolve in Phase 2 before splits are written.

## 2026-10-10 — Colab setup verified (L4)

Done
- Launcher cells 1–6 ran on Colab Pro, L4 runtime: Drive mounted, private repo cloned at `a2ae872`
  via GH_TOKEN, HF_TOKEN read, `--device-info` OK. Recorded in README hardware table: NVIDIA L4
  22.0 GB, CUDA 13.0, cuDNN 9.24, driver 580.82.07, Python 3.13.15, torch 2.14.1+cu130.
- Repinned numpy 2.4.6 → 2.2.6 and pandas 3.0.6 → 2.2.3: the old pins overwrote Colab's
  preinstalled versions and broke google-colab (needs pandas==2.2.3) and numba (needs numpy<2.3).
  The project code depends on neither newer version. Local venv reinstalled, `uv pip check` clean,
  60 tests pass.

Found / open
- Python differs: 3.11 locally vs 3.13 on Colab. Torch is identical (2.14.1).
- Remaining pip warnings on Colab (dask-cudf, cudf, google-adk/opentelemetry) concern packages this
  project never imports. To be re-checked on the next Colab install.
- DINOv3 access requested on HF (manual approval pending).

## 2026-10-10 — Phase 2: data acquisition and unified schema

Done
- 2.1 Downloaded Geometry3K (Inter-GPS GitHub zips), PGDP5K and PGPS9K (Hugging Face copies, no
  password needed) into `data/raw/` (gitignored). URLs, licences (MIT on repo/dataset cards),
  verified citations in `data/README.md`. sha256:
  geometry3k train e12ee4d2ad85117a4d4279ce46917ac5f2a3aaa2266d4413118b6087a151a95b,
  val 67b1b0a9b424cb5e6634d20a6494a554c529156a6cfd0492c636e047e13f314a,
  test a0d5dece634686303eb6cabca0a7306136ca8ade5f562a4af4c6561ecaf6c9b3,
  logic_forms 264cd7316056d7ee57171b57397e3555c97e40d648a4e901db7baec59423665d,
  symbols 90966955e3d8a81b6d171ad820ffbb8223e3a6a3aa8f54307c94ef4ea6ef65bb;
  PGDP5K.zip 2fe9c0c361f231591e3727360a69fa7a639a67ee8b1c13da549c12af0b80f60f;
  PGPS9K.zip a74b2b819ed4dff91e870b17a33b7860de5f118b254c1f48e8be0fe10ab9bc21.
- 2.2 Raw formats documented field by field in `data/README.md` from hand inspection.
- 2.3 `data/schema.py`: Diagram / Relation / Circle with tagged primitive refs (`["seg", A, B]`, ...)
  because point names can be multi-character. Added fields beyond the plan sketch: `split`,
  `circles[i].center_xy`, `meta`.
- 2.4 Loaders `data/loaders/{geometry3k,pgdp5k,pgps9k}.py` plus a shared PGDP-format parser
  (`pgdp_format.py`) and helpers (`common.py`). `tests/test_loaders.py` round-trips 3 diagrams per
  source and renders `reports/figures/loader_check_<source>.png` (checked by eye).
- 2.5 `python -m data.build` writes `data/processed/<source>.jsonl` in 12 s: 3002 / 5000 / 4000 diagrams.
- 2.6 **Gate passed.** `python -m data.census` -> `reports/tables/relation_census.csv`. Tier 2 decided
  on Geometry3K train, unique figures: collinear 4528 include; midpoint 50, equal_angle 53,
  perp_bisector 5 exclude. Recorded in plan §0.2 #3 and `configs/base.yaml` (`tier2_included`).
- 2.7 `python -m data.sanity`: points outside the image: Geometry3K 15 of 16718 (0.09%, 11 diagrams),
  PGDP5K 0, PGPS9K 0; below the 1% investigation threshold. Image sizes similar across sources
  (median ~370x275 px, so the 448 px input upsamples most images).
- Extra: `python -m data.overlap` finds duplicate images within and across sources (dHash candidates,
  confirmed by pixel difference, ink IoU and point-name agreement; the low-resolution check alone
  matched different simple triangles, so the rule was calibrated on the borderline cases by eye).
- Tests: 79 passed in 4.9 s.

Found
- **Geometry3K coordinates are not pixels**: each diagram's `point_positions` are translated by an
  unknown offset (e.g. +123/+124 px). Recovered from the red dots in `img_diagram_point.png`:
  2985/3001 full match, 16 partial (flagged). Before recovery 63 diagrams had negative coordinates.
- **Geometry3K positions are noisy**: against PGDP5K's pixel re-annotation of the same images
  (7620 points, 1641 image pairs) the median error is 2.4 px, p95 8.9 px, p99 89 px; 4.5% of
  points are >10 px off and 194 image pairs (12%) have at least one such point. 309 of the 341
  large errors are real position errors (nothing in PGDP5K within 10 px), 25 are name mismatches.
  Geometric labels computed from Geometry3K coordinates will inherit these errors.
- 261 Geometry3K diagrams reference points with no position (relations using them are dropped and
  the diagram is flagged `missing_points`); this removes 79 of 193 explicit angle equalities.
- **Geometry3K is split by problem, not figure**: 3002 problems, 1851 distinct figures. 302 of 601
  test images (50%) have an exact duplicate in train; 327 (54%) share a figure cluster with
  train or val. Only 274 test images (258 figures) are free of train/val copies.
- **Cross-source overlap**: 1809 PGDP5K images are Geometry3K images (paper: 1813); 273 PGDP5K test
  images duplicate a Geometry3K train image. PGPS9K contains 515 Geometry3K test images and 558
  PGDP5K test images, so Part B self-training on PGPS9K would train on test figures unless
  filtered. 660 PGDP5K test figures have no Geometry3K copy. Tables: `reports/tables/source_overlap.csv`,
  clusters: `data/processed/figure_clusters.csv`.
- PGDP-format parser validated against the PGDP5K authors' derived logic forms: parallel 520/520,
  equal-length pairs 2863/2863, perpendicular 2452/2467, equal angles 1420 common.
- Geometry3K under-annotates marks: equal_angle 120 relations vs 1493 in PGDP5K, which covers the
  same figures plus more. `MeasureOf = MeasureOf` forms are mostly angle naming (2084 of 2331).
- **tangent**: 1 asserted positive across all three sources (PGDP5K/PGPS9K have no tangent
  annotation at all). Tier 1 tangent will rest on verifier labels and synthetic data.
- PGDP5K readme.txt swaps the val/test sizes (actual: val 500, test 1000).

Open (needs a decision before Phase 5.7 splits; §0.2 #9 unchanged for now)
- Leak-free splits by figure cluster, and which coordinates to use for Geometry3K figures that
  PGDP5K re-annotated (1807 of 1851 figures).

## 2026-10-10 — Leak-free figure-level splits (plan §0.2 #9 revised)

Done
- Plan §0.2 #9 edited first (old text kept in the row), then implemented; tasks 5.7 and 7.11 updated.
- `python -m data.splits` (`data/splits.py`): one diagram per figure (duplicate cluster), seeded
  (split_seed 0) 70/10/20 re-split of Geometry3K figures stratified by has-circle: train 1296,
  val 185, test 370 (was 2101/300/601 problems). 1807 figures use PGDP5K's pixel annotation, 44 keep
  Geometry3K's. Held-out source: 660 PGDP5K test figures with no Geometry3K copy. Part B pool: 3253
  PGPS9K figures in no val/test set (1216 are training figures, flagged `in_train`).
  Outputs: `data/processed/{geometry3k_figures,pgdp5k_heldout,pgps9k_pool}.jsonl`,
  `data/processed/splits/*.txt`, `figures.csv`. Disjointness enforced in code and tests: evaluation
  sets share no figure with anything; train <-> pool is the only allowed overlap.
- Config: `data.split_seed`, `data.split_fractions`; test set `pgdp5k_test` renamed `pgdp5k_heldout`;
  unlabelled source `pgps9k_pool`.
- Census now includes the curated sets and applies the Tier 2 rule to the new training split.
- Tests: 84 passed in 4.9 s.

Found
- Tier 2 re-checked on the new training split: collinear 2137 (include); midpoint 83, equal_angle 87,
  perp_bisector 8 (exclude). Decision unchanged. My earlier expectation that PGDP5K annotations would
  lift midpoint/equal_angle over 200 was wrong: that came from all of PGDP5K train, which includes
  ~2000 non-Geometry3K textbook figures.
- Collinear drops from 4528 to 2137 because PGDP5K only annotates the longest segment of a line;
  equal_length rises 219 -> 315 and parallel 135 -> 161 (marks are annotated more completely).
- tangent: 0 asserted positives in the training split.
