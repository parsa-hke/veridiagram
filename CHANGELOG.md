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

## 2026-10-10 — Phase 3: verifier (3.1–3.4, 3.6–3.8 done; 3.5 awaiting hand labels)

Done
- 3.1 `verifier/primitives.py` (float64 vector helpers). 3.2 `verifier/checks.py`: one check per
  relation returning (is_true, margin), plus `evaluate()` on schema references. 3.3
  `tests/test_checks.py`: exact positive, exact negative and just-inside/just-outside boundary
  cases for every check.
- `data/queries.py`: candidate enumeration (the enumeration half of 5.1). Parallel, perpendicular,
  tangent and collinear are line properties, so asserted relations are matched through carrier
  lines (an asserted pair of lines makes every pair of their sub-segments asserted).
- 3.4 `python -m verifier.tolerance` on the training split (1296 figures). Rule: valley of the
  smoothed log-margin density between the asserted mode and the non-asserted bulk (centre of the
  gap when the valley is flat); knee of the asserted CDF when there is no valley. Result, written to
  `configs/base.yaml` and plan §0.2 #5: parallel 2.56°, perpendicular 2.03°, equal_length 2.32%
  (knee), on_circle 7.94%, tangent 7.94% (= on_circle; no asserted tangents), collinear 1.19% of
  diagram scale. Within tolerance: 94% / 98% / 81% / 99.9% / 99.9% of asserted relations.
  Figures `reports/figures/tolerance_*.png`, table `reports/tables/tolerance_calibration.csv`.
- 3.6/3.7 `python -m verifier.label`: geometric labels with margins for every candidate query of all
  six sets (2.8M query-relations in 32 s) into `Diagram.geometric` and
  `data/processed/queries/<set>.csv.gz`; not-to-scale flags (>10° / >15%), `reports/tables/not_to_scale.csv`.
- 3.8 `python -m verifier.audit` -> `reports/tables/audit_<source>.csv` and example figures
  `reports/figures/audit_<source>_{asserted_not_geometric,geometric_not_asserted}.png`.
- 3.5 tooling: `python -m verifier.gold sample` draws 100 queries (relation x source stratified,
  half verifier-positive, ~2/3 near the tolerance) and builds a blind labelling page
  (`verifier/gold/label.html`, gitignored); `python -m verifier.gold score` computes FP/FN with
  Wilson intervals, re-running the checks with the current tolerances.
- Geometry3K loader: collinear groups that share two points are merged (one line stated with
  different endpoints had produced duplicate groups: 7860 -> 4058 lines).
- Tests: 103 passed.

Found
- Equal length has no density valley: unmarked equal lengths are common in textbook figures, so
  the non-asserted density rises smoothly from 1e-3. The first valley pass returned 0.94% (an
  artefact beside the asserted median); replaced by the knee rule (2.32%).
- on_circle has an empty gap from ~4% to ~12%; the valley centre (7.94%) is loose relative to
  asserted precision (99% within 3.2%). Kept as the plan's rule; the gold set will test it
  (tangent borrows it).
- **Not-to-scale is mostly a Geometry3K coordinate artefact**: 3.6% of Geometry3K diagrams are
  flagged with its own coordinates (36 points "off" their circle), 0.6% of the same figures with
  PGDP5K coordinates; PGDP5K 0.6%, PGPS9K 0.3%. Equal-length marks dominate the remaining flags.
- **Audit (Result 1), curated Geometry3K figures, per carrier pair**: asserted relations confirmed
  by the drawing: perpendicular 98.4%, parallel 93.4%, equal_length 77.3%, on_circle 99.9%,
  collinear 99.8%. True-but-unmarked share of geometric positives: parallel 89%, perpendicular 54%,
  equal_length 95%, tangent 100% (297 tangencies, none annotated). Geometry3K with its own
  coordinates confirms less (parallel 86%, perpendicular 88%, equal_length 61%).
- Unmarked geometric positives inspected by eye are real (rectangle sides, apothems, an inscribed
  circle's tangent, symmetric lengths). Collinear also fires for points in separate sub-figures
  that share a horizontal line (true on the infinite line, as defined).

## 2026-10-10 — Gold set round 1: gate failed; verifier fixed, tolerances revised

Done
- Round 1 labelled by hand (100 items: 57 yes, 39 no, 4 unsure). `python -m verifier.gold score`:
  FP 7.7% (3/39), FN 21% (12/57) -> **gate failed** (ceiling 3%).
- Diagnosed all 15 disagreements by eye (renders of the queried primitives):
  - verifier bugs: a short chord passed as tangent (|d-r|/r = 1 - cos(θ/2) is small for short
    chords); a tangent at a segment endpoint failed because the foot fell just past the end.
    Fixed in `verifier/checks.py` (chord exclusion: both endpoints within tol*r of the circle;
    foot slack = max(2% of length, tol*r)); tests added.
  - tolerance vs the eye: humans accepted perpendicular/parallel up to ~2.9° (first "no" at 4.7°)
    and equal lengths up to 9.1% (first "no" at 33%, unsure at 3.2% and 5.6%).
  - likely label slips (radius highlighted but tangent lines visible; point on the other circle;
    two horizontal bases called non-parallel) and one ambiguous question (tangent wording allowed
    "segment ends on the circle"). Not changed by me; sent to a blind re-check instead.
- **Plan §0.2 #5 edited first**, then tolerances revised into the range consistent with every
  round-1 label: parallel = perpendicular = 3.7°, equal_length 10%; on_circle 7.94%, tangent
  7.94%, collinear 1.19% unchanged. `verifier.tolerance` now writes the config only with
  `--write-config`. Labels and audit regenerated.
- Revised verifier on round 1 (in-sample, not a gate): FP 5.1% (2/39), FN 8.8% (5/57).
- Round 2 prepared (`python -m verifier.gold sample --round 2`): 60 new queries (10 per relation,
  1/3 near tolerance, each with a sampling weight) + blind re-check of the 7 round-1 items the
  revised verifier still disagrees with, mixed with 7 random agreements; 74 items, neutral ids.
  **Pre-registered gate** (plan §3.5): population-weighted FP and FN on the 60 new items, both <= 3%.
- Tests: 104 passed.

Found
- The histogram valleys are stricter than human perception (angles ~2° vs ~3-4° by eye; lengths
  2.3% vs ~10%). The eye also says "yes" to 6.5-7.2° perpendiculars inconsistently (7.17° "no").
- Audit after revision (curated Geometry3K figures): asserted perpendicular confirmed 99.6%,
  parallel 96.4%, equal_length 93.5%; tangencies true-but-unmarked: 315.

## 2026-10-10 — Gold round 2: gate 3.5 passed (adjudicated); Phase 3 complete

Done
- Round 2 labelled (74 items: 60 new held-out queries + 14 blind re-checks). Scored with
  `python -m verifier.gold score` -> `reports/tables/verifier_audit.csv`, `verifier_recheck.csv`.
- Two more rule fixes found by the gold rounds (applied to every query; no label edited):
  - nested segments on one line (AM inside AB) are no longer queries: highlighted together they
    look like one segment (h033 was such a pair; 1 item per round is now "invalid");
  - coincident lines are not parallel; "one line" uses the collinear tolerance
    (collinear_rel x diagram scale).
- Adjudication: the labeller re-inspected the two items that drove the weighted FN (h025: sides
  113 vs 148 px labelled equal; h039: a radius labelled tangent) and the radius item from round 1
  (g020/h068), and confirmed all as slips. Recorded in `verifier/gold/adjudication.csv`; raw label
  files untouched; raw and adjudicated rates both reported.
- Plan 3.5 ticked with both results. Tests: 105 passed.

Found
- **Gate 3.5, round 2 (pre-registered population-weighted rates)**: raw FP 0.1%, FN 43.9% -> FAIL;
  adjudicated FP 0.0%, FN 0.0% -> PASS. Unweighted stress rates (adjudicated): FP 3.8% (1/26,
  a parallel at 3.68° vs the 3.7° tolerance), FN 0% (0/33).
- Importance weighting with 60 items is fragile: one error in a large stratum (44k candidate pairs)
  moved the weighted FN from 0% to 44%. Worth stating next to the gate in the report.
- Blind re-check: of 7 round-1 disagreements the labeller changed 2 (both toward the verifier); of
  7 agreements, 0. Persistent disagreements: two perpendiculars at 6.5° and 7.2° called "yes" (the
  labeller also called 4.7° and 7.2° "no": perception noise), two bases 4 px apart on what looks
  like one baseline called "not parallel", and a rectangle side ending on a circle called tangent.
- Final audit (curated Geometry3K figures): asserted relations confirmed by the drawing:
  perpendicular 99.6%, parallel 96.4%, equal_length 93.5%, on_circle 99.9%, collinear 99.8%.
  Unmarked share of geometric positives: parallel 89%, perpendicular 56%, equal_length 96.5%,
  tangent 100% (315).

## 2026-10-11 — Phase 4: synthetic diagram generator

Done
- 4.1 `data/synthetic/generator.py`: 0-2 circles, then construction steps; with probability
  `inject_rate` (0.5) an exact construction (parallel; perpendicular at an endpoint or a foot;
  equal length from a shared endpoint or elsewhere; point on a segment or its midpoint; point on a
  circle; tangent at a circle point, sometimes with the tangency point and radius drawn), otherwise
  a random point (min separation 30 px) or a random segment that passes >= 8 px from other points
  and preferably starts at an isolated point. Weights in `synthetic.construction_weights` (tangent
  x2: real data has ~no tangent positives). Then a slight rotation (<= 5°) and N(0, 1 px) jitter on
  every coordinate; exact coordinates kept in `meta.exact_points`. Pure function of
  (seed, pool, index).
- 4.2 `data/synthetic/render.py`: black on white, letter labels placed in the widest free direction,
  right-angle squares / tick marks (1-3 per class) / parallel arrows (1-2 per class) for the
  injected relations; sampled line width, font family and slant, font size, label offset, dots,
  mark size, scan noise (p 0.5) and JPEG artefacts (p 0.5). 28 ms per diagram.
- 4.3 `python -m data.synthetic.build` (8 workers, ~4 min): synth_train 50,000 (1k / 10k / 50k are
  nested prefixes, `data/processed/splits/synth_train_{1k,10k,50k}.txt`), synth_extrap 1,000
  (12-20 points, 8-16 segments), synth_unlabelled 2,000 (public file has images only; annotation
  sealed in `data/processed/sealed/synth_unlabelled.jsonl`, sha256
  c093038836421a0dc0c0b156d4b85c26706bc3195f09192e8679cc214567cdd7). Images under
  `data/raw/synthetic/` (gitignored, 2.1 GB).
- 4.4 `python -m verifier.label --sets synth_train synth_extrap` (5.0M query-relations, 57 s) and
  `python -m data.synthetic.check` -> `reports/tables/synthetic_verifier_check.csv`,
  `reports/figures/synthetic_examples.png`.
- Tests: `tests/test_synthetic.py` (counts, separation, determinism, exact constructions, injected
  relations reachable as queries, render, seal). 113 passed.

Found
- Constructions are exact: before jitter every injected relation has margin <= 4e-15.
- After jitter the verifier confirms 99.4-100% of injected relations (train and extrap, every
  relation); random candidate queries are geometric positives 1.3-4.3% of the time, except
  equal_length 11% (the 10% tolerance makes chance near-equal pairs common).
- Synthetic train has 24,792 geometric tangent positives (real training split: 0 asserted).
- First build was 7.2 GB: scan-noise images do not compress in PNG (388 KB vs 19 KB clean, 91% of
  the bytes). Noisy images are now stored as JPEG quality 95: 2.1 GB total.
- Remaining look: about 1 in 3 training diagrams has no injected relation (random only), and
  extrapolation diagrams have some unconnected points (20 points, <= 16 segments). Left as is.

## 2026-10-11 — Phase 5: queries, overlays, dataset (gate 5.8 passed)

Done
- 5.1/5.2/5.4 `data/queries.py` index (`python -m data.queries`): per query a stable id, multi-label
  `geo` (verifier) and `asr` (annotation) vectors over [parallel, perpendicular, equal_length,
  on_circle, tangent, collinear] (none = all zeros), an applicability mask (e.g. on_circle is masked
  for a segment pair, not a negative) and margins. Cap 60 per diagram: training keeps all positives,
  evaluation samples uniformly (natural ratio in expectation). Training only: all-negative queries
  <= 3x positive queries per query kind. Units: train (+ first `synthetic_train_size` synth_train
  diagrams), val, geometry3k_test, pgdp5k_heldout, synth_extrap.
- 5.3 `data/overlay.py`: core line + semi-transparent halo for segments and circles, filled marker
  for points; collinear queries highlight line AB and point P. Returns the touched-pixel mask; unit
  test confirms nothing else changes.
- 5.5 `data/augment.py`: rotation +-180° (canvas expanded), flips, isotropic scale (ink-box crop,
  then s x 448 capped at 448), line thickening (3x3 min filter), contrast, Gaussian noise, JPEG.
  No shear or perspective. Test: a highlighted straight line stays straight.
- 5.6 `data/dataset.py`: flatten -> resize long side 448 -> overlay -> (train) augment -> pad to a
  multiple of 16 -> normalise (ImageNet mean/std, to be confirmed against the DINOv3 processor in
  6.1); overlays cached as PNG keyed by (settings hash, set, diagram, query id); collate pads a batch
  with normalised white. Returns (image, label_geometric, label_asserted, meta with mask).
- 5.7 units disjoint by diagram id (`check_disjoint`, tests).
- 5.8 **Gate passed**: `python -m data.dataset --config configs/smoke.yaml` builds every unit on 50
  diagrams (train: 50 real + 50 synthetic) and loads one batch per unit: images (16, 3, H, W) with
  H, W <= 448 and multiples of 16, labels and masks (16, 6). Example inputs:
  `reports/figures/dataset_examples.png` (`--examples`).
- Full indexes (7 s): train 37,846 queries from 1,296 diagrams; val 6,227; geometry3k_test 11,319;
  pgdp5k_heldout 25,272; synth_extrap 60,000; train + 10k synthetic 221,519.
- Tests: 120 passed.

Found
- **Highlight colour changed from red to green (0, 200, 0)**: in 600 sampled real diagrams red,
  orange and cyan pixels occur in 13.8%, 15.0% and 12.7% of images (right-angle marks, ticks,
  labels are often pink/red), saturated green in 0%. A red highlight would be confusable with
  drawn marks.
- The overlay is drawn after resizing to 448 (plan order: overlay, augment, resize). Drawing first
  would make the 3 px highlight thinner or thicker depending on the source resolution (images
  range ~150-1600 px); augmentation still happens after the overlay.
- Ink-box detection for scale augmentation must ignore scan noise: threshold 200, not 245 (the
  noisy synthetic background counted as ink).
- Training tangent positives: 476 from real data alone, 5,360 with 10k synthetic diagrams (11x).
- synth_extrap gives 60k evaluation queries (1,000 diagrams x 60); evaluating ViT-L on all of them
  is costly, so Phase 7 may subsample it.

## 2026-10-11 — Phase 6: model code (6.1-6.5 done; gate 6.6 to be timed on Colab)

Done
- 6.1 `models/backbone.py`: DINOv3 ViT-L/16 via transformers 5.19 (`DINOv3ViTModel`; tokens = CLS + 4
  registers + patches; RoPE, any multiple of 16), fallback DINOv2-L/14; outputs CLS, patch tokens
  (registers dropped) and pooled = CLS ++ mean patch; gradient checkpointing on; `layer_depth` for
  LLRD (embeddings 0, block i -> i+1, final norm on top); the checkpoint's image-processor mean/std
  are checked against `input.mean/std`; which backbone loaded (and why a fallback) goes to metrics.
- 6.2 `models/head.py`: LayerNorm -> Linear (sigmoid at inference); masked BCE over applicable
  relations; positive weights = negatives/positives per relation in the training split (cap 50).
- 6.3 `models/probe.py`: frozen features cached at <cache>/features/<backbone>/<overlay hash>/<index>.npz
  (verified by query ids), head trained with AdamW, early stopping on val macro-F1.
- 6.4 `models/finetune.py`: AdamW, head LR 1e-3, backbone 2e-5 x 0.8^k from the last block down,
  wd 0.05 (none on biases/norms/tokens), cosine with linear warm-up per step, grad accumulation,
  clipping 1.0, bf16 autocast, EMA 0.999 with warm-up, best.pt = EMA weights by val macro-F1,
  last.pt for resume (tested), history.jsonl with loss, val per-class F1, end and peak LRs.
- 6.5 seeds from config drive torch/numpy/random, data order, augmentation and head init.
- `models/common.py` (autocast, EMA, schedule, prediction, evaluation of every unit against both
  label sets, predictions saved to <run_dir>/predictions/<unit>.npz); `analysis/metrics.py`
  (per-relation P/R/F1, macro-F1 over relations with positives; ECE/bootstrap in Phase 8).
- `data/download.py` (download + sha256 check + safe extraction) and `data/prepare.py` (one command:
  download -> build -> overlap -> splits -> census -> synthetic -> labels -> query indexes).
- `GEO_OVERLAYS` environment variable: overlays applied by `load_config`, so every CLI follows the
  Colab paths. `configs/colab.yaml` now keeps data and caches on the VM's local disk (Drive is too
  slow for 50k small images); the launcher unpacks one data tarball from Drive per session.
- Colab launcher rebuilt: data cell (prepare once, tarball on Drive keyed by the data-code git hash)
  and the Phase 6 smoke gate.
- Smoke config: `data.max_queries_per_unit: 256`, `finetune.save_last: false`.
- Tests: tiny random DINOv3/DINOv2 (no download) for shapes, LLRD, loss masking, metrics, EMA,
  schedule, probe and fine-tune end to end with resume. 134 passed.

Found
- DINOv3 is gated; this Mac has no Hugging Face login (401), so local runs used the DINOv2-L
  fallback (recorded in metrics). DINOv2-L's processor mean/std match the config.
- Local smoke on the M5 (MPS), DINOv2-L fallback: probe 157 s; fine-tune 724 s (16 optimiser steps
  in 498 s, ~25-30 s each; evaluation ~55 s per 256 queries). Over the 10-minute gate, so the gate
  is timed on a Colab GPU (plan §0.6 updated); the Mac keeps the probe smoke and the tiny tests.
- A Phase 1 test used "fine-tune smoke raises NotImplementedError" as its failure case; with a real
  trainer it started a 9.5-minute training run. Replaced by an injected failure.
- The feature-cache log line used a path relative to the repo, which crashes when the cache is
  outside it (Colab /content/cache). Fixed.
- Added `python -m models.backbone`: loads the primary backbone (DINOv3) with no fallback, runs a
  forward pass and prints architecture (class, hidden size, layers, patch, register tokens,
  parameters), token counts and normalisation; exits non-zero if the gated checkpoint is not
  accessible. The Colab launcher runs it before the smoke runs, so the gate cannot silently pass on
  the DINOv2 fallback. Locally it fails as expected (no Hugging Face login, 401).

## 2026-10-11 — First Colab run (L4): data ready, fine-tune smoke failed to load weights

Found (from the exported notebook)
- Setup, data preparation (`data.prepare --download`: 3.2 GB raw, 295 MB processed, all indexes;
  labelling took 359 s on Colab) and the probe smoke (ok, 106 s) worked on an L4.
- The fine-tune smoke and the three full runs failed with "no backbone could be loaded": DINOv3
  raised a SafetensorError, not a 401, so the token and gated access worked; reading the weights
  failed. Cause: HF_HOME pointed at Google Drive; safetensors files are memory-mapped, which the
  Drive mount does not support reliably (the DINOv2 fallback lives in the same cache).
- "Run all" also started the full runs before the gate: a launcher flaw.

Done
- Launcher: HF_HOME on the VM's local disk (re-download per session, < 1 min); DINOv3 access check
  in its own cell; full-run cell guarded by `RUN_FULL = False`.
- `python -m experiments.gate smoke` (`experiments/gate.py`): checks both smoke runs for status ok,
  primary backbone (DINOv3, not the fallback), < 10 min, predictions for every unit and best.pt;
  prints PASS/FAIL. Tests: 137 passed.

## 2026-10-11 — Gate 6.6 passed on Colab (L4); Phase 6 complete

Done
- Second Colab session at `201cfcf`: data unpacked from the Drive tarball (data revision 4fa8896,
  3.2 GB raw, 295 MB processed); `python -m models.backbone` loaded the gated checkpoint with no
  fallback: DINOv3ViTModel, hidden 1024, 24 layers, patch 16, 4 register tokens, 303.1M
  parameters, 560 patch tokens for a 448x320 input as expected, image-processor mean/std equal to
  the config (ImageNet). "facebook/dinov3-vitl16-pretrain-lvd1689m: OK".
- `python -m experiments.gate smoke`: PASS. Probe smoke ok, DINOv3, 97 s; fine-tune smoke ok,
  DINOv3, 97 s; predictions for every evaluation unit; best.pt written to Drive.

Found
- With HF_HOME on the local disk the safetensors load works (the first session's SafetensorError
  came from the Drive-mounted cache).
- Both smoke runs took 97 s on the L4. Checked the gate is not reading one file twice: run locally
  it reports the Mac's own runs (157 s and 724 s) and fails them for using the DINOv2 fallback.
- Smoke-scale numbers (256 queries per unit, 16 optimiser steps) are not results.

## 2026-10-12 — Lab readiness for Phase 7 (UOWD AI Lab GPUs)

Done
- Plan §0.6 revised first: Phase 7 runs on the UOWD AI Lab GPUs (Colab Pro's ~167 compute units,
  about 14 A100-hours, are far below the plan's budget); Colab stays as a backup.
- `LAB_GUIDE.md`: step-by-step runbook (home preparation, setup, data, access check, smoke gate,
  benchmark, queue, monitoring, packing results, token clean-up, troubleshooting).
- `scripts/lab_setup.sh`: uv + Python 3.11 + pins in .venv without admin rights; picks a PyTorch
  CUDA build the driver supports (the default torch 2.14 wheel is CUDA 13.0, driver >= 580; older
  drivers get cu128 / cu126); checks GPUs and bf16; runs the tests.
- `python -m experiments.queue <queue.yaml>`: one run per GPU (CUDA_VISIBLE_DEVICES), file order,
  skips runs that are "ok" in their own log.csv, `serial` groups (probe seeds share cached
  features), per-run logs in runs/queue_logs/, `--status`, `--dry-run`, `--only`, `--set`. Verified
  with a real two-job queue (DINOv2-S, smoke data): both ran, a restart skipped both.
- `configs/queue_phase7.yaml`: 3 probes, 3 fine-tunes (real), 3 fine-tunes (+10k synthetic),
  ablations DINOv2 / 224 px / no augmentation (1 seed), Part A 1k (1 seed) and 50k (3 seeds).
  Not yet: 7.8 (no-overlay, needs a query-token model) and Part B (needs the 7.3 teacher).
- `python -m experiments.benchmark`: measured train/eval images per second and peak memory at the
  queue's resolutions, per-job hours, total GPU-hours, wall time on N GPUs, and wall time for
  30/20/15/10 fine-tune epochs.
- New configs: `finetune_synth10k.yaml` (7.3), `ablation_{dinov2,res224,noaug}.yaml`, `lab.yaml`.
  `experiments/scaling.py` (Part A: a fine-tune with the synthetic prefix, size checked) and
  `experiments/evaluate.py` (re-evaluate a best.pt on every unit) implemented.
- Parallel-safety: overlay PNGs, feature caches and query indexes are written atomically
  (temporary file + rename), since several runs share them.
- Disk: best.pt saved in bfloat16 (~0.6 GB for ViT-L); last.pt (~5 GB) deleted when a run finishes
  (`finetune.keep_last` to keep it).
- float16 training (GPUs without bf16) now uses a gradient scaler.
- Tests: 147 passed.

## 2026-10-12 — Windows support for the lab machines

Done
- The UOWD AI Lab machines run Windows: `scripts/lab_setup.ps1` (uv, Python 3.11, PyTorch from the
  CUDA index matching the driver, since PyTorch's Windows wheels on PyPI are CPU-only; GPU check;
  tests) and `scripts/cuda_check.py` (shared with the Linux script). LAB_GUIDE.md: Windows section.
- DataLoader collate is now a picklable class (`data.dataset.Collate`): Windows starts workers with
  `spawn`, which cannot pickle the previous closure. Verified on the Mac by forcing spawn.
- Cache writes tolerate Windows file locks (`replace_or_discard`); the queue stops children with
  terminate() on Windows (SIGINT cannot be sent there); lab overlay turns the overlay PNG cache off
  (over a million files for the 50k pool).
- Tests: 147 passed. The PowerShell script could not be executed here (no PowerShell on the Mac);
  reviewed by hand.

## 2026-10-12 — Lab benchmark and the compute-bounded Phase 7 design

Found
- Lab benchmark (4x RTX 5000 Ada 32 GB, Windows, DINOv3, gradient checkpointing on): 448 px train
  22.5 img/s, eval 98.5 img/s, peak 5.5 GB; 224 px train 64.9, eval 491.5 img/s. The plan as
  written: 1,420 GPU-hours, ~19 days on 4 GPUs (94% from 30 epochs over the 10k/50k synthetic sets).
- The lab PC is a shared classroom machine: 2-3 h per visit (~10 GPU-hours), may be switched off;
  a second visit is uncertain.

Done (plan §0.2 #10, #11 and task 6.4 edited first)
- LP-FT (Kumar et al., ICLR 2022): `finetune.lp_queries` (20,000) frozen-feature head fit, then full
  fine-tuning for 2 epochs x 40,000 images (`finetune.epoch_images`, drawn with replacement);
  synthetic runs draw 50% real / 50% synthetic (`finetune.real_fraction`), so every fine-tune has the
  same budget and Part A compares synthetic diversity at equal compute and real-data exposure.
- Synthetic extrapolation evaluated on 10,000 of its 60,000 queries (`eval.max_queries`).
- Lab overlay: gradient checkpointing off (5.5 GB of 32 GB used; it costs ~30% speed).
- Queues: `configs/queue_phase7.yaml` (today: probes, 7.2 x3, 7.3 x3, 224 px and no-aug ablations, in
  priority order, ~2-2.5 h on 4 GPUs) and `configs/queue_phase7_part2.yaml` (DINOv2 ablation, Part A
  1k and 50k; for Colab or a second visit). 7.8 and Part B follow in a later update.
- Benchmark estimates include the LP stage, epoch images and evaluation caps. Tests: 148 passed.

## 2026-10-12 — First Phase 7 results (lab) and Colab for the second batch

Found (lab, in progress)
- Probes (3 seeds), geometric macro-F1: val 0.597-0.601, geometry3k_test 0.585-0.609,
  pgdp5k_heldout 0.515-0.520, synth_extrap 0.17-0.20 (probe_s0 20.4 min incl. features; s1/s2 ~0.5 min
  from the cache).
- finetune_s0 (LP-FT, 2 x 40k images): val 0.869, geometry3k_test 0.858, pgdp5k_heldout 0.861
  (asserted 0.535-0.624). A fine-tune takes ~74 min on one RTX 5000 Ada (~28 min per 40k-image epoch,
  ~10 min evaluation); GPU utilisation 62-100% with four runs sharing the CPU for data loading.

Done
- `data/VERSION` names the Colab data tarball (bump only when pipeline outputs change); the launcher
  renames the first tarball (data_4fa8896.tar, identical outputs) instead of rebuilding.
- `configs/colab.yaml`: no overlay cache, no gradient checkpointing (A100 40 GB).
- Launcher run cell takes a queue (`QUEUE = "configs/queue_phase7_part2.yaml"`): DINOv2 ablation,
  Part A 1k and 50k, in parallel with the lab batch.

## 2026-10-13 — Phase 7: lab batch complete; 7.8, Part B and the wrap-up built; the rest goes to Colab

Found (lab batch, 11 runs, all ok on DINOv3; archives verified: 13 metrics, 52 prediction files with
exact query counts, three 7.3 checkpoints that load with weights_only and match their metrics)
- Geometric macro-F1, mean ± sd over 3 seeds (geometry3k_test / pgdp5k_heldout / synth_extrap):
  probe 0.595 ± 0.013 / 0.517 ± 0.003 / 0.185 ± 0.013; fine-tune 0.867 ± 0.007 / 0.869 ± 0.008 /
  0.631 ± 0.017; fine-tune + 10k synthetic 0.871 ± 0.010 / 0.885 ± 0.004 / 0.713 ± 0.009.
  Synthetic data helps out of distribution (held-out source +0.016, extrapolation +0.082, seed ranges
  disjoint) and not in distribution. 224 px (seed 0): 0.855 / 0.861 / 0.609; no augmentation: 0.868 /
  0.870 / 0.632. Asserted labels: 0.54-0.56 for every fine-tune (most true relations are unmarked).
- Not-to-scale subset (7.4): 7 test diagrams, 353 queries; reported per run, too small for claims.
- Part B teacher = finetune_synth10k_s2 (val 0.8824). On a 150-diagram sample of the pool (local,
  ViT-L rebuilt from the checkpoint; it reproduces the lab's saved val predictions, 0 decisions
  differ): at τ 0.9 the confident entries' pseudo-positive precision is 0.89 (accuracy 0.979). The
  plan's contradiction alone (parallel and perpendicular) flags 1/150 diagrams; with confident
  negatives as facts (C2) 13/150, and the entries it removes have precision 0.74 vs 0.92 kept. C2
  is therefore on. It also fires on 10.4% of the pool's verifier labels: tolerance relations are not
  transitive (equal_length chains within 10% drift past 10%).

Done (plan §0.2 #9 and #11 and §0.6 edited first: Part B pool = pool figures outside the training
split; everything remaining on Colab)
- 7.10 `verifier/consistency.py`: closure (parallel classes of carrier lines, perpendicular between
  classes, two perpendiculars to one line are parallel, equal_length classes); contradictions C1/C2.
- 7.11 `experiments/selftrain.py`: teacher predictions on the `pgps9k_pool` unit (2,037 diagrams,
  66,019 queries), cached and shared; confidence / consistency / none (control) filters; student from
  the teacher's weights, 2 x 40k images, 50% Geometry3K / 50% pseudo-labelled
  (`data.extra_train_index`); post-hoc precision of both filters in metrics.json.
- 7.8 `configs/ablation_nooverlay.yaml`: no highlight; the query as kind + point-name tokens
  (`data/query_tokens.py`) cross-attending over the patch tokens (`models.head.QueryAttention`).
- `python -m analysis.phase7`: phase7_runs/summary tables, not_to_scale_eval.csv,
  selftrain_pseudo_labels.csv, reports/figures/scaling.png. `python -m experiments.gate phase7`.
- `configs/queue_phase7_colab.yaml` (13 runs, seed 0 of everything first; replaces
  queue_phase7_part2.yaml), launcher cells for the teacher (unpacked from the lab archive on Drive,
  sha256-checked), the Phase 7 smoke gate and packing the results; `COLAB_GUIDE.md`.
- Lab results imported into the local runs/ (gitignored) and runs/log.csv. Tests: 164 passed.

## 2026-10-13 — Phase 7 complete (Colab batch: 13 runs, all ok)

Found (geometric macro-F1, geometry3k_test / pgdp5k_heldout / synth_extrap; `reports/tables/phase7_summary.md`)
- Colab A100: ~38-45 min per run (2x the lab's RTX 5000 Ada), 13 runs in ~8.5 h. The archive was
  checked before import: status ok, two epochs, the expected backbone and exact prediction counts.
- 7.9 Part A: 1k synthetic 0.876 / 0.885 / 0.718; 10k 0.871 / 0.885 / 0.713; 50k 0.872 ± 0.004 /
  0.882 ± 0.005 / 0.707 ± 0.008 (0: 0.867 / 0.869 / 0.631). The held-out-source and extrapolation
  gains are reached by 1k diagrams and flat after, at a fixed 80k-image budget with 50% synthetic.
- 7.5 DINOv2-L (seed 0) 0.913 / 0.920 / 0.754, ~0.05 above DINOv3-L at the same recipe; one seed.
- 7.8 no overlay (query as point-name tokens) 0.431 / 0.429 / 0.137: below the linear probe; the
  highlight is what makes the task learnable.
- 7.11 Part B (teacher finetune_synth10k_s2, 0.882 / 0.889 / 0.716): pool 2,037 diagrams; τ 0.9 keeps
  18,252 pseudo-positives at precision 0.882 (equal_length 0.709, others 0.94-1.00); the consistency
  filter flags 162 diagrams, kept precision 0.898, removed 0.798. Students: confidence 0.886 ± 0.003 /
  0.896 ± 0.003 / 0.737 ± 0.009; consistency 0.895 ± 0.010 / 0.903 ± 0.006 / 0.724 ± 0.010; control
  without pseudo-labels (seed 0) 0.909 / 0.919 / 0.741. The gain over the teacher is the extra
  training; the pseudo-labels cost ~0.015-0.02 against the control (one control seed).
- Not-to-scale subset (7 diagrams): no different from the full sets within its noise.

Done
- Colab runs imported into runs/ (finetune_s0-2 on Drive are failed Phase 6 attempts from
  2026-10-10, not imported); `python -m analysis.phase7` regenerated every Phase 7 table and
  reports/figures/scaling.png; plan 7.4, 7.5, 7.8, 7.9, 7.11, 7.12 ticked. COLAB_GUIDE.md corrected
  (run progress is in /content/veridiagram/runs/queue_logs, not in the notebook cell; Colab's own
  package conflict message is harmless; caffeinate without -d lets the screen sleep).

## 2026-10-13 — Website, milestone 1: data export and a plain-text site (WEBSITE_PLAN.md §6.1)

Decisions (WEBSITE_PLAN.md §8, recorded there and in `web/site.yaml`): demo model
`ablation_dinov2_s0`; free CPU Space; run the two extra Part B control seeds (the page labels the
control "one seed" until then); build now, Phase 8 later; name "Parsa Hosseini", repository private.

Done
- `web/export.py` (`python -m web.export`, 2.5 s): runs/ + reports/tables/ + configs/ + split
  manifests -> `web/site/public/data/{results,scaling,pseudo_labels,facts}.json`. results: every
  experiment x unit x label set (seeds, values, mean, sd, per-relation F1 mean/sd) and the
  not-to-scale subset; scaling: the 7.9 points; pseudo_labels: the 7.11 precision table; facts: site
  metadata, dataset sizes, tolerances and calibration, overlay, recipe, compute, backbone
  architecture (from the Hugging Face config on the meta device, no weights), audit, gold-set rates,
  not-to-scale counts, colour check. Each file is stamped with the commit it was exported from.
  Game and gallery exports are deferred to milestone 5 (they need the demo pool and images).
- `analysis/phase7.py`: `scaling_points()` factored out of `scaling_figure()` so the figure and the
  site share one computation (figure unchanged).
- `data/colour_check.py` (`python -m data.colour_check`, 3 s): scripted version of the Phase 5.3
  highlight-colour check -> `reports/tables/highlight_colour_check.csv`.
- `web/site/`: Vite 8 + TypeScript 7, no framework. Renders the frontispiece, Books I-III
  (Propositions I-X) and a colophon as plain text from the JSON. Every number goes through
  `n("doc:path", fmt)`, which wraps it in a span carrying its JSON path; comparative words ("below",
  "beat both", "by less than two standard deviations") are computed from the data too. A trace
  check (`src/audit.ts`) flags any number in visible text that is not traced; in dev mode (or with
  `?trace`) the page shows the count: 234 traced, 0 missing, 0 untraced.
- Tests: `tests/test_web_export.py` (fake runs, tables, manifests; 1.3 s), `tests/test_colour_check.py`,
  `tests/test_web_site.py` (runs the site's vitest suite, 0.5 s: formats, path resolution, and the
  whole page rendered from the real export with no untraced or missing number). `web` and
  `data.colour_check` added to `tests/test_skeleton.py`. 171 passed.

Found
- **The "green in 0 of 600" fact did not reproduce as stated.** The 2026-10-11 check had no script.
  Re-measured on a fixed sample (200 each of Geometry3K, PGDP5K, PGPS9K): green-family ink (hue
  70-165°, saturated) appears in 6/600 images (1.0%: green labels, lines and sector fills in coloured
  textbook figures, e.g. (0, 142, 68)), but no image has 20+ pixels within RGB distance 80 of the
  highlight (0, 200, 0) (3/600 at distance 100). The highlight is still the safest choice (red family
  52.7%, blue 85.3% under the same rule; the old 13.8% red figure used an unrecorded, stricter rule).
  The site says "the highlight's colour", with the rule in a margin note; WEBSITE_PLAN §1.1 and plan
  5.3 annotated.
- The DINOv3 config is gated and not in the local Hugging Face cache, so its architecture facts are
  exported as unavailable (the page says so); `python -m web.export --online` after `hf auth login`
  fills them. DINOv2-L from the cache: 304.4M parameters, 24 layers, 1,024 patch tokens at 448 px.
- 7.11 kept pseudo-positive precision is 0.8975, which formats as 0.897 at three decimals (the
  changelog above rounded it to 0.898).
