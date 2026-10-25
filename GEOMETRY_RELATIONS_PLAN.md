# Geometry Diagram → Formal Relations: End-to-End Build Plan

MAAI 901 Assessment 2 (Applied Vision Transformer Report) · Parsa Hosseini · UOWD · October 2026

This file is the working plan for the project and is written to be executed with Claude Code, one phase per session. Each task has a definition of done. Tasks marked **GATE** must be done and checked before the next phase starts. Decisions recorded in §0.2 are fixed; change them only by editing this file first.

---

## 0. Project definition

### 0.1 Research question

Can a fine-tuned vision transformer read the formal geometric relations present in a textbook diagram, when correctness is judged by a coordinate-based verifier rather than by human labels? And does verifier-generated data extend the model beyond the distribution it was trained on?

### 0.2 Fixed decisions

| # | Decision | Choice |
|---|----------|--------|
| 1 | Primitive detection | Out of scope. Points, segments and circles come from dataset annotations. |
| 2 | Task formulation | Pairwise (n-ary) query: the queried primitives are overlaid on the image in a distinct colour; the ViT classifies the relation. |
| 3 | Relation vocabulary | Tier 1 (must): `parallel`, `perpendicular`, `equal_length`, `on_circle`, `tangent`, `none`. Tier 2 (only if ≥200 positives in the data): `collinear`, `midpoint`, `equal_angle`, `perp_bisector`. **Decided 2026-10-10 (task 2.6, `reports/tables/relation_census.csv`, asserted positives in Geometry3K train counted once per unique figure): include `collinear` (4,528); exclude `midpoint` (50), `equal_angle` (53), `perp_bisector` (5).** **Re-checked 2026-10-10 after the §0.2 #9 revision, on the new training split (1,296 Geometry3K figures, mostly PGDP5K annotations): collinear 2,137 include; midpoint 83, equal_angle 87, perp_bisector 8 exclude. Decision unchanged.** Note: `tangent` has 0 asserted positives in the new training split (1 across all real data); it relies on verifier labels and synthetic data. |
| 4 | Ground truth | Two label sets per query: `asserted` (human annotation) and `geometric` (coordinate verifier within tolerance). Train and headline on `geometric`; report `asserted` alongside. Not-to-scale diagrams are flagged and reported as a category. |
| 5 | Verifier tolerances | Chosen from the empirical distributions in Phase 3, then fixed. **Fixed 2026-10-10 (task 3.4, `reports/tables/tolerance_calibration.csv`, training split): parallel 2.56°, perpendicular 2.03° (density valleys); equal_length 2.32% (no valley: knee of the asserted CDF); on_circle 7.94% (valley, centre of an empty gap); tangent 7.94% (= on_circle, no asserted tangents); collinear 1.19% of diagram scale (valley). Not-to-scale: >10° / >15%. Subject to the gold-set gate (3.5).** **Revised 2026-10-10 after gold round 1 failed the gate (FP 7.7%, FN 21%; `reports/tables/verifier_audit.csv`): the eye accepts more than the histogram valleys. Tolerances moved into the range consistent with every round-1 human label: one angle tolerance 3.7° for parallel and perpendicular (round-1 consistent range 2.92°–4.69°), equal_length 10% (human 'yes' up to 9.1%, first 'no' at 33%); on_circle 7.94%, tangent 7.94%, collinear 1.19% unchanged (already inside their ranges). Tangent check fixed: a segment with both endpoints on the circle is a chord, not a tangent; the foot-of-perpendicular slack scales with tolerance × radius. Because round 1 informed the revision, the gate is decided on a fresh round 2 (new queries + a blind re-check of round-1 disagreements mixed with an equal number of agreements).** |
| 6 | Backbone | DINOv3 ViT-L/16 (verify checkpoint name on the Hugging Face hub; expected `facebook/dinov3-vitl16-pretrain-lvd1689m`). Fallback: `facebook/dinov2-large`. |
| 7 | Resolution | Long side 448, aspect preserved, padded. |
| 8 | Fine-tuning | Linear probe (baseline) and full fine-tune with layer-wise LR decay. |
| 9 | Splits | **Revised 2026-10-10 (Phase 2 findings: Geometry3K is split by problem and 54% of its test images share a figure with train/val; its coordinates are noisy; the sources overlap).** Unit = figure (duplicate-image cluster, `data/processed/figure_clusters.csv`); every split is disjoint by figure. (a) Geometry3K figures are re-split 70/10/20 by figure with a fixed seed, stratified by whether the figure has a circle → train / val / test (ID). (c) Each Geometry3K figure takes its annotation from PGDP5K when PGDP5K re-annotated it (pixel-accurate coordinates, labels from drawn marks), otherwise from Geometry3K. (b) Held-out source test: PGDP5K test figures with no Geometry3K copy. (d) Part B unlabelled pool: PGPS9K figures not in any val or test set. Train = Geometry3K-figure train + synthetic. Plus synthetic extrapolation (1.5–2× points). Was: Geometry3K train/val/test as published, PGDP5K test, PGPS9K. **Revised 2026-10-13 (Part B): self-training uses the 2,037 pool figures that are not training figures (`meta.in_train` false); the other 1,216 are Geometry3K training figures the teacher has already seen with labels, so pseudo-labelling them adds nothing unlabelled.** |
| 10 | Evaluation | Per-relation P/R/F1 + macro-F1 on each test set, 3 seeds, bootstrap CIs, asserted-vs-geometric audit, headline "solved by probe only / fine-tune only / both" table, ECE, attention maps. **Synthetic extrapolation is evaluated on 10,000 of its 60,000 queries (evenly spaced, 2026-10-12, compute).** **Note 2026-10-20 (evaluation padding): the pooled feature averages every patch token, padding included, and Phase 7 evaluation padded each batch of 32 to its largest image, so a saved prediction depended on its batch neighbours (2.2% of decisions change when a query is asked alone; CHANGELOG 2026-10-13, milestone 4 part 2). Every Phase 7 checkpoint that exists is re-evaluated with each query padded to a fixed `long_side` x `long_side` square (`eval.pad: square`, `configs/eval_square.yaml`, runs `sq_<experiment>_s<seed>`; the original runs are kept). The lab did not keep the 7.2, 7.6 and 7.7 checkpoints; 7.2 is retrained on Colab with the same config (`finetune_colab`, 3 seeds) and evaluated both ways. Numbers the report uses (decided before the re-evaluation results): the square-padded evaluation for every experiment that has one (7.2 = `sq_finetune_colab`; the probes keep their trained heads and get square-padded features); 7.6 (224 px) and 7.7 (no augmentation) have no checkpoint and are reported with the original batch padding next to the original 7.2 under the same padding. The original batch-padded numbers of every run and the per-run change go in an appendix table (`reports/tables/padding_effect.csv`).** **Done 2026-10-25 (Colab, 24 re-evaluations + `finetune_colab` x3, all imported; `reports/tables/padding_effect.csv`, `phase7_summary.md`): square padding changes 1.3-2.9% of decisions on the real test sets and 0% on synth_extrap (its images are already square); every fine-tune gains +0.004 to +0.019 geometric macro-F1 on geometry3k_test / pgdp5k_heldout, the probes lose 0.012-0.019 on geometry3k_test (heads trained on batch-padded features). No conclusion changes.** |
| 11 | Extension | Part A: synthetic verifier-labelled data scaling (0 → 1k → 10k → 50k). Part B: self-training on real unlabelled scans with a deductive consistency filter vs a confidence filter. **Revised 2026-10-12 (lab benchmark: ViT-L at 448 px trains at 22.5 img/s per RTX 5000 Ada; the plan as written needed 1,420 GPU-hours, ~19 days on 4 GPUs): the lab PC is a shared classroom machine usable 2-3 h per visit (~10 GPU-hours), so fine-tunes use **LP-FT** (Kumar et al., ICLR 2022: fit the linear head on frozen features first, then fine-tune everything; faster and more robust out of distribution) with the same budget for every run: a linear-probe stage on 20,000 training queries, then 2 epochs x 40,000 images. Runs with synthetic data draw 50% real / 50% synthetic per batch, so Part A compares synthetic diversity at equal compute and equal real-data exposure. 3 seeds where the plan asks for them. Lab today: probes, 7.2, 7.3, ablations 7.5-7.7; Part A, Part B and 7.8 on Colab (or a second lab visit).** **2026-10-13: the lab batch (11 runs) finished; every remaining run (Part A 1k/50k, Part B with a no-pseudo-label control, 7.5, 7.8) runs on Colab, `configs/queue_phase7_colab.yaml`.** |
| 12 | Engineering | PyTorch, Hugging Face `transformers` or `timm`, single YAML config, fixed seeds, every run logged, 10-minute smoke-test mode. |

### 0.3 Deliverables

- Technical report, max 2,500 words (rubric in §9).
- Code repository (GitHub), clean, runnable from one command.
- Visualisations and sample outputs (figures listed in §8.5).

### 0.4 Repository layout

```
geo-relations/
  CLAUDE.md                 # instructions for Claude Code (see §0.5)
  GEOMETRY_RELATIONS_PLAN.md
  README.md
  pyproject.toml / requirements.txt
  configs/
    base.yaml
    smoke.yaml
    probe.yaml
    finetune.yaml
    scaling_{1k,10k,50k}.yaml
    selftrain_{confidence,consistency}.yaml
  data/
    raw/                    # downloaded datasets, untouched (gitignored)
    schema.py               # unified Diagram dataclass + JSON (de)serialisation
    loaders/
      geometry3k.py
      pgdp5k.py
      pgps9k.py
    synthetic/
      generator.py          # random diagrams with injected relations
      render.py             # matplotlib textbook-style renderer
    queries.py              # enumerate n-ary queries per diagram
    overlay.py              # draw queried primitives onto the image
    dataset.py              # torch Dataset over (image, query) → labels
    augment.py              # relation-preserving augmentations
    build.py                # CLI: raw → unified JSONL + cached query index
  verifier/
    primitives.py           # vector ops
    checks.py               # one function per relation
    tolerance.py            # calibration from distributions
    audit.py                # asserted-vs-geometric comparison, not-to-scale flags
    consistency.py          # deductive closure checker (Part B)
    gold/                   # hand-checked gold set (100 queries)
  models/
    backbone.py             # DINOv3 / DINOv2 loading, feature extraction
    head.py                 # classification head
    probe.py                # linear probe trainer
    finetune.py             # full fine-tune with LLRD + EMA
  experiments/
    run.py                  # single entry point: python -m experiments.run --config X
    evaluate.py
    scaling.py
    selftrain.py
  analysis/
    metrics.py              # P/R/F1, macro-F1, ECE, bootstrap CIs, val-tuned thresholds
    phase7.py               # Phase 7 tables, scaling curve, padding effect
    phase8.py               # 8.1 table: CIs, ECE, tuned thresholds, per-relation F1
    headline_table.py
    attention_maps.py
    error_analysis.py
  notebooks/                # figures only, no logic
  tests/
  reports/
    figures/
    tables/
    report.md               # draft of the 2,500-word report
```

### 0.5 CLAUDE.md (put this at the repo root)

```
# Project: Geometry diagram → formal relations (MAAI 901 Assessment 2)
Read GEOMETRY_RELATIONS_PLAN.md before any task. Work one phase at a time.
Rules:
- Never change a decision in §0.2 without editing the plan first and saying so.
- Every new module gets a test in tests/ that runs in under 10 s.
- Every experiment is launched through experiments/run.py with a config in configs/.
- Every run writes results to runs/<name>/metrics.json and a one-line entry to runs/log.csv.
- Smoke mode (configs/smoke.yaml, 50 diagrams, 1 epoch) must pass before any full run.
- Do not invent dataset field names: inspect the raw files and write what is actually there.
- Plots go in reports/figures/, tables in reports/tables/, both regenerated by scripts.
- Keep a running CHANGELOG.md: date, what was done, what was found.
```

### 0.6 Compute (added 2026-10-10; revised 2026-10-12)

**Revised 2026-10-13: the lab ran the first Phase 7 batch (11 runs, 2026-10-12); the remaining 13 runs go to Colab Pro (the units are free for this project after all), step by step in `COLAB_GUIDE.md`.** **Revised 2026-10-12: Phase 7 runs on the UOWD AI Lab GPUs** (Colab Pro has ~167 compute units, about 14 A100-hours, far below the plan's fine-tune budget). Same code and configs; the lab uses `configs/lab.yaml` (repo-local paths), `scripts/lab_setup.sh`, `python -m experiments.benchmark` (measured throughput and per-experiment time) and `python -m experiments.queue configs/queue_phase7.yaml` (one run per GPU, skips finished runs, resumes interrupted ones). Step-by-step: `LAB_GUIDE.md`. Colab stays as a backup. The Colab notes below still apply to Colab sessions.


- **Local (Apple M5, MPS, 24 GB unified memory, no CUDA):** development, tests, data build, verifier, synthetic generation, smoke runs.
- **Google Colab Pro (CUDA):** probe feature extraction, every fine-tune, scaling and self-training runs. A100 for full fine-tunes; L4 acceptable for probes and smoke. Not T4 (no bf16) and not TPU.
- Same code and configs on both; Colab adds `--overlay configs/colab.yaml` (paths on Google Drive) and is driven by `notebooks/colab_launcher.ipynb`, which clones the GitHub repo at `main`.
- Colab sessions can disconnect, so the trainers in Phase 6 must checkpoint every epoch and resume from the last checkpoint in the run directory.
- Measured 2026-10-11: a ViT-L fine-tune step (batch 16, 448 px) takes ~25 s on the M5's GPU (MPS), so the fine-tune smoke takes ~13 min locally; the probe smoke takes 2.6 min. Gate 6.6 (< 10 min) is therefore timed on a Colab GPU; the Mac runs the probe smoke and the tiny-model tests.
- On Colab, data and caches sit on the VM's local disk (`configs/colab.yaml`, `GEO_OVERLAYS`), unpacked from one tarball on Drive built by `python -m data.prepare --download`; run outputs go to Drive.

---

## Phase 1 — Environment and skeleton

- [x] 1.1 Create the repo with the layout in §0.4. Add `.gitignore` for `data/raw`, `runs/`, checkpoints.
- [x] 1.2 Python 3.11 environment. Pin: `torch`, `torchvision`, `transformers`, `timm`, `numpy`, `pandas`, `matplotlib`, `pyyaml`, `scikit-learn`, `pytest`, `tqdm`. Optional: `wandb`.
- [x] 1.3 Confirm GPU: print device, VRAM, CUDA version. Record in README.
- [x] 1.4 Write `configs/base.yaml` with every tunable: paths, backbone id, resolution, batch size, LRs, LLRD factor, epochs, seed, relation tiers, tolerance values (empty until Phase 3), augmentation flags.
- [x] 1.5 Write `experiments/run.py`: loads config, sets seeds (`torch`, `numpy`, `random`, CUDA deterministic), dispatches to probe / finetune / evaluate / scaling / selftrain.
- [x] 1.6 Write `tests/test_skeleton.py` that imports every module. **GATE: `pytest` green.**

---

## Phase 2 — Data acquisition and unified schema

- [x] 2.1 Download Geometry3K (Lu et al., 2021, Inter-GPS repo), PGDP5K (Zhang et al., 2022) and PGPS9K (Zhang et al., 2023). Record URLs, licences and citation in `data/README.md`.
- [x] 2.2 Inspect each dataset's annotation files by hand (open 5 examples each). Document the actual field names: how points are named, how coordinates are stored (pixel vs normalised), how lines/segments/circles are represented, how relations are stored (logic forms, text, both).
- [x] 2.3 Define `data/schema.py`:
  ```
  Diagram:
    id, source ('geometry3k'|'pgdp5k'|'pgps9k'|'synthetic'), image_path,
    width, height,
    points: {name: (x, y)}              # pixel coordinates, origin top-left
    segments: [(p1, p2)]                # endpoints by name
    lines: [[p1, p2, ...]]              # collinear point groups if provided
    circles: [{center: name, radius: float, points_on: [names]}]
    angles: [(p1, vertex, p2)] if provided
    asserted: [Relation]                # from human annotation
    geometric: [Relation]               # filled in Phase 3
    flags: {not_to_scale: bool, ...}
  Relation: {type, args: [primitive refs], source: 'asserted'|'geometric', margin: float}
  ```
- [x] 2.4 Write one loader per dataset converting raw → `Diagram`. Each loader has a test that round-trips 3 diagrams and renders them with points and segments drawn over the image for visual confirmation (save to `reports/figures/loader_check_<source>.png`).
- [x] 2.5 `data/build.py`: write `data/processed/<source>.jsonl`. Print per-source counts: diagrams, points, segments, circles, asserted relations by type.
- [x] 2.6 **Relation census.** Table `reports/tables/relation_census.csv`: positives per relation type per source. Decide Tier 2 inclusion from this table and record the decision in §0.2 and CHANGELOG. **GATE: census table exists; Tier 2 decided.**
- [x] 2.7 Coordinate sanity: for each source, plot image size distribution and verify that annotated points fall inside the image (count violations; investigate if > 1%).

---

## Phase 3 — Verifier

- [x] 3.1 `verifier/primitives.py`: vector helpers (direction, length, cross, dot, point-to-line distance, foot of perpendicular, angle between rays), all in float64.
- [x] 3.2 `verifier/checks.py`: one function per relation, each returning `(is_true: bool, margin: float)` where `margin` is the raw quantity before thresholding so distributions can be plotted. Definitions:
  - `parallel(s1, s2)`: `|sin θ|` between directions, where θ is the angle between the segments. True if `|sin θ| ≤ sin(tol_angle)`.
  - `perpendicular(s1, s2)`: `|cos θ|`. True if `|cos θ| ≤ sin(tol_angle)`.
  - `equal_length(s1, s2)`: `| |s1| − |s2| | / max(|s1|, |s2|)`. True if `≤ tol_rel`.
  - `on_circle(p, c)`: `| |p − center| − r | / r`. True if `≤ tol_rel`.
  - `tangent(s, c)`: `| dist(line(s), center) − r | / r ≤ tol_rel` **and** the foot of the perpendicular from the center lies on the segment (or on the line if the dataset marks infinite lines).
  - Tier 2 if enabled: `collinear(a, b, c)`: `|cross(b−a, c−a)| / (|b−a| · diagram_scale)`; `midpoint(m, s)`: relative distance of m from the segment midpoint; `equal_angle(∠1, ∠2)`: absolute angle difference in degrees; `perp_bisector(s1, s2)`: perpendicular **and** passes through midpoint of s2.
  - `diagram_scale` = diagonal of the bounding box of all points; used to make relative tolerances scale-free.
- [x] 3.3 Unit tests with hand-constructed coordinates for every check: exact positive, exact negative, boundary case.
- [x] 3.4 **Calibration.** `verifier/tolerance.py`: for each relation, compute the margin for every candidate query in Geometry3K train, split by whether the query is `asserted` true or not. Plot both histograms on a log x-axis (`reports/figures/tolerance_<relation>.png`). Choose the threshold at the valley between the two modes. Record the chosen values in `configs/base.yaml` and the plan. Expect roughly 1–3° for angles and 2–5% for relative lengths, but take the numbers from the plots. *(Done 2026-10-10: values in §0.2 #5; rule = valley of the smoothed log-margin density, knee of the asserted CDF where there is no valley; figure `reports/figures/tolerance_all.png`.)*
- [x] 3.5 **Gold set.** Sample 100 queries stratified by relation and source. Hand-label each by looking at the diagram (not the annotation). Store in `verifier/gold/gold.csv` with your label and a note. Compute verifier false-positive and false-negative rates against gold. Report in `reports/tables/verifier_audit.csv`. **GATE: FP and FN rates measured; both below an agreed ceiling (propose 3%) or the tolerance is revisited.** *(Ceiling agreed: 3%. Tooling done 2026-10-10: `python -m verifier.gold sample` → `verifier/gold/gold_queries.csv` + `verifier/gold/label.html`; `python -m verifier.gold score` → `reports/tables/verifier_audit.csv`. Awaiting hand labels.)* *Round 1 (2026-10-10) failed: FP 7.7%, FN 21% (sample enriched near the tolerance); tolerances revised in §0.2 #5. Pre-registered gate for round 2: population-weighted FP and FN rates (weights undo the near-tolerance oversampling) on 60 new queries, both ≤ 3%; unweighted stress rates and a blind re-check of round-1 disagreements (mixed with as many agreements) are reported alongside.* *Round 2 (2026-10-10, 60 held-out queries): raw population-weighted FP 0.1%, FN 43.9% (FAIL), driven by two items the labeller then confirmed as slips on re-inspection (a radius labelled tangent; sides 113 vs 148 px labelled equal); recorded in `verifier/gold/adjudication.csv`, raw labels untouched. **Adjudicated: weighted FP 0.0%, FN 0.0% → PASS** (unweighted stress: FP 3.8%, FN 0%). Also fixed from the gold rounds: chords are not tangents, tangency may sit at a segment end, coincident lines are not parallel, nested segments on one line are not queries.*
- [x] 3.6 **Not-to-scale detection.** A diagram is flagged `not_to_scale` if any asserted relation fails its check by a large margin (propose: angle > 10°, relative length > 15%). Report the flagged fraction per source. Flagged diagrams stay in the data with the flag; they are evaluated as a separate category.
- [x] 3.7 **Geometric label generation.** For every diagram, enumerate all candidate queries (§4.1) and fill `geometric` relations via the checks. Store margins.
- [x] 3.8 **Dataset audit.** `verifier/audit.py` → `reports/tables/audit_<source>.csv`: for each relation type: asserted ∧ geometric, asserted ∧ ¬geometric, ¬asserted ∧ geometric (true-but-unmarked), with examples saved as figures. This is Result 1 of the report.

---

## Phase 4 — Synthetic diagram generator

- [x] 4.1 `data/synthetic/generator.py`: sample 4–10 points in a canvas with minimum separation; sample 3–8 segments between them; 0–2 circles. Inject relations by construction (place a point exactly on a circle, construct a segment parallel to another, etc.) with a configurable rate so positives are not rare. Then jitter coordinates by a small, configurable amount so that "geometric" truth is decided by the verifier, not by the construction. *(Done 2026-10-11: `data/synthetic/generator.py`; construction weights in config, tangent doubled; exact pre-jitter coordinates kept in `meta`.)*
- [x] 4.2 `data/synthetic/render.py`: matplotlib, black lines on white, point labels A, B, C…, right-angle markers and tick marks drawn when the relation is injected (so "asserted" exists for synthetic too). Style knobs: line width, font, label offset, slight rotation, optional scan noise and JPEG artefacts. *(Done 2026-10-11: `data/synthetic/render.py`; 28 ms/diagram; noisy styles stored as JPEG q95 to keep the pools small.)*
- [x] 4.3 Generate three pools: `synth_train` (sizes 1k, 10k, 50k for Part A), `synth_extrap` (12–20 points, 1.5–2× the max seen in training), and `synth_unlabelled` (images only; coordinates kept in a separate sealed file for post-hoc analysis in Part B). *(Done 2026-10-11: `python -m data.synthetic.build`, 50k train (1k/10k/50k nested prefixes, manifests in `data/processed/splits/`), 1k extrap, 2k unlabelled with the annotation sealed (sha256 in CHANGELOG); 2.1 GB of images, regenerable from the seed.)*
- [x] 4.4 Run the verifier on synthetic data; confirm that injected relations pass and random pairs mostly fail. Save 20 rendered examples to `reports/figures/synthetic_examples.png`. *(Done 2026-10-11: `python -m data.synthetic.check` → `reports/tables/synthetic_verifier_check.csv`: injected relations pass 99.4–100%, random candidates 1–4% (equal_length 11%); figure `reports/figures/synthetic_examples.png`.)*

---

## Phase 5 — Queries, overlays, dataset

- [x] 5.1 `data/queries.py`: enumerate candidate queries per diagram by arity: all segment pairs (parallel, perpendicular, equal_length), all point–circle pairs (on_circle), all segment–circle pairs (tangent), plus Tier 2 tuples if enabled. Cap per diagram (propose 60) by random sampling to bound dataset size; keep all positives. *(Done 2026-10-11: `python -m data.queries`; training keeps all positives under the cap, evaluation samples uniformly; nested same-line pairs excluded since Phase 3.)*
- [x] 5.2 Labels per query: multi-label vector over the vocabulary (a segment pair can be both perpendicular and equal_length), one from `asserted`, one from `geometric`. Store both. *(Done: `geo`/`asr` vectors over [parallel, perpendicular, equal_length, on_circle, tangent, collinear], none = all zeros, plus an applicability mask for relations a query kind cannot have.)*
- [x] 5.3 `data/overlay.py`: draw the queried primitives on the image in a fixed highlight colour (propose red, 3 px, with semi-transparent halo), everything else unchanged. For a circle query, highlight the circle; for a point, a filled marker. Unit test: overlay does not alter pixels outside the highlighted primitives. *(Done: `data/overlay.py`. Colour changed from the proposed red to green (0,200,0): red/orange/cyan occur in 13–15% of real diagrams as marks, green in 0% of 600 sampled. *Re-measured 2026-10-13 by script (`python -m data.colour_check`): the highlight colour itself in 0 of 600 images (RGB distance < 80), green-family ink in 1.0%.* Drawn after resizing to 448 so the highlight has the same width in every input. Unit test passes.)*
- [x] 5.4 Negative sampling: keep the natural ratio for evaluation; for training, downsample `none` to at most 3× the positives per relation (configurable) so the head does not collapse to "none". *(Done: training only, per query kind (each relation belongs to one kind), all-negative queries ≤ 3× positive queries.)*
- [x] 5.5 `data/augment.py` (relation-preserving only): rotation ±180°, horizontal/vertical flip, scale 0.8–1.2, line-thickness jitter (morphological), Gaussian and JPEG noise, slight contrast change. **No** shear or perspective (these break parallelism and equality). Augmentations are applied to the image after overlay so the highlight transforms with the diagram. *(Done: `data/augment.py`. Scale is applied as ink-box crop + resize to s × 448 capped at 448, so nothing is cut; line thickening = 3×3 min filter.)*
- [x] 5.6 `data/dataset.py`: returns `(image_tensor, label_geometric, label_asserted, meta)`. Resize long side to 448, pad to multiple of patch size, normalise with the backbone's mean/std. Cache preprocessed overlays to disk as PNG keyed by (diagram id, query id). *(Done: `data/dataset.py`; cache at `<cache>/overlays/<settings hash>/...`; collate pads batches with normalised white.)*
- [x] 5.7 Splits as in §0.2 #9 (revised): figure-level re-split of Geometry3K figures, PGDP5K held-out figures, PGPS9K pool, synthetic as in Phase 4. Write split manifests to `data/processed/splits/*.txt`. Test that no diagram id and no figure appears in two splits. *(Real-data part done early, 2026-10-10, `data/splits.py`; synthetic pool manifests written by `data.synthetic.build` 2026-10-11.)* *(Done 2026-10-11: dataset units are disjoint by diagram id, checked by `data.dataset.check_disjoint` and tests.)*
- [x] 5.8 **GATE: `configs/smoke.yaml` builds the dataset on 50 diagrams and one batch loads with correct shapes.** *(PASSED 2026-10-11: `python -m data.dataset --config configs/smoke.yaml`: every unit builds on 50 diagrams (train: 50 real + 50 synthetic) and one batch per unit loads as (16, 3, H≤448, W≤448) with (16, 6) labels and masks.)*

---

## Phase 6 — Model

- [x] 6.1 `models/backbone.py`: load DINOv3 ViT-L/16 (fallback DINOv2-L/14) from the hub. Expose: patch tokens, CLS token, pooled feature (CLS concatenated with mean of patch tokens). Support variable input size. Gradient checkpointing on. *(Done 2026-10-11: `models/backbone.py`; pooled = CLS ++ mean patch (2048-d); register tokens dropped; fallback recorded in metrics; image-processor mean/std checked against the config (DINOv2-L: match).)*
- [x] 6.2 `models/head.py`: LayerNorm → Linear → sigmoid per relation (multi-label, BCE loss with per-class positive weights from the census). *(Done: `models/head.py`; masked BCE over applicable relations; positive weights = negatives/positives in the training split, capped at 50.)*
- [x] 6.3 `models/probe.py`: extract frozen features once for all splits (cache to disk), train the head only with AdamW, LR 1e-3, 50 epochs, early stopping on val macro-F1. *(Done: `models/probe.py`; features cached per backbone/overlay settings/index and verified by query ids.)*
- [x] 6.4 `models/finetune.py`: full fine-tune. AdamW, head LR 1e-3, backbone LR 2e-5 with layer-wise decay 0.8 from the last block downward, weight decay 0.05, cosine schedule with 1-epoch warm-up, 30 epochs (revised 2026-10-12, §0.2 #11: LP-FT, linear-probe stage then 2 epochs of 40,000 images; `finetune.lp_queries`, `finetune.epoch_images`, `finetune.real_fraction`), batch 16 (gradient accumulation if needed), mixed precision, EMA of weights (0.999), checkpoint best val macro-F1. Log loss, per-class F1, LR every epoch. *(Done: `models/finetune.py`; EMA with warm-up; best.pt = EMA weights by val macro-F1; last.pt for resume; history.jsonl per epoch incl. peak LRs.)*
- [x] 6.5 Seeds: every run takes `seed` from config; three seeds (0, 1, 2) for every reported number. *(Done: seed from config drives torch/numpy/random, data order, augmentation and head init; `--set seed=k`.)*
- [x] 6.6 **GATE: smoke run of probe and fine-tune completes end to end in under 10 minutes and writes `runs/smoke_*/metrics.json`.** *(PASSED 2026-10-11 on Colab, NVIDIA L4: `python -m experiments.gate smoke` — probe and fine-tune smoke both ok, both loaded facebook/dinov3-vitl16-pretrain-lvd1689m (access check: DINOv3ViTModel, 1024-d, 24 layers, patch 16, 4 register tokens, 303.1M params, normalisation matches), 97 s each, predictions for every unit, best.pt written.)*

---

## Phase 7 — Experiments

Core (must):

- [x] 7.1 Linear probe, 3 seeds, train on Geometry3K train only. *(Done 2026-10-12, lab: geometric macro-F1 geometry3k_test 0.595 ± 0.013, pgdp5k_heldout 0.517 ± 0.003.)*
- [x] 7.2 Full fine-tune, 3 seeds, train on Geometry3K train only. *(Done 2026-10-12, lab, LP-FT 2 x 40k images: 0.867 ± 0.007 / 0.869 ± 0.008.)* *(2026-10-20: retrained on Colab A100 as `finetune_colab` (same config, 3 seeds) because the lab did not keep these checkpoints; needed for the square-padded re-evaluation, §0.2 #10 note.)* *(2026-10-25: done, 3 seeds, batch-padded 0.868 ± 0.007 / 0.869 ± 0.008 / 0.629 ± 0.018, reproducing the lab's 0.867 / 0.869 / 0.631; square-padded 0.872 ± 0.008 / 0.874 ± 0.008 / 0.629 ± 0.018 = the report's 7.2.)*
- [x] 7.3 Full fine-tune, 3 seeds, train on Geometry3K train + synth_train 10k. *(Done 2026-10-12, lab: 0.871 ± 0.010 / 0.885 ± 0.004; synth_extrap 0.713 ± 0.009 vs 0.631 ± 0.017. Teacher for Part B: seed 2, best val.)*
- [x] 7.4 Evaluate every checkpoint on the three test sets plus the not-to-scale subset, against both label sets. *(Every run evaluates all units against both label sets at its end; `python -m analysis.phase7` adds the not-to-scale subset (7 diagrams, 353 queries, geometry3k_test + pgdp5k_heldout pooled) → `reports/tables/not_to_scale_eval.csv`, and the summary tables. Done 2026-10-13 for all 24 runs: `reports/tables/phase7_summary.md`. Not-to-scale geometric macro-F1 is close to the full test sets (fine-tune 0.859, + 10k synthetic 0.874), but 7 diagrams support no claim.)*

Ablations (cheap, one seed each):

- [x] 7.5 Backbone: DINOv2-L vs DINOv3-L (same config). *(Done 2026-10-13, Colab, seed 0: DINOv2-L 0.913 / 0.920 / 0.754 (geometry3k_test / pgdp5k_heldout / synth_extrap) vs DINOv3-L 0.867 ± 0.007 / 0.869 ± 0.008 / 0.631 ± 0.017. DINOv2 is better by ~0.05 at this budget; one seed. Note DINOv2/14 sees 1,024 patch tokens at 448 px vs 784 for DINOv3/16.)*
- [x] 7.6 Resolution: 224 vs 448. *(Done 2026-10-12, lab, seed 0.)*
- [x] 7.7 No augmentation vs full augmentation. *(Done 2026-10-12, lab, seed 0.)*
- [x] 7.8 Highlight overlay vs no overlay (query given only as a text token index — expected to fail, which shows the overlay is doing the work). *(Implemented 2026-10-13: `configs/ablation_nooverlay.yaml`; the query is the kind plus the point names as hashed token indices (`data/query_tokens.py`), projected to one vector that cross-attends over the CLS and patch tokens (`models.head.QueryAttention`), concatenated with the pooled feature; same budget, no LP stage (it needs the query encoder). Done 2026-10-13, Colab, seed 0: 0.431 / 0.429 / 0.137, below even the linear probe with the overlay (0.595 / 0.517 / 0.185): the overlay does the work.)*

Part A — verifier as unlimited labeller (must):

- [x] 7.9 Fine-tune on Geometry3K train + synth_train at 0, 1k, 10k, 50k. Three seeds at 0 and 50k, one seed in between. Plot held-out-source and extrapolation macro-F1 against synthetic size (`reports/figures/scaling.png`). *(Done 2026-10-13 (0, 10k: lab, 3 seeds each; 1k: 1 seed and 50k: 3 seeds on Colab). pgdp5k_heldout 0.869 → 0.885 (1k) → 0.885 (10k) → 0.882 (50k); synth_extrap 0.631 → 0.718 → 0.713 → 0.707; geometry3k_test flat (0.867-0.876). The gain is reached by 1k diagrams and does not grow with more, at a fixed budget of 80k images with 50% synthetic.)*

Part B — self-training on real unlabelled scans (if Part A done):

- [x] 7.10 `verifier/consistency.py`: given a predicted relation set for a diagram, compute the deductive closure with rules: parallel transitivity; perpendicular ∘ parallel → perpendicular; equal_length transitivity; two segments both perpendicular to a third are parallel; a segment cannot be both parallel and perpendicular to the same segment. Return the set of contradictions and the closure size. *(Done 2026-10-13: union-find over parallel classes of carrier lines (sub-segments of one line are one entity), perpendicular edges between classes, equal_length classes; contradiction C1 = a pair both parallel and perpendicular after closure. C2 = a derived fact that the model confidently predicted false (`selftrain.consistency_negatives`, on). Measured on a 150-diagram sample of the pool with the real teacher (τ 0.9): C1 alone flags 1/150 diagrams (removes 1% of the confident entries), too few for the two filters to differ; C1 + C2 flags 13/150, removes 16% of the entries, and the removed pseudo-positives have precision 0.74 against 0.92 for the kept ones (confidence filter alone 0.89). Caveat: tolerance-based relations are not exactly transitive, so C2 also fires on 10.4% of the pool diagrams with the verifier's own labels (C1 on 0%).)*
- [x] 7.11 `experiments/selftrain.py`: teacher = best 7.3 model. Pseudo-label the PGPS9K pool from §0.2 #9 (d) (coordinates and labels unused). Two filters: (a) confidence > τ; (b) confidence > τ **and** diagram-level relation set has zero contradictions after closure. Retrain student from the 7.3 checkpoint on Geometry3K + filtered pseudo-labels, one round, 3 seeds each. Evaluate on all test sets. Post-hoc: unseal PGPS9K coordinates and report pseudo-label precision under each filter (this is the evidence that the consistency filter removed wrong labels, or didn't). *(Implemented 2026-10-13: `experiments/selftrain.py`; teacher finetune_synth10k_s2; pool = PGPS9K pool figures outside the training split (2,037 diagrams, 66,019 queries at the evaluation cap); an entry is kept when max(p, 1-p) > 0.9 (others masked), consistency drops whole diagrams with a contradiction; student from the teacher's weights, 2 x 40k images, 50% Geometry3K / 50% pseudo-labelled; plus a control student with no pseudo-labels. Precision of both filters and of the removed entries in every student's metrics.json. Done 2026-10-13, Colab: the consistency filter flagged 162/2,037 pool diagrams and raised pseudo-positive precision from 0.882 to 0.898 (removed entries 0.798; equal_length is the weak relation, 0.71 kept, 0.64 removed); students geometry3k_test / pgdp5k_heldout / synth_extrap: confidence 0.886 ± 0.003 / 0.896 ± 0.003 / 0.737 ± 0.009, consistency 0.895 ± 0.010 / 0.903 ± 0.006 / 0.724 ± 0.010, teacher (s2) 0.882 / 0.889 / 0.716, **control (same continued training, no pseudo-labels, 1 seed) 0.909 / 0.919 / 0.741**: the students' gain over the teacher comes from the extra training, and the pseudo-labels (50% of every batch) cost ~0.015-0.02 against the control; consistency beats confidence on the real test sets by less than 1-2 sd.)* *(2026-10-25, control seeds 1-2 done on Colab, 3 seeds: batch-padded 0.903 ± 0.005 / 0.914 ± 0.005 / 0.734 ± 0.007; square-padded (the report's numbers) control 0.915 ± 0.005 / 0.924 ± 0.003 / 0.734 ± 0.007 vs consistency 0.907 ± 0.006 / 0.912 ± 0.005 / 0.724 ± 0.010 vs confidence 0.898 ± 0.002 / 0.906 ± 0.003 / 0.737 ± 0.009, teacher (sq, s2) 0.887 / 0.896 / 0.716. Conclusion unchanged and now beyond seed noise: the pseudo-labels cost ~0.01-0.02 on the real test sets against the control; consistency beats confidence by ~0.006-0.009 (1-2 sd).)*
- [x] 7.12 Record every run in `runs/log.csv` with config hash, seed, best epoch, all test metrics. *(Done: 24 runs (11 lab, 13 Colab) merged into the local runs/log.csv; every run dir has config.yaml, metrics.json, history.jsonl and predictions for the four evaluation units.)*

---

## Phase 8 — Analysis and figures

- [x] 8.1 `analysis/metrics.py`: per-relation precision, recall, F1; macro-F1; ECE (15 bins); bootstrap 95% CIs over diagrams (1,000 resamples). *(Done 2026-10-25: `analysis.metrics` (ECE of the positive-class probability over applicable entries, pooled and per relation; per-relation thresholds maximising F1 on the run's own validation predictions; bootstrap over diagrams by resampling per-diagram confusion counts, also jointly over seeds for a CI of the seed-mean) and `python -m analysis.phase8` → `reports/tables/phase8_metrics.csv` (per run), `phase8_summary.csv`, `phase8_summary.md` (report set per §0.2 #10). Geometric macro-F1 at val-tuned thresholds, seed-mean [95% CI], geometry3k_test / pgdp5k_heldout / synth_extrap: probe 0.569 [0.522, 0.606] / 0.537 [0.503, 0.567] / 0.182 [0.160, 0.201]; 7.2 (sq_finetune_colab) 0.886 [0.867, 0.902] / 0.894 [0.884, 0.903] / 0.658 [0.634, 0.678]; Part A 1k 0.896 / 0.894 / 0.721, 10k 0.906 [0.889, 0.920] / 0.912 [0.901, 0.921] / 0.764 [0.744, 0.782], 50k 0.905 / 0.911 / 0.757; DINOv2-L 0.940 / 0.947 / 0.769; no overlay 0.462 / 0.443 / 0.129; Part B confidence 0.915 [0.899, 0.929] / 0.924 [0.913, 0.931] / 0.755, consistency 0.925 [0.912, 0.936] / 0.928 [0.919, 0.935] / 0.758, control 0.929 [0.915, 0.940] / 0.933 [0.923, 0.940] / 0.772. Tuning the thresholds adds +0.01-0.02 on the real sets and +0.03-0.05 on synth_extrap over 0.5 (the chosen thresholds sit mostly at 0.7-0.99: the positive-weighted loss makes the models over-predict positives). CI half-widths: ~0.015-0.02 on geometry3k_test, ~0.01 on pgdp5k_heldout, ~0.02 on synth_extrap; seed sd is smaller than the CI for every fine-tune. ECE (geometric, pooled): probe 0.13-0.20, fine-tunes 0.02-0.07, no overlay 0.22-0.46, Part B control the best calibrated (0.023 / 0.026 / 0.035). At tuned thresholds the Part B control still leads but the gap to the consistency student shrinks to 0.004-0.005 (CIs overlap; ~1 sd), so part of the 0.01-0.02 'cost' of pseudo-labels at threshold 0.5 was calibration, not accuracy.)*
- [x] 8.2 `analysis/headline_table.py`: for each test set, count queries solved by probe only, fine-tune only, both, neither (at the operating threshold tuned on val). This mirrors the thesis's base-only / RL-only / both table. *(Done 2026-10-25: `python -m analysis.headline_table [--probe sq_probe] [--finetune sq_finetune_colab]` → `reports/tables/headline_table.{csv,md}`. A query is solved when every applicable decision is right (geometric labels, each model's own val-tuned thresholds); seeds matched and averaged; also per (query, relation) decision. geometry3k_test: both 67.2%, fine-tune only 22.6%, probe only 3.5%, neither 6.7%; pgdp5k_heldout 55.7 / 33.0 / 2.9 / 8.4; synth_extrap 70.1 / 19.4 / 3.7 / 6.8. Per decision the fine-tune's gain is concentrated in equal_length (fine-tune only 23% / 39% / 35% of its decisions on the three sets, and 9-10% solved by neither), then perpendicular (12-14%); parallel, on_circle and collinear are mostly solved by both.)*
- [ ] 8.3 Asserted vs geometric: model scores against both label sets side by side; discussion of which "errors" are the labels' errors.
- [x] 8.4 `analysis/error_analysis.py`: F1 by number of points in the diagram, by segment length (short segments are hard), by relation, by source, by not-to-scale flag, by augmentation condition. Confusion between parallel/none and perpendicular/none at near-threshold margins: plot F1 against verifier margin to show where the model fails relative to the tolerance. *(Done 2026-10-25: `python -m analysis.error_analysis` → `reports/tables/error_analysis.{csv,md}`, `error_margin.csv`, `reports/figures/error_margin.png`, `error_breakdowns.png`; seeds pooled, geometric labels, val-tuned thresholds; default experiments sq_finetune_colab (figures), sq_finetune_synth10k, sq_selftrain_control, sq_probe, sq_ablation_dinov2. The augmentation condition is the 7.7 training ablation, not a test-query property, so it is not a breakdown. Findings for 7.2 (geometry3k_test / pgdp5k_heldout / synth_extrap): **segment length is the strongest factor**: macro-F1 0.67 / 0.74 / 0.35 when the shortest queried segment is under 10% of the diagram diagonal, 0.57 / 0.69 / 0.56 at 10-20%, 0.81 / 0.81 / 0.61 at 20-35%, 0.88 / 0.89 / 0.64 above; collinear collapses on short lines (0.00 and 0.21 at 10-20%) and equal_length is the weakest relation at every length (0.26-0.72). Points: flat (0.86-0.90) up to 12 points on Geometry3K, 0.80 at 13-16; PGDP5K 0.92 (≤4) to 0.81 (17-20); synth_extrap 0.62-0.68 over 12-20 points. Source: the 12 Geometry3K-annotated test figures 0.908 vs 0.886 for PGDP5K-annotated. Not to scale: 0.885 vs 0.886 (3 figures) and 0.834 vs 0.894 (4 figures), no claim. Margin: the share of correct decisions dips at the tolerance and recovers by 2× it: parallel 0.98 ([0, 0.25) × tol) → 0.76 ([0.5, 1)) → 0.54 ([1, 1.5)) → 1.00 (≥ 2×); perpendicular 0.96 → 0.56 → 0.84 → 0.65 → 0.95; equal_length is poor across a wide band (0.52-0.72 from 0.5× to 2×, and only 0.80 for clear positives under 0.25×: the eye does not measure lengths to 10%); on_circle ≈ 1 everywhere. On synth_extrap collinear falls to 0.15 at [0.5, 1): the collinear tolerance (1.19% of the diagonal) is ~6 px on a 448 px image of a 12-20-point figure, at the resolution limit.)*
- [ ] 8.5 `analysis/attention_maps.py`: attention rollout or last-layer CLS attention for 10 diagrams (5 correct, 5 wrong), saved to `reports/figures/attention_*.png`. Question answered: does the model attend to the highlighted primitives and to markers (tick marks, right-angle symbols), or elsewhere?
- [ ] 8.6 Figures list for the report (all generated by scripts): loader check, tolerance histograms (one combined), synthetic examples, scaling curve, headline table, per-relation F1 bars per test set, attention maps, audit examples (asserted-but-false, true-but-unmarked).

---

## Phase 9 — Report (max 2,500 words)

Rubric mapping and word budget:

| Section | Words | Rubric line |
|---------|-------|-------------|
| 1. Introduction and research question | 250 | Structure |
| 2. Dataset selection and justification; the audit (Result 1) | 400 | Dataset justification & pre-processing (15%) |
| 3. Pre-processing: queries, overlays, augmentation, synthetic generator | 250 | Dataset justification & pre-processing |
| 4. Model and training: DINOv3 ViT-L, probe vs fine-tune, LLRD, configuration | 400 | Model selection & implementation (25%) |
| 5. Verifier: definitions, calibration, gold-set FP/FN | 250 | Evaluation (20%) |
| 6. Results: three test sets, headline table, scaling, self-training, ablations | 500 | Evaluation & performance analysis (20%) |
| 7. Bias and ethics: convention bias, not-to-scale, downstream uses (grading, tutoring, accessibility), failure costs | 350 | Ethical & societal considerations (20%) |
| 8. Conclusion and link to further work | 100 | Structure |
| References (not counted) | — | Referencing |

- [ ] 9.1 Draft in `reports/report.md`, then move to the submission format. Every number in the text is copied from a file in `reports/tables/` by a script, not typed.
- [ ] 9.2 Reference list: Geometry3K/Inter-GPS, PGDP5K, PGPS9K, DINOv2, DINOv3, ViT, AlphaGeometry, GEOS, FormalGeo, GeoX (verify each citation against the actual paper).
- [ ] 9.3 GenAI acknowledgement section as required by the subject outline (tools used, for what).
- [ ] 9.4 Word count check; figures captioned; tables numbered.

---

## Phase 10 — Submission

- [ ] 10.1 README with: one-command setup, one-command data build, one-command reproduction of every table and figure, expected runtime per step, hardware used.
- [ ] 10.2 Fresh-clone test: clone the repo into a new directory and run smoke mode; it must pass.
- [ ] 10.3 Notebook `notebooks/walkthrough.ipynb` that loads the best checkpoint, runs one diagram end to end, and shows the overlay, the prediction, the verifier verdict and the attention map.
- [ ] 10.4 Tag the release; submit the GitHub link and the report via Moodle/Turnitin as required.
- [ ] 10.5 Archive: `runs/log.csv`, all checkpoints' metrics, and the gold set, so the thesis can cite them.

---

## Appendix A — Working with Claude Code on this project

- Start every session with: "Read GEOMETRY_RELATIONS_PLAN.md and CHANGELOG.md. We are on Phase N, task N.M."
- Ask for one task at a time, and for the test before the implementation when a module is new.
- After any data inspection task, ask it to paste the actual field names it found into `data/README.md`.
- Before any full run, ask it to run smoke mode and show the metrics file.
- Ask it to update `CHANGELOG.md` at the end of each session with what changed and what was found, including negative findings.
- When a result is surprising, ask for the three most likely bugs that would produce it before accepting it.

## Appendix B — Things that will go wrong

- Annotation formats differ between the three datasets more than their papers suggest; budget inspection time.
- Point names collide (two "A"s) or coordinates are normalised in one dataset and pixel in another.
- Tangent is rare; it may not reach 200 positives in real data and may need synthetic support.
- Class imbalance makes macro-F1 noisy; report CIs.
- A model can score well on `asserted` and badly on `geometric` if the dataset is drawn out of scale; that is a finding, not a bug.
- DINOv3 checkpoint name or licence may differ from what is written here; verify on the hub and fall back to DINOv2 without changing anything else.
