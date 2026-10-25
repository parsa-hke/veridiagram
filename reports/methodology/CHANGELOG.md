
## 2026-10-25 — Methodology workshop: LaTeX report (METHODOLOGY_PLAN.md)

Decisions
- The brief attached first (Overleaf paper drafting) is a later workshop; `PAPER_DRAFT_PLAN.md` kept
  for it. This one is "Methodology Design and Research Implementation": report only (its
  submission instructions list no presentation), LaTeX, 12 pt / 1.5 spacing / 1-inch margins /
  Times, IEEE references. Numbers rule: design choices, instrument validation (gold set), data
  facts and throughput are in; every Phase 7 outcome is out.

Done
- `scripts/methodology_numbers.py` -> `reports/methodology/numbers.tex` (184 macros from
  reports/tables, configs/base.yaml and `reports/tables/methodology_inputs.json`, a committed copy
  of the numbers that come from gitignored files: lab benchmark, query-index sizes, the
  figure-cluster overlap). `scripts/methodology_figures.py` -> recap construction, research-design
  and pipeline figures (PDF) + copies of three report figures. `scripts/methodology_zip.py` ->
  `reports/methodology/Methodology_7744547.zip` (Overleaf project).
- `reports/methodology/`: main.tex + nine section files in the brief's order (sections numbered
  3-11 to match it), refs.bib (Week 4's 12 + 13 methodology references, each checked against the
  published paper: LP-FT, AdamW, SGDR, ULMFiT, Polyak averaging, mixed precision, gradient
  checkpointing, registers, calibration/ECE, Noisy Student, bootstrap, Pineau reproducibility,
  datasheets, model cards). Compiles with tectonic (XeTeX) locally; Overleaf pdfLaTeX expected.
- `tests/test_methodology.py` (8 tests): numbers.tex regenerates identically, macro names are
  letters only, every macro used is defined, sections in the brief's order, citations <-> bib
  one-to-one, inputs and figures exist, zip contents, compile (skipped without tectonic).
- `tectonic` installed with Homebrew for the local compile check.

Found
- LaTeX macro names cannot contain digits: unit names like geometry3k are stripped to letters
  (`macro_name`), e.g. \nQueriesGeometryTest.
- At the agreed settings the draft runs to 27 pages (about 7,500 words of body text plus 6
  figures and 5 tables), above the 12-16 planned; decision on cutting vs spacing pending.
