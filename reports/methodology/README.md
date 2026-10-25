# Methodology Design and Research Implementation (MAAI901 workshop report)

Parsa Hosseini · 7744547 · University of Wollongong in Dubai

LaTeX source of the methodology report for the Veridiagram project (geometry diagram → formal
relations with a fine-tuned vision transformer and a coordinate verifier).

## Compile

- **Overleaf:** upload this folder as a zip (New Project → Upload Project). The default compiler
  (pdfLaTeX) with BibTeX produces `main.pdf`; no settings to change.
- **Locally:** `tectonic main.tex` (or `pdflatex`, `bibtex`, `pdflatex` ×2).

## Layout

```
main.tex          document (article, 12 pt, single spacing, 1-inch margins, Times)
sections/*.tex    one file per section, in the order required by the brief
refs.bib          references (IEEE style via IEEEtran.bst)
numbers.tex       GENERATED: one \newcommand per measured quantity
figures/          fig_recap, fig_design, fig_pipeline (generated) + three figures from reports/figures
```

## Generated inputs (from the project repository)

`numbers.tex` is written by `scripts/methodology_numbers.py` from `reports/tables/*.csv`,
`configs/base.yaml` and `reports/tables/methodology_inputs.json`; the text uses the macros, so no
measured number is typed by hand. The three `fig_*.pdf` files are drawn by
`scripts/methodology_figures.py`. `tests/test_methodology.py` checks that the macros are up to
date, that every citation resolves, that every input and figure exists, and that the project
compiles.
