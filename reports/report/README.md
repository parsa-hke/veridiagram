# Veridiagram: technical report (MAAI901 Assessment 2)

Parsa Hosseini · 7744547 · University of Wollongong in Dubai

LaTeX source of the 2,500-word technical report (plan §9): a fine-tuned DINOv3 reads formal
geometric relations in textbook diagrams, judged by a coordinate verifier.

## Compile

- **Overleaf:** upload `reports/Report_7744547.zip` (New Project → Upload Project). The default
  compiler (pdfLaTeX) with BibTeX produces `main.pdf`; no settings to change.
- **Locally:** `tectonic main.tex` (or `pdflatex`, `bibtex`, `pdflatex` ×2) in this directory.

## Layout

```
main.tex          article, 12 pt, A4, 1-inch margins, Times, IEEEtran.bst
sections/*.tex    one file per section (body, GenAI acknowledgement, appendix)
refs.bib          references, every entry checked against the published paper
numbers.tex       GENERATED: one \newcommand per measured quantity
figures/          copies of reports/figures/ made by scripts/report_figures.py
```

## Generated inputs and checks (from the repository root)

```
python -m scripts.report_numbers            # numbers.tex from reports/tables/*.csv and configs/base.yaml
python -m scripts.report_numbers --check    # exit 1 if numbers.tex would change
python -m scripts.report_figures            # copy the figures
python -m scripts.report_wordcount          # words from Introduction to Conclusion (captions and tables
                                            # included; title page, GenAI statement, references and
                                            # appendix excluded); exit 1 above 2,500
python -m scripts.report_zip                # reports/Report_7744547.zip for Overleaf
pytest tests/test_report.py                 # macros up to date, citations resolve, inputs and figures
                                            # exist, no typed measured number, word count, compile
```
