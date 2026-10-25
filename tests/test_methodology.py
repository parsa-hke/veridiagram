"""scripts/methodology_numbers.py, scripts/methodology_zip.py and the LaTeX project's consistency."""

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import methodology_numbers as mn
from scripts import methodology_zip as mz

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "reports" / "methodology"
SECTIONS = [
    "Introduction and Recap of the Research Problem", "Research Design", "Dataset Description",
    "Data Preprocessing and Preparation", "Proposed Model or Transformer",
    "Training Strategy and Experimental Setup", "Baseline Methods or Comparative Models",
    "Evaluation Metrics and Analysis Plan", "Ethics, Bias, Privacy, and Reproducibility",
]


def tex_sources() -> str:
    """main.tex followed by its \\input files in document order."""
    main = (PAPER / "main.tex").read_text()
    inputs = [PAPER / f"{name}.tex" for name in re.findall(r"\\input\{([^}]+)\}", main) if name != "numbers"]
    return "\n".join([main, *(p.read_text() for p in inputs)])


def test_macro_names_are_letters_only():
    assert mn.macro_name("geometry3k_test") == "GeometryTest"
    assert mn.macro_name("equal_length") == "EqualLength"
    for k in mn.collect():
        assert re.fullmatch(r"[A-Za-z]+", k), k


def test_numbers_tex_is_up_to_date():
    assert mn.render(mn.collect()) == (PAPER / "numbers.tex").read_text()


def test_every_report_macro_used_is_defined():
    """Control sequences in the sections that look like generated macros must exist in numbers.tex
    (an undefined one would only surface at compile time)."""
    defined = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", (PAPER / "numbers.tex").read_text()))
    prefixes = tuple(sorted({re.match(r"[a-z]+", k).group(0) for k in defined}))
    used = set(re.findall(r"\\([A-Za-z]+)", tex_sources()))
    suspicious = {u for u in used if u.startswith(prefixes) and u[len(re.match(r"[a-z]+", u).group(0)):][:1].isupper()}
    assert suspicious <= defined, suspicious - defined


def test_sections_in_the_briefs_order():
    src = tex_sources()
    positions = [src.index(f"\\section{{{s}}}") for s in SECTIONS]
    assert positions == sorted(positions)
    assert "\\tableofcontents" in src and "\\begin{titlepage}" in src and "\\bibliography{refs}" in src


def test_citations_match_bibliography():
    src = tex_sources()
    cited = {k.strip() for group in re.findall(r"\\cite\{([^}]*)\}", src) for k in group.split(",")}
    keys = set(re.findall(r"@\w+\{([^,]+),", (PAPER / "refs.bib").read_text()))
    assert cited <= keys, cited - keys
    assert keys <= cited, keys - cited


def test_inputs_and_figures_exist():
    src = tex_sources()
    for name in re.findall(r"\\input\{([^}]+)\}", src):
        assert (PAPER / f"{name}.tex").exists(), name
    for name in re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", src):
        assert list((PAPER / "figures").glob(f"{name}.*")), name


def test_zip_contents(tmp_path):
    out = mz.build(tmp_path / "m.zip")
    names = mz.listing(out)
    assert "main.tex" in names and "refs.bib" in names and "numbers.tex" in names
    assert any(n.startswith("sections/") for n in names) and any(n.startswith("figures/") for n in names)
    assert not any(n.endswith((".pdf", ".log", ".aux")) and n.startswith("main") for n in names)


@pytest.mark.skipif(shutil.which("tectonic") is None, reason="tectonic not installed")
def test_compiles(tmp_path):
    shutil.copytree(PAPER, tmp_path / "p", ignore=shutil.ignore_patterns("main.pdf", "*.log", "*.aux", "*.zip"))
    r = subprocess.run(["tectonic", "-c", "minimal", "main.tex"], cwd=tmp_path / "p", capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    assert (tmp_path / "p" / "main.pdf").exists()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
