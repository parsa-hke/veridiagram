"""scripts/report_numbers.py, report_wordcount.py, report_zip.py, report_figures.py and the LaTeX project."""

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import report_figures as rf
from scripts import report_numbers as rn
from scripts import report_wordcount as rw
from scripts import report_zip as rz

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "reports" / "report"
BODY = ["introduction", "dataset", "preprocessing", "model", "verifier", "results", "ethics", "conclusion"]


def section_files() -> list[Path]:
    main = (PAPER / "main.tex").read_text()
    return [PAPER / f"{n}.tex" for n in re.findall(r"\\input\{([^}]+)\}", main) if n != "numbers"]


def tex_sources() -> str:
    return "\n".join([(PAPER / "main.tex").read_text(), *(p.read_text() for p in section_files())])


def defined_macros() -> set[str]:
    return set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", (PAPER / "numbers.tex").read_text()))


def test_numbers_tex_is_up_to_date():
    assert rn.render(rn.collect()) == (PAPER / "numbers.tex").read_text()
    for k in rn.collect():
        assert re.fullmatch(r"[A-Za-z]+", k), k


def test_every_macro_used_is_defined():
    """A generated-looking control sequence in the sections must exist in numbers.tex."""
    defined = defined_macros()
    prefixes = tuple(sorted({re.match(r"[a-z]+", k).group(0) for k in defined}))
    used = set(re.findall(r"\\([A-Za-z]+)", tex_sources()))
    suspicious = {u for u in used if u.startswith(prefixes) and u[len(re.match(r"[a-z]+", u).group(0)):][:1].isupper()}
    assert suspicious <= defined, suspicious - defined


def test_sections_in_order():
    src = tex_sources()
    positions = [src.index(f"\\input{{sections/{s}}}") for s in BODY]
    assert positions == sorted(positions)
    assert "\\bibliography{refs}" in src and "\\input{sections/genai}" in src


def test_citations_match_bibliography():
    src = tex_sources()
    cited = {k.strip() for group in re.findall(r"\\cite\{([^}]*)\}", src) for k in group.split(",")}
    keys = set(re.findall(r"@\w+\{([^,]+),", (PAPER / "refs.bib").read_text()))
    assert cited <= keys, cited - keys
    assert keys <= cited, keys - cited


def test_inputs_and_figures_exist():
    for p in section_files():
        assert p.exists(), p
    for name in re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", tex_sources()):
        assert list((PAPER / "figures").glob(f"{name}.*")), name
        assert f"{name}.png" in rf.FIGURES, f"{name} is not copied by scripts/report_figures.py"


TYPED = re.compile(r"(?<![\w.])(?:\d+\.\d+|\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?\\?%)(?![\w])")


def test_no_typed_measured_number_next_to_a_macro():
    """A sentence that uses a numbers.tex macro must not also contain a hand-typed decimal,
    thousands-separated or percentage number: such a number is a measured quantity that should be a macro."""
    defined = defined_macros()
    offenders = []
    for p in section_files():
        text = p.read_text()
        text = re.sub(r"(?<!\\)%.*", "", text)
        text = re.sub(r"\\(cite|ref|label|includegraphics|eqref)\*?(\[[^]]*\])?\{[^}]*\}", " ", text)
        for sentence in re.split(r"(?<=[.!?])\s+|\n\n", text):
            uses_macro = any(f"\\{k}" in sentence for k in defined)
            if uses_macro and TYPED.search(sentence):
                offenders.append((p.name, TYPED.search(sentence).group(0), sentence.strip()[:90]))
    assert not offenders, offenders


def test_word_count_within_limit():
    n = rw.count(rw.body_from_sources())
    assert 0 < n <= rw.LIMIT, n


def test_zip_contents(tmp_path):
    out = rz.build(tmp_path / "r.zip")
    names = rz.listing(out)
    assert {"main.tex", "refs.bib", "numbers.tex", "README.md"} <= set(names)
    assert any(n.startswith("sections/") for n in names) and any(n.startswith("figures/") for n in names)
    assert not any(n.endswith((".pdf", ".log", ".aux")) and n.startswith("main") for n in names)


@pytest.mark.skipif(shutil.which("tectonic") is None, reason="tectonic not installed")
def test_compiles_and_pdf_word_count(tmp_path):
    shutil.copytree(PAPER, tmp_path / "p", ignore=shutil.ignore_patterns("main.pdf", "*.log", "*.aux", "*.zip"))
    r = subprocess.run(["tectonic", "-c", "minimal", "main.tex"], cwd=tmp_path / "p", capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    pdf = tmp_path / "p" / "main.pdf"
    assert pdf.exists()
    if shutil.which("pdftotext"):
        n = rw.count(rw.body_from_pdf(rw.pdf_text(pdf)))
        assert 0 < n <= rw.LIMIT, n


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
