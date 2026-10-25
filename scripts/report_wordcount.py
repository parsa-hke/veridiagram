"""Word count of the compiled technical report body.

Counts the words of the compiled PDF from the Introduction heading to the end of the Conclusion,
captions and tables included; the title page, the generative-AI statement, the reference list and the
appendix are excluded. Exits 1 above the limit (plan §9: 2,500 words). Text is extracted with
pdftotext (poppler); without it, or without a compiled main.pdf, the script falls back to the LaTeX
sources with the generated macros expanded, which counts the same words to within a few per cent.

    python -m scripts.report_wordcount            # prints the count, exit 1 above the limit
    python -m scripts.report_wordcount --compile  # compile with tectonic first
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "reports" / "report"
LIMIT = 2500
BODY_START = re.compile(r"^\s*1\s+Introduction\s*$", re.M)
BODY_END = re.compile(r"^\s*(Generative AI Acknowledgement|References|A\s+Threshold)", re.M)


def pdf_text(pdf: Path) -> str:
    r = subprocess.run(["pdftotext", "-layout", str(pdf), "-"], capture_output=True, text=True, check=True)
    return r.stdout


def body_from_pdf(text: str) -> str:
    start = BODY_START.search(text)
    if not start:
        raise ValueError("Introduction heading not found in the PDF text")
    end = BODY_END.search(text, start.end())
    body = text[start.end(): end.start() if end else len(text)]
    body = re.sub(r"\f", " ", body)
    body = re.sub(r"^\s*\d+\s*$", "", body, flags=re.M)        # page numbers
    return body


def macros() -> dict[str, str]:
    src = (PAPER / "numbers.tex").read_text()
    return dict(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}", src))


def body_from_sources() -> str:
    main = (PAPER / "main.tex").read_text()
    names = [n for n in re.findall(r"\\input\{sections/([^}]+)\}", main) if n not in ("genai", "appendix")]
    text = "\n".join((PAPER / "sections" / f"{n}.tex").read_text() for n in names)
    m = macros()
    text = re.sub(r"(?<!\\)%.*", "", text)
    text = re.sub(r"\\([A-Za-z]+)(\{\})?", lambda g: m.get(g.group(1), " "), text)
    text = re.sub(r"\\(label|ref|cite|includegraphics|begin|end|input)\*?(\[[^]]*\])?\{[^}]*\}", " ", text)
    text = re.sub(r"[{}~&$^_]", " ", text)
    return text


def count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9][A-Za-z0-9'’.,:;%()\-/\\]*", text))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--compile", action="store_true", help="run tectonic before counting")
    ap.add_argument("--limit", type=int, default=LIMIT)
    args = ap.parse_args(argv)
    pdf = PAPER / "main.pdf"
    if args.compile:
        subprocess.run(["tectonic", "-c", "minimal", "main.tex"], cwd=PAPER, check=True, capture_output=True)
    if pdf.exists() and shutil.which("pdftotext"):
        n, how = count(body_from_pdf(pdf_text(pdf))), "compiled PDF"
    else:
        n, how = count(body_from_sources()), "LaTeX sources with macros expanded"
    print(f"{n} words in the body ({how}); limit {args.limit}")
    return 0 if n <= args.limit else 1


if __name__ == "__main__":
    sys.exit(main())
