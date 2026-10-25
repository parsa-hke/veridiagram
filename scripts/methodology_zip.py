"""Package the methodology report as an Overleaf project (.zip).

Includes main.tex, sections/, refs.bib, numbers.tex (generated), figures/ and README.md;
excludes build products. Upload the zip to Overleaf (New Project -> Upload Project); it compiles
with the default pdfLaTeX + BibTeX.

    python scripts/methodology_zip.py            # -> reports/methodology/Methodology_7744547.zip
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "reports" / "methodology"
OUT = PAPER / "Methodology_7744547.zip"
INCLUDE = ["main.tex", "refs.bib", "numbers.tex", "README.md"]
DIRS = ["sections", "figures"]


def build(out: Path = OUT) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for name in INCLUDE:
            z.write(PAPER / name, name)
        for d in DIRS:
            for p in sorted((PAPER / d).iterdir()):
                if p.is_file() and p.suffix in (".tex", ".pdf", ".png"):
                    z.write(p, f"{d}/{p.name}")
    return out


def listing(path: Path) -> list[str]:
    with zipfile.ZipFile(path) as z:
        return z.namelist()


def main() -> int:
    out = build()
    print(f"wrote {out.relative_to(ROOT)} ({out.stat().st_size / 2**20:.1f} MB, {len(listing(out))} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
