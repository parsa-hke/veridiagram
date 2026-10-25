"""Copy the technical report's figures from reports/figures/ into reports/report/figures/.

Nothing is drawn here: every figure is produced by its analysis module (the manifest
reports/figures/README.md lists the regenerating command). This script only makes the Overleaf
project self-contained.

    python -m scripts.report_figures
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "reports" / "figures"
OUT = ROOT / "reports" / "report" / "figures"
FIGURES = [
    "audit_examples_model.png",   # title page and audit section: the clearest label disagreements
    "headline_table.png",         # results: queries solved by probe / fine-tune / both / neither
    "per_relation_f1.png",        # results: per-relation F1 with CIs, every experiment and test set
    "attention_grid.png",         # results: last-layer CLS attention on ten decisions
    "error_margin.png",           # results: accuracy against the verifier margin
    "scaling_square.png",         # results: Part A scaling curve
]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for name in FIGURES:
        shutil.copyfile(SRC / name, OUT / name)
    print("copied", ", ".join(FIGURES))
    return 0


if __name__ == "__main__":
    sys.exit(main())
