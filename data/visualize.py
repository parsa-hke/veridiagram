"""Draw a Diagram's annotated primitives over its image, for visual checks (plan §2.4)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

from data.schema import Diagram  # noqa: E402


def draw(ax, d: Diagram, raw_dir: str | Path, title: bool = True) -> None:
    with Image.open(Path(raw_dir) / d.image_path) as im:
        ax.imshow(im.convert("RGB"))
    for a, b in d.segments:
        (x1, y1), (x2, y2) = d.points[a], d.points[b]
        ax.plot([x1, x2], [y1, y2], color="#00a0ff", lw=1.2, alpha=0.8)
    for c in d.circles:
        if c.center_xy and c.radius:
            ax.add_patch(plt.Circle(c.center_xy, c.radius, fill=False, color="#00c060", lw=1.2))
    for name, (x, y) in d.points.items():
        ax.plot(x, y, "o", ms=4, color="red")
        ax.annotate(name, (x, y), xytext=(4, -4), textcoords="offset points", color="red", fontsize=8)
    ax.set_xlim(0, d.width)
    ax.set_ylim(d.height, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        kinds = {}
        for r in d.asserted:
            kinds[r.type] = kinds.get(r.type, 0) + 1
        ax.set_title(f"{d.id} ({d.split})\n" + ", ".join(f"{k}:{v}" for k, v in sorted(kinds.items())), fontsize=7)


def save_grid(diagrams: list[Diagram], raw_dir: str | Path, out_path: str | Path) -> Path:
    n = len(diagrams)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5), squeeze=False)
    for ax, d in zip(axes[0], diagrams):
        draw(ax, d, raw_dir)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path
