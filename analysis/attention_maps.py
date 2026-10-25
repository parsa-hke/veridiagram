"""Phase 8.5: where does the model look? Attention rollout and last-layer CLS attention for 10 queries
(5 correct, 5 wrong) of a fine-tuned checkpoint, plus attention statistics over a sample (plan §8.5).

    python -m analysis.attention_maps [--run finetune_synth10k_s2] [--unit geometry3k_test] [--sample 60]

The checkpoint `runs/<run>/checkpoints/best.pt` is loaded with eager attention (the only
implementation that returns attention weights); every query is fed exactly as the square-padded
evaluation fed it (`runs/sq_<run>/predictions`, long_side x long_side), so the probabilities match
that evaluation. Decisions use the run's val-tuned thresholds (8.1).

Maps. Last layer: the CLS row of the head-averaged attention. Rollout (Abnar & Zeldes, 2020): per
layer A = 0.5 (mean over heads) + 0.5 I, rows renormalised, multiplied through the layers; the CLS
row of the product. Both are renormalised over the patch tokens (DINOv3 puts about half of the CLS
attention on its four register tokens) and reshaped to the (L/16, L/16) patch grid.

Statistics. Each patch of the model's square view is one of: highlight (the overlay touched it;
`data.overlay.draw_overlay`'s mask), ink (dark pixels of the diagram without highlight), blank
(white inside the image) or padding (outside the image). The share of attention on each class is
compared with the class's share of the patches; ratio > 1 means the model attends there more than
uniformly. The question answered: does the model attend to the highlighted primitives and the
diagram's marks, or elsewhere?

Selection. Correct: the most confident right decisions on positive labels, one relation at a time
(round robin), so the maps show where the model looks when it asserts a relation. Wrong: the most
confident wrong decisions, round robin over relations.

Writes
  reports/figures/attention_<correct|wrong>_<k>_<relation>.png   the view, last-layer map, rollout
  reports/figures/attention_grid.png                              the ten last-layer maps on one page
  reports/tables/attention_stats.csv                              per query (10 shown + sample): shares, ratios
Predictions are matched to queries by position (row i of the npz is row i of the unit); a qid alone is
not unique across diagrams (it hashes the query's kind and point names).
  reports/tables/attention_summary.csv                            mean shares and ratios per class and map
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from analysis.phase7 import write_csv
from analysis.phase8 import run_thresholds
from data.dataset import QueryDataset, fit_long_side, pad_batch
from data.overlay import draw_overlay, flatten
from data.queries import vocabulary
from experiments.run import REPO_ROOT, get_device, load_config

CLASSES = ("highlight", "ink", "blank", "padding")
INK_THRESHOLD = 128


# ------------------------------------------------------------------------------------------ model

def load_model(cfg: dict, checkpoint: Path, device: torch.device):
    """The classifier of a fine-tune checkpoint with eager attention (no pretrained download)."""
    from transformers import AutoConfig, AutoModel

    from models.backbone import Backbone
    from models.finetune import build_classifier
    sd = torch.load(checkpoint, map_location="cpu", weights_only=False)
    name = sd.get("backbone") or cfg["model"]["backbone"]
    hf = AutoModel.from_config(AutoConfig.from_pretrained(name), attn_implementation="eager")
    model = build_classifier(cfg, Backbone(hf, name), len(vocabulary(cfg)))
    model.load_state_dict(sd["model"])
    return model.float().to(device).eval(), name


@torch.no_grad()
def forward_with_attention(model, x: torch.Tensor) -> tuple[np.ndarray, list[torch.Tensor]]:
    """probs (R,) and the per-layer attention tensors (heads, T, T) of one square-padded image."""
    out = model.backbone(x, output_attentions=True)
    probs = torch.sigmoid(model.head(out["pooled"]).float())[0].cpu().numpy()
    return probs, [a[0].float().cpu() for a in out["attentions"]]


# ------------------------------------------------------------------------------------------- maps

def rollout(attentions: list[torch.Tensor]) -> np.ndarray:
    """Attention rollout: CLS row of prod_l norm(0.5 mean_heads(A_l) + 0.5 I)."""
    T = attentions[0].shape[-1]
    joint = torch.eye(T)
    for a in attentions:
        m = 0.5 * a.mean(0) + 0.5 * torch.eye(T)
        m = m / m.sum(-1, keepdim=True)
        joint = m @ joint
    return joint[0].numpy()


def last_layer(attentions: list[torch.Tensor]) -> np.ndarray:
    return attentions[-1].mean(0)[0].numpy()


def patch_map(cls_row: np.ndarray, prefix: int, grid: tuple[int, int]) -> np.ndarray:
    """Attention of CLS on the patch tokens, renormalised to sum 1, as a (h, w) grid."""
    v = cls_row[prefix:prefix + grid[0] * grid[1]].astype(float)
    return (v / max(v.sum(), 1e-12)).reshape(grid)


# ------------------------------------------------------------------------------------ patch classes

def model_view(ds: QueryDataset, row: dict, ocfg: dict) -> tuple[Image.Image, np.ndarray]:
    """The overlaid image at the model's resolution and the highlight pixel mask (same drawing code as
    the dataset, which only keeps the image)."""
    d = ds.diagrams[row["diagram_id"]]
    with Image.open(ds.raw / d.image_path) as im:
        img, k = fit_long_side(flatten(im), ds.L)
    return draw_overlay(img, d, row["refs"], ocfg, scale=k)


def patch_classes(img: Image.Image, highlight: np.ndarray, L: int, patch: int) -> np.ndarray:
    """(L/patch, L/patch) array of class indices into CLASSES for the square view."""
    g = L // patch
    cls = np.full((g, g), CLASSES.index("padding"))
    gray = np.asarray(img.convert("L"))
    ink = gray < INK_THRESHOLD
    h, w = highlight.shape
    for i in range(g):
        for j in range(g):
            y0, x0 = i * patch, j * patch
            if y0 >= h or x0 >= w:
                continue
            hl = highlight[y0:y0 + patch, x0:x0 + patch]
            ik = ink[y0:y0 + patch, x0:x0 + patch]
            cls[i, j] = CLASSES.index("highlight") if hl.any() else CLASSES.index("ink") if ik.any() else CLASSES.index("blank")
    return cls


def attention_shares(amap: np.ndarray, classes: np.ndarray) -> dict[str, float]:
    """share_<class> (attention mass), area_<class> (share of patches), ratio_<class>."""
    out = {}
    for k, name in enumerate(CLASSES):
        sel = classes == k
        area = float(sel.mean())
        share = float(amap[sel].sum())
        out[f"share_{name}"] = round(share, 4)
        out[f"area_{name}"] = round(area, 4)
        out[f"ratio_{name}"] = round(share / area, 3) if area > 0 else None
    return out


# -------------------------------------------------------------------------------------- selection

def select_queries(preds: dict, thresholds: np.ndarray, vocab: list[str], n_correct: int = 5, n_wrong: int = 5
                   ) -> list[dict]:
    """5 + 5 (query, relation) decisions: confident right positives and confident wrong decisions,
    round robin over relations, deterministic."""
    p, y, m = preds["probs"], preds["geo"] > 0, preds["mask"] > 0
    pred = p >= thresholds
    right = (pred == y) & m
    wrong = (pred != y) & m
    conf = np.abs(p - thresholds)

    used: set[int] = set()                                                 # no query twice across both kinds

    def pick(cands: np.ndarray, n: int, score: np.ndarray) -> list[tuple[int, int]]:
        per_rel = []
        for j in range(len(vocab)):
            idx = np.flatnonzero(cands[:, j])
            per_rel.append(list(idx[np.argsort(-score[idx, j], kind="stable")]))
        chosen = []
        while len(chosen) < n and any(per_rel):
            for j in range(len(vocab)):
                while per_rel[j]:
                    i = per_rel[j].pop(0)
                    if i not in used:
                        chosen.append((int(i), j))
                        used.add(int(i))
                        break
                if len(chosen) >= n:
                    break
        return chosen

    out = []
    for kind, cands, n in (("correct", right & y, n_correct), ("wrong", wrong, n_wrong)):
        for i, j in pick(cands, n, conf):
            out.append({"kind": kind, "index": i, "qid": str(preds["qid"][i]), "relation": vocab[j], "j": j,
                        "label": int(y[i, j]), "prob": round(float(p[i, j]), 4), "threshold": float(thresholds[j])})
    return out


# ---------------------------------------------------------------------------------------- figures

def _heat(ax, img: Image.Image, amap: np.ndarray, L: int, title: str):
    ax.imshow(img)
    ax.imshow(np.kron(amap, np.ones((L // amap.shape[0], L // amap.shape[1]))), cmap="inferno", alpha=0.55,
              vmin=0, vmax=float(np.percentile(amap, 99.5)))
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def query_figure(view: Image.Image, maps: dict[str, np.ndarray], L: int, title: str, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    square = Image.new("RGB", (L, L), (255, 255, 255))
    square.paste(view, (0, 0))
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.9))
    axes[0].imshow(square)
    axes[0].set_title("the model's view", fontsize=8)
    axes[0].axis("off")
    _heat(axes[1], square, maps["last_layer"], L, "last-layer CLS attention")
    _heat(axes[2], square, maps["rollout"], L, "attention rollout")
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)


def grid_figure(items: list[dict], L: int, out: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = max(sum(1 for it in items if it["kind"] == k) for k in ("correct", "wrong")) or 1
    fig, axes = plt.subplots(2, n, figsize=(2.6 * n, 5.8), squeeze=False)
    for r, kind in enumerate(("correct", "wrong")):
        row = [it for it in items if it["kind"] == kind]
        for c in range(n):
            ax = axes[r][c]
            if c < len(row):
                it = row[c]
                square = Image.new("RGB", (L, L), (255, 255, 255))
                square.paste(it["view"], (0, 0))
                _heat(ax, square, it["maps"]["last_layer"], L,
                      f"{kind}: {it['relation']} y={it['label']} p={it['prob']:.2f}\non highlight {it['stats_last']['share_highlight']:.0%}")
            else:
                ax.axis("off")
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


# ------------------------------------------------------------------------------------------- main

def analyse_query(model, ds: QueryDataset, row: dict, cfg: dict, device: torch.device) -> dict:
    L, patch = ds.L, model.backbone.patch_size
    view, highlight = model_view(ds, row, cfg["overlay"])
    x = pad_batch([ds.to_tensor(view)], ds.white(), square=L).to(device)
    probs, att = forward_with_attention(model, x)
    grid = (L // patch, L // patch)
    maps = {"last_layer": patch_map(last_layer(att), model.backbone.num_prefix, grid),
            "rollout": patch_map(rollout(att), model.backbone.num_prefix, grid)}
    classes = patch_classes(view, highlight, L, patch)
    return {"view": view, "probs": probs, "maps": maps, "classes": classes,
            "stats_last": attention_shares(maps["last_layer"], classes),
            "stats_rollout": attention_shares(maps["rollout"], classes)}


def summarise(stats: list[dict]) -> list[dict]:
    out = []
    for m in ("last_layer", "rollout"):
        for kind in ("all", "correct", "wrong", "sample"):
            rows = [s for s in stats if s["map"] == m and (kind == "all" or s["kind"] == kind)]
            if not rows:
                continue
            rec = {"map": m, "kind": kind, "n": len(rows)}
            for c in CLASSES:
                rec[f"share_{c}"] = round(float(np.mean([r[f"share_{c}"] for r in rows])), 4)
                rec[f"area_{c}"] = round(float(np.mean([r[f"area_{c}"] for r in rows])), 4)
                ratios = [r[f"ratio_{c}"] for r in rows if r[f"ratio_{c}"] is not None]
                rec[f"ratio_{c}"] = round(float(np.mean(ratios)), 3) if ratios else None
            out.append(rec)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="finetune_synth10k_s2", help="run with checkpoints/best.pt and a sq_<run> evaluation")
    ap.add_argument("--unit", default="geometry3k_test")
    ap.add_argument("--sample", type=int, default=60, help="extra random queries for the statistics")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    cfg = load_config(REPO_ROOT / "configs" / "base.yaml")
    runs = REPO_ROOT / cfg["paths"]["runs"]
    figures, tables = REPO_ROOT / cfg["paths"]["figures"], REPO_ROOT / cfg["paths"]["tables"]
    vocab = vocabulary(cfg)
    device = get_device(args.device)
    sq = f"sq_{args.run}"
    z = np.load(runs / sq / "predictions" / f"{args.unit}.npz")
    preds = {k: z[k] for k in z.files}
    thresholds = run_thresholds(runs, sq, vocab)
    model, name = load_model(cfg, runs / args.run / "checkpoints" / "best.pt", device)
    ds = QueryDataset(cfg, args.unit, train=False)
    # predictions are positional: row i of the npz is ds.rows[i]. A qid alone is not a key (it hashes the
    # query's kind and point names, which recur across diagrams), so the check is the whole sequence.
    if len(ds.rows) != len(preds["qid"]) or any(r["qid"] != q for r, q in zip(ds.rows, preds["qid"])):
        raise SystemExit(f"{sq}/{args.unit}: predictions do not match the current evaluation rows")
    chosen = select_queries(preds, thresholds, vocab)
    rng = np.random.default_rng(args.seed)
    extra = [int(i) for i in rng.choice(len(preds["qid"]), size=min(args.sample, len(preds["qid"])), replace=False)]
    stats, items = [], []
    for k, it in enumerate(chosen):
        row = ds.rows[it["index"]]
        a = analyse_query(model, ds, row, cfg, device)
        p_eval = float(preds["probs"][it["index"], it["j"]])
        if abs(float(a["probs"][it["j"]]) - p_eval) > 0.05:
            print(f"WARNING: {it['qid']} p={a['probs'][it['j']]:.3f} here vs {p_eval:.3f} in {sq}")
        n = sum(1 for c in chosen[:k] if c["kind"] == it["kind"]) + 1
        refs = " ".join(":".join(str(x) for x in ref) for ref in row["refs"])
        title = (f"{it['kind']} {n}: {it['relation']} on {row['diagram_id']} ({refs}); label {it['label']}, "
                 f"p {it['prob']:.2f} (threshold {it['threshold']:.2f})\n"
                 f"CLS attention on the highlight: last layer {a['stats_last']['share_highlight']:.0%}, "
                 f"rollout {a['stats_rollout']['share_highlight']:.0%}")
        query_figure(a["view"], a["maps"], ds.L, title, figures / f"attention_{it['kind']}_{n}_{it['relation']}.png")
        items.append({**it, **a})
        for m, s in (("last_layer", a["stats_last"]), ("rollout", a["stats_rollout"])):
            stats.append({"map": m, "kind": it["kind"], "qid": it["qid"], "diagram_id": row["diagram_id"], "relation": it["relation"],
                          "label": it["label"], "prob": it["prob"], **s})
    for i in extra:
        row = ds.rows[i]
        a = analyse_query(model, ds, row, cfg, device)
        for m, s in (("last_layer", a["stats_last"]), ("rollout", a["stats_rollout"])):
            stats.append({"map": m, "kind": "sample", "qid": row["qid"], "diagram_id": row["diagram_id"], "relation": row["kind"],
                          "label": None, "prob": None, **s})
    grid_figure(items, ds.L, figures / "attention_grid.png",
                f"{args.run} on {args.unit}: last-layer CLS attention of five correct and five wrong decisions")
    write_csv(stats, tables / "attention_stats.csv")
    summary = summarise(stats)
    write_csv(summary, tables / "attention_summary.csv")
    print(f"{name}, {args.run}, {args.unit}: {len(chosen)} shown + {len(extra)} sampled")
    for r in summary:
        print(f"  {r['map']:10s} {r['kind']:8s} n={r['n']:3d}  " + "  ".join(
            f"{c}: {r[f'share_{c}']:.2f} of attention on {r[f'area_{c}']:.2f} of patches (x{r[f'ratio_{c}'] or 0:.1f})" for c in CLASSES))
    print(f"-> {figures}/attention_*.png, {tables}/attention_stats.csv, attention_summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
