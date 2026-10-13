"""Part B: self-training on PGPS9K with a confidence or consistency filter (plan §7.11).

Dispatched by experiments.run (mode: selftrain). One run = one student:

1. Teacher = the best 7.3 model (`selftrain.teacher_run`, default finetune_synth10k_s2: highest val
   macro-F1 of the three seeds), loaded from <runs>/<teacher_run>/checkpoints/best.pt (or
   `selftrain.teacher_checkpoint`). It predicts every query of the unlabelled pool, the unit
   `pgps9k_pool` (data.queries.POOL_UNITS: PGPS9K pool figures not shared with the training split).
   The predictions are cached in <runs>/selftrain_teacher/<tag>/pool_predictions.npz and shared by all
   students (checked by query ids).
2. Pseudo-labels. A (query, relation) entry is confident when max(p, 1 - p) > `selftrain.tau`; its
   pseudo-label is p > 0.5. Filters (`selftrain.filter`):
     confidence   keep the confident entries;
     consistency  the same, then drop every diagram whose confident relation set has a contradiction
                  after deductive closure (verifier.consistency: parallel/perpendicular on carrier
                  lines, equal_length on segments; with `selftrain.consistency_negatives`, confident
                  negatives count as facts predicted false, closure rule C2);
     none         no pseudo-labels (control: the same continued fine-tune on the labelled data only).
   A query with no confident entry is dropped; non-confident entries are masked out of the loss.
   All-negative pseudo-labelled queries are downsampled to `data.train_none_ratio` x the positive
   ones per kind, as for the labelled training set (plan §5.4).
3. Student = the teacher's weights (`finetune.init_checkpoint`, no LP stage), fine-tuned with the same
   budget as every run (2 x 40k images) on Geometry3K train + the pseudo-labelled queries, 50/50 per
   batch (`finetune.real_fraction`), then evaluated on every unit like any fine-tune.
4. Post-hoc (plan §7.11): the pool's sealed verifier labels are compared with the pseudo-labels of
   BOTH filters (and with the entries the consistency filter removed): precision of pseudo-positives
   and accuracy of pseudo-negatives per relation, in metrics.json under `pseudo_labels`. The sealed
   labels never reach the student (its rows carry the pseudo-labels in `geo`). The same closure on
   the verifier labels themselves gives the rules' false-alarm rate (`truth_contradiction_rate`).
"""

from __future__ import annotations

import copy
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from data.dataset import QueryDataset, loader, replace_or_discard
from data.queries import Carriers, downsample_none, vocabulary
from experiments.run import REPO_ROOT
from models import finetune
from models.backbone import Backbone, load_backbone
from models.common import device_of, predict
from verifier.consistency import closure

POOL = "pgps9k_pool"
LINE_RELS = ("parallel", "perpendicular")


def teacher_checkpoint(cfg: dict, runs_root: Path) -> Path:
    st = cfg["selftrain"]
    p = Path(st["teacher_checkpoint"]) if st.get("teacher_checkpoint") else \
        runs_root / st["teacher_run"] / "checkpoints" / "best.pt"
    if not p.is_absolute():
        p = REPO_ROOT / p
    if not p.exists():
        raise FileNotFoundError(f"teacher checkpoint not found: {p}")
    return p


def teacher_predictions(cfg: dict, ckpt: Path, backbone: Backbone, device: torch.device, cache: Path) -> dict:
    """Teacher probabilities for every pool query, cached (and checked by query ids)."""
    ds = QueryDataset(cfg, POOL, train=False)
    qids = np.array([r["qid"] for r in ds.rows])
    if cache.exists():
        z = dict(np.load(cache))
        if len(z["qid"]) == len(qids) and (z["qid"] == qids).all() and str(z["teacher"]) == str(ckpt):
            print(f"selftrain: teacher predictions from {cache}", flush=True)
            return {"rows": ds.rows, "probs": z["probs"], "diagrams": ds.diagrams}
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    if sd.get("backbone") and sd["backbone"] != backbone.name:
        raise ValueError(f"teacher was trained on {sd['backbone']}, loaded {backbone.name}")
    model = finetune.build_classifier(cfg, backbone, len(vocabulary(cfg)))
    model.load_state_dict(sd["model"])
    model.to(device).eval()
    dl = loader(ds, cfg["eval"]["batch_size"], shuffle=False, num_workers=cfg["data"]["num_workers"])
    pred = predict(model, dl, device, cfg)
    assert (pred["qid"] == qids).all()
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_name(f".{cache.stem}.{os.getpid()}.npz")
    np.savez_compressed(tmp, qid=qids, probs=pred["probs"], teacher=str(ckpt))
    replace_or_discard(tmp, cache)
    del model
    return {"rows": ds.rows, "probs": pred["probs"], "diagrams": ds.diagrams}


def query_facts(d, row: dict, vocab: list[str], labels: np.ndarray, use: np.ndarray, car: Carriers):
    """(positives, negatives) facts of one query for the closure, from labels where `use`."""
    pos, neg = [], []
    for j, rel in enumerate(vocab):
        if not use[j]:
            continue
        refs = row["refs"]
        if rel in LINE_RELS:
            a, b = car.of(*refs[0][1:]), car.of(*refs[1][1:])
        elif rel == "equal_length":
            a, b = "-".join(sorted(refs[0][1:])), "-".join(sorted(refs[1][1:]))
        else:
            a, b = json.dumps(refs[0]), json.dumps(refs[1:])
        (pos if labels[j] else neg).append((rel, a, b))
    return pos, neg


def contradictory_diagrams(rows: list[dict], diagrams: dict, vocab: list[str], labels: np.ndarray,
                           use: np.ndarray, negatives: bool = False) -> tuple[set, int]:
    """Diagram ids whose relation set (labels where `use`) has a contradiction after closure, and the
    number of diagrams checked. With `negatives`, entries labelled false are facts predicted false
    (closure rule C2), otherwise only the plan's contradiction (parallel and perpendicular) counts."""
    by_d = defaultdict(list)
    for i, r in enumerate(rows):
        by_d[r["diagram_id"]].append(i)
    bad = set()
    for did, idx in by_d.items():
        d = diagrams[did]
        car = Carriers(d)
        pos, neg = [], []
        for i in idx:
            p, n = query_facts(d, rows[i], vocab, labels[i], use[i], car)
            pos += p
            neg += n
        contradictions, _ = closure(pos, neg if negatives else ())
        if contradictions:
            bad.add(did)
    return bad, len(by_d)


def label_stats(true: np.ndarray, pseudo: np.ndarray, use: np.ndarray, vocab: list[str]) -> dict:
    """Pseudo-label quality against the sealed verifier labels, over the entries in `use`."""
    out = {"entries": int(use.sum())}
    for j, rel in enumerate(vocab):
        u = use[:, j]
        p, t = pseudo[u, j].astype(bool), true[u, j].astype(bool)
        out[rel] = {"pos": int(p.sum()), "pos_precision": round(float(t[p].mean()), 4) if p.any() else None,
                    "neg": int((~p).sum()), "neg_accuracy": round(float((~t[~p]).mean()), 4) if (~p).any() else None}
    pos = pseudo.astype(bool) & use
    out["pos"] = int(pos.sum())
    out["pos_precision"] = round(float(true[pos].astype(bool).mean()), 4) if pos.any() else None
    out["accuracy"] = round(float((pseudo[use] == true[use]).mean()), 4) if use.any() else None
    return out


def pseudo_label(cfg: dict, rows: list[dict], probs: np.ndarray, diagrams: dict) -> tuple[list[dict], dict]:
    """Pseudo-labelled training rows for `selftrain.filter`, and the post-hoc report for every filter."""
    st = cfg["selftrain"]
    vocab = vocabulary(cfg)
    tau = float(st["tau"])
    mask = np.array([r["mask"] for r in rows], dtype=bool)
    true = np.array([r["geo"] for r in rows], dtype=int)          # sealed: post-hoc report only
    pseudo = (probs > 0.5).astype(int)
    conf = mask & (np.maximum(probs, 1 - probs) > tau)
    neg = bool(st.get("consistency_negatives", False))
    bad, n_diag = contradictory_diagrams(rows, diagrams, vocab, pseudo, conf, neg)
    bad_truth, _ = contradictory_diagrams(rows, diagrams, vocab, true, mask, neg)
    in_bad = np.array([r["diagram_id"] in bad for r in rows], dtype=bool)
    keep = {"confidence": conf, "consistency": conf & ~in_bad[:, None]}
    report = {"tau": tau, "consistency_negatives": neg, "pool_queries": len(rows), "pool_diagrams": n_diag,
              "contradiction_diagrams": len(bad), "contradiction_rate": round(len(bad) / max(n_diag, 1), 4),
              "truth_contradiction_rate": round(len(bad_truth) / max(n_diag, 1), 4),
              "confidence": label_stats(true, pseudo, keep["confidence"], vocab),
              "consistency": label_stats(true, pseudo, keep["consistency"], vocab),
              "removed_by_consistency": label_stats(true, pseudo, conf & in_bad[:, None], vocab)}
    flt = st["filter"]
    if flt == "none":
        return [], report
    use = keep[flt]
    out = []
    for i, r in enumerate(rows):
        if not use[i].any():
            continue
        out.append({**{k: r[k] for k in ("qid", "diagram_id", "set", "split", "kind", "refs")},
                    "geo": (pseudo[i] * use[i]).tolist(), "asr": [0] * len(vocab),
                    "mask": use[i].astype(int).tolist(), "margins": [None] * len(vocab), "pseudo": True})
    rng = np.random.default_rng([cfg["data"]["index_seed"], 9])
    out = downsample_none(out, cfg["data"]["train_none_ratio"], rng)
    report["train_rows"] = len(out)
    report["train_diagrams"] = len({r["diagram_id"] for r in out})
    return out, report


def run(cfg: dict, run_dir: Path, backbone: Backbone | None = None) -> dict:
    st = cfg["selftrain"]
    if st["filter"] not in ("confidence", "consistency", "none"):
        raise ValueError(f"selftrain.filter must be confidence | consistency | none, got {st['filter']!r}")
    if int(st.get("rounds", 1)) != 1:
        raise NotImplementedError("one round of self-training (plan §7.11)")
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    runs_root = run_dir.parent
    device = device_of(cfg)
    ckpt = teacher_checkpoint(cfg, runs_root)
    backbone = backbone or load_backbone(cfg)
    tag = ckpt.parent.parent.name + ("_smoke" if cfg["name"].startswith("smoke") else "")
    tp = teacher_predictions(cfg, ckpt, backbone, device, runs_root / "selftrain_teacher" / tag / "pool_predictions.npz")
    rows, report = pseudo_label(cfg, tp["rows"], tp["probs"], tp["diagrams"])
    print(f"selftrain: filter {st['filter']}, tau {st['tau']}: {len(rows)} pseudo-labelled queries from "
          f"{report['pool_queries']} pool queries; teacher relation sets with a contradiction "
          f"{report['contradiction_diagrams']}/{report['pool_diagrams']} diagrams; pseudo-positive precision "
          f"confidence {report['confidence']['pos_precision']}, consistency {report['consistency']['pos_precision']}",
          flush=True)

    student = copy.deepcopy(cfg)
    student["finetune"]["init_checkpoint"] = str(ckpt)
    if rows:
        path = run_dir / "pseudo_index.jsonl"
        tmp = path.with_name(f".{path.name}.{os.getpid()}")
        with open(tmp, "w") as f:
            for r in rows:
                f.write(json.dumps(r, separators=(",", ":")) + "\n")
        replace_or_discard(tmp, path)
        student["data"]["extra_train_index"] = str(path)
    out = finetune.run(student, run_dir, backbone=backbone)
    return {**out, "teacher": str(ckpt), "filter": st["filter"], "pseudo_labels": report}
