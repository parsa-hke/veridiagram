"""Gold set for the verifier (plan §3.5).

    python -m verifier.gold sample --round 2   # -> verifier/gold/gold_r2_queries.csv + label_r2.html
    python -m verifier.gold score              # gold.csv (+ gold_r2.csv) -> reports/tables/verifier_audit.csv

Round 1 (`--round 1`, files gold_queries.csv / gold.csv / label.html): 100 queries stratified by
relation (6) and source (geometry3k_figures, pgdp5k_heldout, pgps9k_pool); within each cell half
verifier-positive, half verifier-negative, about 2/3 near the tolerance (positives with margin in
[thr/4, thr], negatives in (thr, 4 thr]). It failed the gate and informed the tolerance revision
(plan §0.2 #5), so its queries are frozen (sampling depended on the tolerances at the time).

Round 2 (`--round 2`, files gold_r2_queries.csv / gold_r2.csv / label_r2.html): 60 new queries
(10 per relation, round-1 items excluded), about 1/3 near the tolerance, each with a sampling
weight = stratum size / sampled count (stratum = source x relation x verdict x near/far band);
plus a blind re-check of every round-1 item the revised verifier disagrees with, mixed with as
many random round-1 agreements. Items get neutral ids in random order.

Pre-registered gate (plan §3.5): population-weighted FP and FN rates on the 60 new round-2 queries,
both <= 3%. Label corrections the labeller confirmed on re-inspection live in
verifier/gold/adjudication.csv (raw label files are never edited); raw and adjudicated rates are
both reported. Reported alongside: unweighted (stress) rates for both rounds and the re-check table.
Line relations are sampled once per pair of carrier lines. The labelling page shows the diagram
and the queried primitives only: no verdict, margin or annotation. `score` re-runs the checks with
the current tolerances. "unsure" labels are excluded from rates.
  FP rate = verifier yes & gold no / gold no      FN rate = verifier no & gold yes / gold yes
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from data.queries import enumerate_queries
from data.schema import read_jsonl
from experiments.run import REPO_ROOT, load_config
from verifier.audit import grouped, margin_threshold
from verifier.checks import evaluate

GOLD_DIR = REPO_ROOT / "verifier" / "gold"
SETS = ("geometry3k_figures", "pgdp5k_heldout", "pgps9k_pool")
ALLOC = {"parallel": 17, "perpendicular": 17, "equal_length": 18, "tangent": 18, "on_circle": 15, "collinear": 15}
QUESTIONS = {
    "parallel": "Are the two highlighted segments parallel?",
    "perpendicular": "Are the two highlighted segments perpendicular?",
    "equal_length": "Are the two highlighted segments equal in length?",
    "on_circle": "Does the highlighted point lie on the highlighted circle?",
    "tangent": "Is the highlighted segment tangent to the highlighted circle? (Its line touches the circle at exactly one point without cutting through it, and that point lies on the segment or at its end.)",
    "collinear": "Does the circled point lie on the straight line through the two square-marked points (extended if needed)?",
}
CEILING = 0.03
FILES = {1: ("gold_queries.csv", "gold.csv", "label.html", "veridiagram-gold-v1"),
         2: ("gold_r2_queries.csv", "gold_r2.csv", "label_r2.html", "veridiagram-gold-v2")}
ALLOC_R2 = {rel: 10 for rel in ALLOC}
QUERY_FIELDS = ["gold_id", "set", "diagram_id", "relation", "refs", "stratum", "weight", "recheck_of"]


def _split(n: int, k: int) -> list[int]:
    return [n // k + (1 if i < n % k else 0) for i in range(k)]


def sample(cfg: dict, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    proc = REPO_ROOT / cfg["paths"]["processed"]
    tol = cfg["tolerances"]
    pools = {s: grouped(proc / "queries" / f"{s}.csv.gz", tol) for s in SETS}
    out = []
    for rel, n_rel in ALLOC.items():
        thr = margin_threshold(rel, tol)
        for src, n in zip(SETS, _split(n_rel, len(SETS))):
            recs = [r for r in pools[src] if r["relation"] == rel and not math.isnan(r["margin"])]
            pos = [r for r in recs if r["geometric"]]
            neg = [r for r in recs if not r["geometric"]]
            n_pos, n_neg = _split(n, 2)
            if rng.random() < 0.5:
                n_pos, n_neg = n_neg, n_pos
            picked = _pick(pos, n_pos, lambda r: thr / 4 <= r["margin"] <= thr, rng) + \
                _pick(neg, n_neg, lambda r: thr < r["margin"] <= 4 * thr, rng)
            short = n - len(picked)
            if short:                                      # e.g. too few positives: fill from the other side
                rest = [r for r in recs if r not in picked]
                picked += rng.sample(rest, min(short, len(rest)))
            for r in picked:
                out.append({"set": src, "diagram_id": r["diagram_id"], "relation": rel, "refs": r["refs"]})
    rng.shuffle(out)
    for i, r in enumerate(out):
        r["gold_id"] = f"g{i + 1:03d}"
    return out


def _pick(recs: list, n: int, near, rng: random.Random) -> list:
    """About 2/3 near the threshold, the rest anywhere."""
    near_r = [r for r in recs if near(r)]
    n_near = min(len(near_r), round(n * 2 / 3))
    picked = rng.sample(near_r, n_near)
    rest = [r for r in recs if r not in picked]
    return picked + rng.sample(rest, min(n - n_near, len(rest)))


def _near(rel_thr: float, positive: bool):
    return (lambda r: rel_thr / 4 <= r["margin"] <= rel_thr) if positive else (lambda r: rel_thr < r["margin"] <= 4 * rel_thr)


def _diagrams(cfg: dict, items) -> dict:
    proc = REPO_ROOT / cfg["paths"]["processed"]
    out = {}
    for s in SETS:
        need = {it["diagram_id"] for it in items if it["set"] == s}
        if need:
            out.update({(s, d.id): d for d in read_jsonl(proc / f"{s}.jsonl") if d.id in need})
    return out


def _read(name: str) -> dict[str, dict]:
    path = GOLD_DIR / name
    if not path.exists():
        return {}
    with open(path) as f:
        return {r["gold_id"]: r for r in csv.DictReader(f)}


def verdicts(cfg: dict, items) -> dict[str, str]:
    """Current verifier verdict ('yes'/'no') per gold item, or 'invalid' when the query is no longer
    a candidate (e.g. nested segments on one line, removed from the query space after round 2)."""
    D = _diagrams(cfg, items)
    tier2 = tuple(cfg["relations"].get("tier2_included", []))
    cands = {}
    out = {}
    for it in items:
        d = D[(it["set"], it["diagram_id"])]
        if d.id not in cands:
            cands[d.id] = {(q.key(), r) for q in enumerate_queries(d, tier2) for r in q.relations}
        refs = json.loads(it["refs"])
        if (json.dumps(refs, separators=(",", ":")), it["relation"]) not in cands[d.id]:
            out[it["gold_id"]] = "invalid"
            continue
        ok, _ = evaluate(d, it["relation"], refs, cfg["tolerances"])
        out[it["gold_id"]] = "yes" if ok else "no"
    return out


def sample_round2(cfg: dict, seed: int = 1) -> list[dict]:
    rng = random.Random(seed)
    proc = REPO_ROOT / cfg["paths"]["processed"]
    tol = cfg["tolerances"]
    r1q, r1l = _read(FILES[1][0]), _read(FILES[1][1])
    used = {(q["diagram_id"], q["refs"]) for q in r1q.values()}
    pools = {s: grouped(proc / "queries" / f"{s}.csv.gz", tol) for s in SETS}
    out = []
    for rel, n_rel in ALLOC_R2.items():
        thr = margin_threshold(rel, tol)
        for src, n in zip(SETS, _split(n_rel, len(SETS))):
            recs = [r for r in pools[src] if r["relation"] == rel and not math.isnan(r["margin"])
                    and (r["diagram_id"], r["refs"]) not in used]
            n_pos, n_neg = _split(n, 2)
            if rng.random() < 0.5:
                n_pos, n_neg = n_neg, n_pos
            for positive, n_v in ((True, n_pos), (False, n_neg)):
                side = [r for r in recs if r["geometric"] == positive]
                near_f = _near(thr, positive)
                bands = {"near": [r for r in side if near_f(r)], "far": [r for r in side if not near_f(r)]}
                want = {"near": min(len(bands["near"]), round(n_v / 3))}
                want["far"] = min(len(bands["far"]), n_v - want["near"])
                want["near"] = min(len(bands["near"]), n_v - want["far"])      # top up if "far" is short
                for band, k in want.items():
                    for r in rng.sample(bands[band], k):
                        out.append({"set": src, "diagram_id": r["diagram_id"], "relation": rel, "refs": r["refs"],
                                    "stratum": f"{src}|{rel}|{'pos' if positive else 'neg'}|{band}",
                                    "weight": f"{len(bands[band]) / k:.4f}", "recheck_of": ""})
    # blind re-check: every round-1 disagreement under the current tolerances + as many agreements
    v1 = verdicts(cfg, list(r1q.values()))
    lab = {g: r1l.get(g, {}).get("label", "") for g in r1q}
    dis = sorted(g for g in r1q if lab[g] in ("yes", "no") and lab[g] != v1[g])
    agr = sorted(g for g in r1q if lab[g] in ("yes", "no") and lab[g] == v1[g])
    for g in dis + rng.sample(agr, min(len(dis), len(agr))):
        q = r1q[g]
        out.append({"set": q["set"], "diagram_id": q["diagram_id"], "relation": q["relation"], "refs": q["refs"],
                    "stratum": "recheck", "weight": "", "recheck_of": g})
    rng.shuffle(out)
    for i, r in enumerate(out):
        r["gold_id"] = f"h{i + 1:03d}"
    return out


def _image_uri(path: Path) -> tuple[str, int, int]:
    with Image.open(path) as im:
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        rgb = Image.alpha_composite(bg, im).convert("RGB")
        buf = io.BytesIO()
        rgb.save(buf, "PNG", optimize=True)
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(), rgb.width, rgb.height


def build_page(items: list[dict], cfg: dict, out: Path, key: str = FILES[1][3], csv_name: str = FILES[1][1]) -> None:
    proc, raw = REPO_ROOT / cfg["paths"]["processed"], REPO_ROOT / cfg["paths"]["raw"]
    diagrams = {}
    for s in SETS:
        need = {it["diagram_id"] for it in items if it["set"] == s}
        diagrams.update({(s, d.id): d for d in read_jsonl(proc / f"{s}.jsonl") if d.id in need})
    payload = []
    for it in items:
        d = diagrams[(it["set"], it["diagram_id"])]
        uri, w, h = _image_uri(raw / d.image_path)
        refs = json.loads(it["refs"])
        prims = []
        for k, ref in enumerate(refs):
            if ref[0] == "seg":
                prims.append({"t": "seg", "a": d.points[ref[1]], "b": d.points[ref[2]], "na": ref[1], "nb": ref[2]})
            elif ref[0] == "pt":
                role = "target" if (it["relation"] != "collinear" or k == 0) else "anchor"
                prims.append({"t": "pt", "p": d.points[ref[1]], "n": ref[1], "role": role})
            elif ref[0] == "circle":
                c = d.circles[ref[1]]
                prims.append({"t": "circle", "c": c.center_xy, "r": c.radius})
        payload.append({"id": it["gold_id"], "rel": it["relation"], "q": QUESTIONS[it["relation"]],
                        "img": uri, "w": w, "h": h, "prims": prims})
    html = PAGE.replace("__DATA__", json.dumps(payload, separators=(",", ":"))).replace("__KEY__", key).replace("__CSV__", csv_name)
    out.write_text(html)


def _rows(tag: str, items: list[dict], labels: dict, verdict: dict, weighted: bool = False) -> list[dict]:
    per = defaultdict(lambda: defaultdict(float))
    for it in items:
        g = it["gold_id"]
        lab = labels.get(g, {}).get("label", "").strip().lower()
        w = float(it.get("weight") or 1.0) if weighted else 1.0
        for key in (it["relation"], "ALL"):
            c = per[key]
            c["n"] += 1
            if verdict[g] == "invalid":
                c["invalid"] += 1
                continue
            if lab not in ("yes", "no"):
                c["unsure" if lab == "unsure" else "unlabelled"] += 1
                continue
            c[f"v{verdict[g]}_g{lab}"] += w
            c[f"k_v{verdict[g]}_g{lab}"] += 1
    rows = []
    for key in list(ALLOC) + ["ALL"]:
        c = per[key]
        tp, fp, fn, tn = (c[f"v{a}_g{b}"] for a, b in (("yes", "yes"), ("yes", "no"), ("no", "yes"), ("no", "no")))
        k = {x: int(c[f"k_v{a}_g{b}"]) for x, (a, b) in zip(("tp", "fp", "fn", "tn"), (("yes", "yes"), ("yes", "no"), ("no", "yes"), ("no", "no")))}
        rows.append({"evaluation": tag, "relation": key, "n": int(c["n"]), "labelled": sum(k.values()),
                     "unsure": int(c["unsure"]), "unlabelled": int(c["unlabelled"]), "invalid": int(c["invalid"]), **k,
                     "fp_rate": _rate(fp, fp + tn), "fp_rate_ci95": "" if weighted else _wilson(k["fp"], k["fp"] + k["tn"]),
                     "fn_rate": _rate(fn, fn + tp), "fn_rate_ci95": "" if weighted else _wilson(k["fn"], k["fn"] + k["tp"]),
                     "err_among_verifier_yes": _rate(fp, tp + fp), "err_among_verifier_no": _rate(fn, fn + tn)})
    return rows


def adjudicated(labels: dict[str, dict]) -> dict[str, dict]:
    """Labels with the corrections in verifier/gold/adjudication.csv applied (raw files are never edited).
    Only corrections the labeller confirmed after seeing the item again are recorded there."""
    adj = _read("adjudication.csv")
    out = {g: dict(r) for g, r in labels.items()}
    for g, a in adj.items():
        if g in out:
            assert out[g]["label"] == a["raw_label"], f"adjudication for {g} does not match the raw label"
            out[g]["label"] = a["adjudicated_label"]
    return out


def score(cfg: dict) -> tuple[list[dict], list[dict]]:
    """Rows for verifier_audit.csv (each round raw and adjudicated; round 2 also weighted) and the re-check table."""
    r1q, r1l = _read(FILES[1][0]), _read(FILES[1][1])
    v1 = verdicts(cfg, list(r1q.values()))
    rows = _rows("round1", list(r1q.values()), r1l, v1) + _rows("round1_adjudicated", list(r1q.values()), adjudicated(r1l), v1)
    recheck = []
    r2q, r2l = _read(FILES[2][0]), _read(FILES[2][1])
    if r2q and r2l:
        new = [q for q in r2q.values() if not q["recheck_of"]]
        v2 = verdicts(cfg, list(r2q.values()))
        a2 = adjudicated(r2l)
        rows += (_rows("round2", new, r2l, v2) + _rows("round2_weighted", new, r2l, v2, weighted=True)
                 + _rows("round2_adjudicated", new, a2, v2) + _rows("round2_adjudicated_weighted", new, a2, v2, weighted=True))
        v1 = verdicts(cfg, list(r1q.values()))
        for q in r2q.values():
            if q["recheck_of"]:
                g1 = q["recheck_of"]
                l1, l2 = r1l[g1]["label"], r2l.get(q["gold_id"], {}).get("label", "")
                if v1[g1] == "invalid":
                    continue
                recheck.append({"round1_id": g1, "round2_id": q["gold_id"], "relation": q["relation"],
                                "verifier": v1[g1], "round1_label": l1, "round2_label": l2,
                                "group": "disagreement" if l1 != v1[g1] else "agreement",
                                "label_changed": int(l1 != l2), "now_agrees": int(l2 == v1[g1])})
    return rows, recheck


def _rate(k: float, n: float) -> str:
    return f"{k / n:.3f}" if n else ""


def _wilson(k: int, n: int, z: float = 1.96) -> str:
    if not n:
        return ""
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return f"[{max(0.0, centre - half):.3f}, {min(1.0, centre + half):.3f}]"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["sample", "score"])
    ap.add_argument("--round", type=int, choices=[1, 2], default=2)
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "base.yaml"))
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--force", action="store_true", help="overwrite an existing round's queries")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    if args.command == "sample":
        qfile, lfile, page, key = FILES[args.round]
        if (GOLD_DIR / qfile).exists() and not args.force:
            if not (GOLD_DIR / page).exists():
                items = list(_read(qfile).values())
                build_page(items, cfg, GOLD_DIR / page, key, lfile)
                print(f"rebuilt {page} from the frozen {qfile}")
                return 0
            raise SystemExit(f"{qfile} exists (frozen; labels refer to it). Use --force to resample.")
        if args.round == 1:
            items = sample(cfg, 0 if args.seed is None else args.seed)
            for it in items:
                it.update(stratum="", weight="", recheck_of="")
        else:
            items = sample_round2(cfg, 1 if args.seed is None else args.seed)
        items = sorted(items, key=lambda r: r["gold_id"])
        with open(GOLD_DIR / qfile, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=QUERY_FIELDS, extrasaction="ignore")
            w.writeheader()
            w.writerows(items)
        build_page(items, cfg, GOLD_DIR / page, key, lfile)
        new = [i for i in items if not i.get("recheck_of")]
        print(f"round {args.round}: {len(items)} items ({len(new)} new, {len(items) - len(new)} re-checks); "
              f"new by relation {dict(Counter(i['relation'] for i in new))}")
        print(f"-> {GOLD_DIR / qfile}, {GOLD_DIR / page}")
        return 0

    rows, recheck = score(cfg)
    tabs = REPO_ROOT / cfg["paths"]["tables"]
    with open(tabs / "verifier_audit.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    if recheck:
        with open(tabs / "verifier_recheck.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(recheck[0]))
            w.writeheader()
            w.writerows(recheck)
    for r in rows:
        if r["relation"] == "ALL":
            print({k: r[k] for k in ("evaluation", "labelled", "unsure", "invalid", "tp", "fp", "fn", "tn", "fp_rate", "fn_rate")})
    for tag in ("round2_weighted", "round2_adjudicated_weighted"):
        gate = next((r for r in rows if r["evaluation"] == tag and r["relation"] == "ALL"), None)
        if gate is None:
            print("GATE 3.5: pending round 2 labels (python -m verifier.gold sample --round 2)")
            break
        fp, fn = float(gate["fp_rate"] or 0), float(gate["fn_rate"] or 0)
        verdict = "PASS" if fp <= CEILING and fn <= CEILING else "FAIL"
        print(f"GATE 3.5 [{tag}]: FP {fp:.3f}, FN {fn:.3f}, ceiling {CEILING} -> {verdict}")
    if recheck:
        for grp in ("disagreement", "agreement"):
            R = [r for r in recheck if r["group"] == grp]
            print(f"re-check {grp}s: {len(R)}, label changed {sum(r['label_changed'] for r in R)}, now agree {sum(r['now_agrees'] for r in R)}")
    return 0


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Verifier Gold Labels</title>
<style>
:root{--bg:#fcfcfb;--panel:#ffffff;--ink:#1f1f1e;--muted:#6b6a64;--line:#e4e3dc;--accent:#2a78d6;--hl:#e8161b;--yes:#1b7f4b;--no:#b3261e;--unsure:#8a6d00}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#1a1a19;--panel:#232322;--ink:#f2f1ec;--muted:#b4b3ab;--line:#3a3a38;--accent:#3987e5}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
header{display:flex;flex-wrap:wrap;gap:12px;align-items:center;justify-content:space-between;padding:12px 16px;border-bottom:1px solid var(--line)}
h1{font-size:16px;margin:0}.muted{color:var(--muted)}main{max-width:1100px;margin:0 auto;padding:16px}
.bar{height:6px;background:var(--line);border-radius:3px;overflow:hidden;min-width:160px}.bar>div{height:100%;background:var(--accent);width:0}
.q{font-size:17px;font-weight:600;margin:4px 0 12px}.stage{position:relative;background:#fff;border:1px solid var(--line);border-radius:8px;overflow:auto;max-height:68vh}
.stage img,.stage svg{display:block}.stage svg{position:absolute;left:0;top:0;pointer-events:none}
.controls{display:flex;flex-wrap:wrap;gap:8px;margin:12px 0}button{font:inherit;padding:8px 14px;border-radius:8px;border:1px solid var(--line);background:var(--panel);color:var(--ink);cursor:pointer}
button.lab{font-weight:600}button.lab.on[data-v=yes]{background:var(--yes);color:#fff;border-color:var(--yes)}button.lab.on[data-v=no]{background:var(--no);color:#fff;border-color:var(--no)}button.lab.on[data-v=unsure]{background:var(--unsure);color:#fff;border-color:var(--unsure)}
textarea{width:100%;min-height:52px;font:inherit;padding:8px;border-radius:8px;border:1px solid var(--line);background:var(--panel);color:var(--ink)}
.help{font-size:13px;color:var(--muted);margin-top:10px}kbd{border:1px solid var(--line);border-radius:4px;padding:0 4px;font-size:12px}
</style></head><body>
<header><div><h1>Verifier gold set</h1><div class="muted" id="pos"></div></div>
<div style="display:flex;gap:10px;align-items:center"><div class="bar"><div id="prog"></div></div><span class="muted" id="done"></span>
<button id="export">Export __CSV__</button></div></header>
<main>
<div class="q" id="question"></div>
<div class="controls">
<button class="lab" data-v="yes">Yes <kbd>1</kbd></button><button class="lab" data-v="no">No <kbd>2</kbd></button><button class="lab" data-v="unsure">Unsure <kbd>3</kbd></button>
<span style="flex:1"></span><button id="toggle">Hide overlay <kbd>H</kbd></button><button id="ext">Extend lines <kbd>E</kbd></button><button id="zoom">Zoom fit <kbd>Z</kbd></button>
<button id="prev">&larr; Prev</button><button id="next">Next &rarr;</button></div>
<div class="stage" id="stage"><img id="img" alt="diagram"><svg id="svg"></svg></div>
<p class="help">Judge the <b>drawing as drawn</b>: ignore tick marks, right-angle symbols, arrows and numbers. Answer <b>Unsure</b> when you cannot tell by eye. Toggle the overlay to see the original lines. Labels autosave in this browser; click <b>Export gold.csv</b> when done and save it as <code>verifier/gold/__CSV__</code>.</p>
<textarea id="note" placeholder="Optional note (what you saw, why unsure)"></textarea>
</main>
<script>
const DATA=__DATA__;const KEY="__KEY__";let st={};try{st=JSON.parse(localStorage.getItem(KEY)||"{}")}catch(e){st={}}
let i=0,show=true,ext=false,zoom="fit";const $=s=>document.querySelector(s);
function save(){try{localStorage.setItem(KEY,JSON.stringify(st))}catch(e){}}
function first(){const k=DATA.findIndex(d=>!(st[d.id]&&st[d.id].label));return k<0?0:k}
function el(n,a){const e=document.createElementNS("http://www.w3.org/2000/svg",n);for(const k in a)e.setAttribute(k,a[k]);return e}
function extend(a,b,w,h){const dx=b[0]-a[0],dy=b[1]-a[1],L=Math.hypot(dx,dy)||1,k=2*(w+h)/L;return[[a[0]-dx*k,a[1]-dy*k],[b[0]+dx*k,b[1]+dy*k]]}
function render(){const d=DATA[i],s=st[d.id]||{};$("#pos").textContent=`Item ${i+1} of ${DATA.length} · ${d.id}`;$("#question").textContent=d.q;
const z=zoom==="fit"?Math.max(0.5,Math.min(3,($("#stage").clientWidth-4)/d.w,(window.innerHeight*0.68-4)/d.h)):zoom;const img=$("#img"),svg=$("#svg");img.src=d.img;img.width=d.w*z;img.height=d.h*z;svg.setAttribute("width",d.w*z);svg.setAttribute("height",d.h*z);svg.setAttribute("viewBox",`0 0 ${d.w} ${d.h}`);svg.innerHTML="";
if(show){const sw=Math.max(1.5,Math.min(d.w,d.h)/220);for(const p of d.prims){if(p.t==="seg"){if(ext){const[e1,e2]=extend(p.a,p.b,d.w,d.h);svg.appendChild(el("line",{x1:e1[0],y1:e1[1],x2:e2[0],y2:e2[1],stroke:"#2a78d6","stroke-width":sw*0.7,"stroke-dasharray":"6 5",opacity:.8}))}
svg.appendChild(el("line",{x1:p.a[0],y1:p.a[1],x2:p.b[0],y2:p.b[1],stroke:"#ff1f6b","stroke-width":sw*2.6,opacity:.85,"stroke-linecap":"round"}));
for(const[q,n]of[[p.a,p.na],[p.b,p.nb]])svg.appendChild(el("circle",{cx:q[0],cy:q[1],r:sw*2.2,fill:"#ff1f6b",opacity:.9}))}
else if(p.t==="pt"){if(p.role==="anchor")svg.appendChild(el("rect",{x:p.p[0]-sw*3,y:p.p[1]-sw*3,width:sw*6,height:sw*6,fill:"none",stroke:"#ff1f6b","stroke-width":sw*1.6}));else svg.appendChild(el("circle",{cx:p.p[0],cy:p.p[1],r:sw*4.5,fill:"none",stroke:"#ff1f6b","stroke-width":sw*1.8}))}
else if(p.t==="circle"&&p.c){svg.appendChild(el("circle",{cx:p.c[0],cy:p.c[1],r:p.r,fill:"none",stroke:"#ff1f6b","stroke-width":sw*2.4,opacity:.8}))}}
if(ext){const pts=d.prims.filter(p=>p.t==="pt"&&p.role==="anchor");if(pts.length===2){const[e1,e2]=extend(pts[0].p,pts[1].p,d.w,d.h);svg.appendChild(el("line",{x1:e1[0],y1:e1[1],x2:e2[0],y2:e2[1],stroke:"#2a78d6","stroke-width":sw*0.7,"stroke-dasharray":"6 5",opacity:.8}))}}}
document.querySelectorAll("button.lab").forEach(b=>b.classList.toggle("on",b.dataset.v===s.label));$("#note").value=s.note||"";
$("#toggle").firstChild.textContent=show?"Hide overlay ":"Show overlay ";$("#ext").firstChild.textContent=ext?"Hide extensions ":"Extend lines ";$("#zoom").firstChild.textContent=zoom==="fit"?"Zoom fit ":`Zoom ${zoom}x `;
const n=DATA.filter(x=>st[x.id]&&st[x.id].label).length;$("#done").textContent=`${n}/${DATA.length} labelled`;$("#prog").style.width=`${100*n/DATA.length}%`}
function setLabel(v){const d=DATA[i];st[d.id]=Object.assign(st[d.id]||{},{label:v,note:$("#note").value});save();if(i<DATA.length-1)i++;render()}
document.querySelectorAll("button.lab").forEach(b=>b.onclick=()=>setLabel(b.dataset.v));
$("#note").oninput=()=>{const d=DATA[i];st[d.id]=Object.assign(st[d.id]||{},{note:$("#note").value});save()};
$("#prev").onclick=()=>{if(i>0)i--;render()};$("#next").onclick=()=>{if(i<DATA.length-1)i++;render()};
$("#toggle").onclick=()=>{show=!show;render()};$("#ext").onclick=()=>{ext=!ext;render()};$("#zoom").onclick=()=>{zoom=zoom==="fit"?1:zoom===1?2:zoom===2?3:"fit";render()};window.onresize=()=>{if(zoom==="fit")render()};
document.addEventListener("keydown",e=>{if(e.target.tagName==="TEXTAREA")return;const k=e.key.toLowerCase();if(k==="1")setLabel("yes");else if(k==="2")setLabel("no");else if(k==="3")setLabel("unsure");else if(k==="h")$("#toggle").click();else if(k==="e")$("#ext").click();else if(k==="z")$("#zoom").click();else if(k==="arrowleft")$("#prev").click();else if(k==="arrowright")$("#next").click()});
$("#export").onclick=()=>{const esc=v=>`"${String(v||"").replace(/"/g,'""')}"`;const lines=["gold_id,label,note"].concat(DATA.map(d=>{const s=st[d.id]||{};return[d.id,s.label||"",esc(s.note)].join(",")}));
const blob=new Blob([lines.join("\n")+"\n"],{type:"text/csv"});const a=document.createElement("a");a.href=URL.createObjectURL(blob);a.download="__CSV__";document.body.appendChild(a);a.click();a.remove()};
i=first();render();
</script></body></html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
