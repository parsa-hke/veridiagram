// Geometric constructions: the frontispiece, Book I's propositions and Book II's query figure.
// Diagrams are real synthetic diagrams exported by web/figures.py; green marks the machine's highlight.

import { format, get, n } from "../data";
import {
  add, arc, circle, dot, fit, intersect, len, line, mid, mul, perp, sub, svg, text, unit,
  type Diagram, type Fit, type XY,
} from "../svg";

const BYRNE = ["bar-red", "bar-yellow", "bar-blue"];

type Ref = [string, ...(string | number)[]];

export const RELATION_WORDS: Record<string, string> = {
  parallel: "parallel",
  perpendicular: "perpendicular",
  equal_length: "equal in length",
  on_circle: "on the circle",
  tangent: "tangent",
  collinear: "collinear",
};

const segXY = (d: Diagram, f: Fit, ref: Ref): [XY, XY] => [f(d.points[ref[1] as string]), f(d.points[ref[2] as string])];

/** The machine's highlight on the queried primitives: a green core inside a translucent halo. */
export function highlight(d: Diagram, f: Fit, refs: Ref[], step: number | null, w = 1): string {
  const parts: string[] = [];
  const both = (draw: (cls: string) => string) => parts.push(draw(`machine-halo`), draw(`machine-core`));
  const pts = refs.filter((r) => r[0] === "pt");
  for (const r of refs) {
    if (r[0] === "seg") {
      const [a, b] = segXY(d, f, r);
      both((cls) => line(a, b, cls, step));
    } else if (r[0] === "circle") {
      const c = d.circles[r[1] as number];
      if (c) both((cls) => circle(f([c.cx, c.cy]), c.r * f.scale, cls, step));
    }
  }
  if (pts.length === 3) {                                // pt_line: P and the line AB
    const [, a, b] = pts.map((r) => f(d.points[r[1] as string]));
    both((cls) => line(a, b, cls, step));
    parts.push(dot(f(d.points[pts[0][1] as string]), 5.5 * w, "machine-dot", step));
  } else {
    for (const r of pts) parts.push(dot(f(d.points[r[1] as string]), 5.5 * w, "machine-dot", step));
  }
  return `<g class="machine-trace">${parts.join("")}</g>`;
}

/** The whole diagram in ink: compass (circles) first, then straightedge (segments), then the points. */
export function inkDiagram(d: Diagram, f: Fit, s0: number, opts: { dots?: boolean; labels?: boolean } = {}): string {
  const out: string[] = [];
  for (const c of d.circles) out.push(circle(f([c.cx, c.cy]), c.r * f.scale, "ink-line thin", s0));
  for (const [a, b] of d.segments) out.push(line(f(d.points[a]), f(d.points[b]), "ink-line thin", s0 + 1));
  if (opts.dots !== false) {
    for (const p of Object.values(d.points)) out.push(dot(f(p), 2.2, "ink-dot", s0 + 2));
  }
  if (opts.labels) {
    const c = Object.values(d.points).reduce((a, p) => add(a, mul(p, 1 / Object.keys(d.points).length)), [0, 0] as XY);
    for (const [name, p] of Object.entries(d.points)) {
      const away = unit(sub(p, c));
      out.push(text(add(f(p), add(mul(away, 11), [0, 4])), name, "point-label", s0 + 2));
    }
  }
  return out.join("");
}

// ------------------------------------------------------------------ marks (what a textbook prints)

function ticks(a: XY, b: XY, k: number): string {
  const m = mid(a, b), u = unit(sub(b, a)), p = perp(u);
  return Array.from({ length: k }, (_, i) => {
    const c = add(m, mul(u, (i - (k - 1) / 2) * 5));
    return line(add(c, mul(p, 6)), sub(c, mul(p, 6)), "ink-line mark", null);
  }).join("");
}

function arrowhead(a: XY, b: XY): string {
  const m = mid(a, b), u = unit(sub(b, a)), p = perp(u);
  const tip = add(m, mul(u, 5));
  return `<polyline class="ink-line mark" points="${[add(sub(tip, mul(u, 8)), mul(p, 5)), tip, sub(sub(tip, mul(u, 8)), mul(p, 5))]
    .map((q) => q.join(",")).join(" ")}" />`;
}

function rightAngleBox(v: XY, u1: XY, u2: XY, size = 11): string {
  const a = add(v, mul(u1, size)), b = add(add(v, mul(u1, size)), mul(u2, size)), c = add(v, mul(u2, size));
  return `<polyline class="ink-line mark" points="${[a, b, c].map((q) => q.join(",")).join(" ")}" />`;
}

/** The mark a textbook prints for an asserted relation between two segments. */
function mark(type: string, s1: [XY, XY], s2: [XY, XY], k: number): string {
  if (type === "equal_length") return ticks(...s1, k) + ticks(...s2, k);
  if (type === "parallel") return arrowhead(...s1) + arrowhead(...s2);
  if (type === "perpendicular") {
    const v = intersect(s1[0], s1[1], s2[0], s2[1]);
    if (!v) return "";
    const away = (s: [XY, XY]) => unit(sub(len(sub(s[0], v)) > len(sub(s[1], v)) ? s[0] : s[1], v));
    return rightAngleBox(v, away(s1), away(s2));
  }
  return "";
}

// ------------------------------------------------------------------ frontispiece

export function hero(): string {
  const h = get("figures:hero");
  if (!h) return "";
  const d: Diagram = h.diagram;
  const refs: Ref[] = h.query.refs;
  const f = fit(Object.values(d.points), d.circles, 320, 240, 14);
  const body: string[] = [];
  for (const c of d.circles) {                          // compass: set the radius, then sweep
    const cc = f([c.cx, c.cy]), r = c.r * f.scale;
    body.push(line(cc, add(cc, [r, 0]), "construction-line", 0), circle(cc, r, "ink-line thin", 1));
  }
  for (const [p, q] of d.segments) body.push(line(f(d.points[p]), f(d.points[q]), "ink-line thin", 2));
  for (const p of Object.values(d.points)) body.push(dot(f(p), 2, "ink-dot", 2));
  const segs = refs.map((r) => segXY(d, f, r));
  segs.forEach(([a, b], i) => body.push(line(a, b, `bar hero-bar ${BYRNE[i]}`, 3)));
  body.push(highlight(d, f, refs, 4));
  return `
<figure class="emblem-figure construct-hero">
  ${svg(320, 240, body.join(""), "A circle drawn with a compass, two lines drawn with a straightedge and painted red and yellow, then traced in green by the model", "construction emblem")}
  <figcaption>
    <span class="hero-verdict" data-step="5" data-kind="fade"><span class="smallcaps">parallel</span> ·
      ${n("figures:hero.verdicts.parallel.prob", "f2")}</span>
    <span class="hero-note" data-step="5" data-kind="fade">A real question from the extrapolation set. The ruler finds the
      lines ${n("figures:hero.angle_off_deg", "deg")} from parallel; the machine says parallel with probability
      ${n("figures:hero.verdicts.parallel.prob", "f2")}.</span>
  </figcaption>
</figure>`;
}

// ------------------------------------------------------------------ Proposition I

export function unmarkedFigure(): string {
  const u = get("figures:unmarked");
  if (!u) return "";
  const d: Diagram = u.diagram;
  const f = fit(Object.values(d.points), [], 360, 250, 22);
  const body: string[] = [inkDiagram(d, f, 0, { dots: true })];
  (u.marked as { type: string; args: Ref[] }[]).forEach((r, i) => {
    body.push(`<g data-step="3" data-kind="fade">${mark(r.type, segXY(d, f, r.args[0]), segXY(d, f, r.args[1]), i + 1)}</g>`);
  });
  (u.unmarked as { type: string; args: Ref[] }[]).forEach((r, i) => {
    for (const a of r.args) {
      const [p, q] = segXY(d, f, a);
      body.push(line(p, q, `bar thin-bar ${BYRNE[i]}`, 4 + i));
    }
  });
  const legend = (u.unmarked as { type: string }[]).map((r, i) => {
    const off = r.type === "equal_length"
      ? `lengths ${n(`figures:unmarked.unmarked.${i}.off_pct`, "pctv")} apart`
      : `${n(`figures:unmarked.unmarked.${i}.off_deg`, "deg")} off`;
    return `<li><span class="key ${BYRNE[i]}"></span>${RELATION_WORDS[r.type]}, unmarked (${off})</li>`;
  }).join("");
  return `
<figure class="construction-figure">
  ${svg(360, 250, body.join(""), "A diagram drawn by the project's renderer; its printed marks in black, and true but unmarked relations painted in red, yellow and blue")}
  <figcaption>
    <ul class="legend">
      <li><span class="key key-ink"></span>printed marks: ${(u.marked as { type: string }[]).map((r) => RELATION_WORDS[r.type]).join(", ")}</li>
      ${legend}
    </ul>
    <p>A diagram drawn by our renderer. The ruler confirms ${n("figures:unmarked.geometric_total", "int")} relations between its
    segments; the page shows the marked ones in black and paints three that carry no mark.</p>
  </figcaption>
</figure>`;
}

// ------------------------------------------------------------------ Proposition II

export function notToScaleFigure(): string {
  const off = Number(get("facts:tolerances.not_to_scale_angle_deg"));
  if (!Number.isFinite(off)) return "";
  const v: XY = [80, 210];
  const rad = (deg: number) => (deg * Math.PI) / 180;
  const along: XY = [1, 0];
  const drawn: XY = [Math.cos(rad(-(90 - off))), Math.sin(rad(-(90 - off)))];
  const truth: XY = [0, -1];
  const L = 170;
  const body = [
    line(v, add(v, mul(along, 230)), "ink-line", 0),
    line(v, add(v, mul(drawn, L)), "ink-line", 0),
    `<g data-step="1" data-kind="fade">${rightAngleBox(v, along, drawn, 16)}</g>`,
    line(v, add(v, mul(truth, L)), "construction-line", 2),
    arc(v, 150, -Math.PI / 2, rad(-(90 - off)), "ink-line accent-red", 3),
    text(add(v, mul([Math.cos(rad(-(90 - off / 2))), Math.sin(rad(-(90 - off / 2)))], 168)),
      `${format(get("facts:tolerances.not_to_scale_angle_deg"), "deg")}`, "fig-label accent-label", 3),
  ];
  return `
<figure class="construction-figure narrow">
  ${svg(320, 240, body.join(""), "A right-angle mark on two lines that actually meet off a right angle; the true perpendicular drawn faintly")}
  <figcaption>The box says “perpendicular”; the faint line is the true perpendicular. The drawn line is
  ${n("facts:tolerances.not_to_scale_angle_deg", "deg")} off, the threshold at which a figure is flagged as not to scale.</figcaption>
</figure>`;
}

// ------------------------------------------------------------------ Proposition III: the Ruler

export function rulerFigure(): string {
  return `
<figure class="construction-figure ruler" data-ruler>
  <svg class="construction ruler-svg" viewBox="0 0 360 240" role="img"
    aria-label="Two segments; drag the end of the yellow one to change its angle and length">
    <line class="bar bar-red" data-seg="1" />
    <line class="bar bar-yellow" data-seg="2" />
    <circle class="ruler-handle" r="11" tabindex="0" role="slider" aria-label="End of the yellow segment: arrow keys turn it, shift and arrows stretch it" />
  </svg>
  <figcaption>
    <p class="ruler-readout" aria-live="polite">
      <span class="ruler-line"><span class="smallcaps">Parallel</span>: <span class="live" data-out="angle"></span>
        from parallel, within ${n("facts:tolerances.parallel_deg", "deg")}? <strong class="live" data-out="par"></strong></span>
      <span class="ruler-line"><span class="smallcaps">Equal</span>: lengths <span class="live" data-out="ratio"></span> apart,
        within ${n("facts:tolerances.equal_length_pct", "pctv")}? <strong class="live" data-out="eq"></strong></span>
    </p>
    <p>Drag the end of the yellow segment (or use the arrow keys) and watch the verdicts flip at the tolerances.</p>
  </figcaption>
</figure>`;
}

/** The Ruler's checks, as verifier/checks.py states them: parallel when |sin θ| <= sin(tol), equal
 * when | |s1| - |s2| | / max <= tol. Illustration only; the model's input never comes from here. */
export function rulerVerdict(a: [XY, XY], b: [XY, XY], tolDeg: number, tolRel: number) {
  const u = unit(sub(a[1], a[0])), v = unit(sub(b[1], b[0]));
  const sin = Math.abs(u[0] * v[1] - u[1] * v[0]);
  const l1 = len(sub(a[1], a[0])), l2 = len(sub(b[1], b[0]));
  const rel = Math.abs(l1 - l2) / Math.max(l1, l2);
  return {
    angleDeg: (Math.asin(Math.min(sin, 1)) * 180) / Math.PI,
    parallel: sin <= Math.sin((tolDeg * Math.PI) / 180),
    rel,
    equal: rel <= tolRel,
  };
}

export function wireRuler(root: Element): void {
  const fig = root.querySelector<HTMLElement>("[data-ruler]");
  if (!fig) return;
  const tolDeg = Number(get("facts:tolerances.parallel_deg"));
  const tolRel = Number(get("facts:tolerances.equal_length_pct")) / 100;
  const svgEl = fig.querySelector("svg")!;
  const [s1, s2] = [1, 2].map((i) => svgEl.querySelector<SVGLineElement>(`[data-seg="${i}"]`)!);
  const handle = svgEl.querySelector<SVGCircleElement>(".ruler-handle")!;
  const A: [XY, XY] = [[50, 80], [300, 80]];
  const C: XY = [50, 170];
  let D: XY = [282, 162];                                 // starts a little off parallel and a little short
  const out = (k: string) => fig.querySelector<HTMLElement>(`[data-out="${k}"]`)!;
  const render = () => {
    const set = (el: SVGLineElement, p: XY, q: XY) => {
      el.setAttribute("x1", String(p[0])); el.setAttribute("y1", String(p[1]));
      el.setAttribute("x2", String(q[0])); el.setAttribute("y2", String(q[1]));
    };
    set(s1, ...A);
    set(s2, C, D);
    handle.setAttribute("cx", String(D[0]));
    handle.setAttribute("cy", String(D[1]));
    const r = rulerVerdict(A, [C, D], tolDeg, tolRel);
    out("angle").textContent = `${r.angleDeg.toFixed(1)}°`;
    out("ratio").textContent = `${(100 * r.rel).toFixed(1)}%`;
    out("par").textContent = r.parallel ? "yes" : "no";
    out("eq").textContent = r.equal ? "yes" : "no";
    handle.setAttribute("aria-valuetext", `${r.angleDeg.toFixed(1)} degrees from parallel, lengths ${(100 * r.rel).toFixed(1)} percent apart`);
    fig.classList.toggle("is-parallel", r.parallel);
    fig.classList.toggle("is-equal", r.equal);
  };
  const toLocal = (e: PointerEvent): XY => {
    const pt = svgEl.getBoundingClientRect();
    return [((e.clientX - pt.left) / pt.width) * 360, ((e.clientY - pt.top) / pt.height) * 240];
  };
  const clampXY = (p: XY): XY => [Math.min(345, Math.max(90, p[0])), Math.min(230, Math.max(10, p[1]))];
  let dragging = false;
  handle.addEventListener("pointerdown", (e) => {
    dragging = true;
    handle.setPointerCapture(e.pointerId);
    e.preventDefault();
  });
  handle.addEventListener("pointermove", (e) => {
    if (!dragging) return;
    D = clampXY(toLocal(e));
    render();
  });
  handle.addEventListener("pointerup", () => (dragging = false));
  handle.addEventListener("keydown", (e) => {
    const v = sub(D, C);
    const ang = Math.atan2(v[1], v[0]);
    const L = len(v);
    const step = e.shiftKey ? 0 : (0.5 * Math.PI) / 180;
    const grow = e.shiftKey ? 4 : 0;
    let na = ang, nl = L;
    if (e.key === "ArrowUp" || e.key === "ArrowLeft") { na -= step; nl -= grow; }
    else if (e.key === "ArrowDown" || e.key === "ArrowRight") { na += step; nl += grow; }
    else return;
    e.preventDefault();
    D = clampXY(add(C, [nl * Math.cos(na), nl * Math.sin(na)]));
    render();
  });
  render();
}

// ------------------------------------------------------------------ Book II: four queries on one diagram

const KIND_WORDS: Record<string, string> = {
  seg_pair: "two segments",
  pt_circle: "a point and a circle",
  seg_circle: "a segment and a circle",
  pt_line: "a point and a line",
};

export function queriesFigure(): string {
  const qs = get("figures:queries") as { kind: string; refs: Ref[]; verdicts: Record<string, { truth: boolean }> }[] | null;
  const h = get("figures:hero");
  if (!qs || !h) return "";
  const d: Diagram = h.diagram;
  const panels = qs.map((q, i) => {
    const f = fit(Object.values(d.points), d.circles, 220, 200, 10);
    const rels = Object.keys(q.verdicts);
    const rel = rels.find((r) => q.verdicts[r].truth) ?? rels[0];
    return `<div class="query-panel">
      ${svg(220, 200, inkDiagram(d, f, 0, { dots: true }) + highlight(d, f, q.refs, 3, 0.8), `Query ${i + 1}: ${KIND_WORDS[q.kind]} highlighted in green`)}
      <p class="query-caption"><span class="smallcaps">${KIND_WORDS[q.kind]}</span><br>
        ${RELATION_WORDS[rel]}? ruler: ${q.verdicts[rel].truth ? "yes" : "no"}; machine:
        ${n(`figures:queries.${i}.verdicts.${rel}.prob`, "f2")}</p>
    </div>`;
  });
  return `
<figure class="construction-figure wide-figure queries">
  <div class="query-grid">${panels.join("")}</div>
  <figcaption>One diagram, four questions. The green is all the model is told about what is being asked.</figcaption>
</figure>`;
}
