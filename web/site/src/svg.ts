// SVG builders for constructions. Coordinates are diagram pixels mapped into a viewBox by fit().
// Every drawable takes a step (draw order) and a class (colour comes from CSS tokens, so both editions work).

export type XY = [number, number];

export interface Diagram {
  id: string;
  width: number;
  height: number;
  points: Record<string, XY>;
  segments: [string, string][];
  circles: { center: string | null; cx: number; cy: number; r: number }[];
}

export interface Fit {
  (p: XY): XY;
  scale: number;
}

/** Map the bounding box of `pts` (and circles) into a w x h box with padding, preserving aspect. */
export function fit(pts: XY[], circles: { cx: number; cy: number; r: number }[], w: number, h: number, pad = 12): Fit {
  const xs = [...pts.map((p) => p[0]), ...circles.flatMap((c) => [c.cx - c.r, c.cx + c.r])];
  const ys = [...pts.map((p) => p[1]), ...circles.flatMap((c) => [c.cy - c.r, c.cy + c.r])];
  const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
  const s = Math.min((w - 2 * pad) / Math.max(x1 - x0, 1), (h - 2 * pad) / Math.max(y1 - y0, 1));
  const ox = (w - s * (x1 - x0)) / 2 - s * x0, oy = (h - s * (y1 - y0)) / 2 - s * y0;
  const f = ((p: XY) => [ox + s * p[0], oy + s * p[1]] as XY) as Fit;
  f.scale = s;
  return f;
}

const r2 = (v: number) => Math.round(v * 100) / 100;

const stepAttr = (step: number | null, kind?: "fade") =>
  step === null ? "" : ` data-step="${step}"${kind ? ` data-kind="${kind}"` : ' pathLength="1"'}`;

export const line = (a: XY, b: XY, cls: string, step: number | null = null) =>
  `<line class="${cls}" x1="${r2(a[0])}" y1="${r2(a[1])}" x2="${r2(b[0])}" y2="${r2(b[1])}"${stepAttr(step)} />`;

export const circle = (c: XY, r: number, cls: string, step: number | null = null) =>
  `<circle class="${cls}" cx="${r2(c[0])}" cy="${r2(c[1])}" r="${r2(r)}"${stepAttr(step)} />`;

export const dot = (c: XY, r: number, cls: string, step: number | null = null) =>
  `<circle class="${cls}" cx="${r2(c[0])}" cy="${r2(c[1])}" r="${r2(r)}"${stepAttr(step, "fade")} />`;

export const polyline = (pts: XY[], cls: string, step: number | null = null) =>
  `<polyline class="${cls}" points="${pts.map((p) => `${r2(p[0])},${r2(p[1])}`).join(" ")}"${stepAttr(step)} />`;

export const path = (d: string, cls: string, step: number | null = null) =>
  `<path class="${cls}" d="${d}"${stepAttr(step)} />`;

export const text = (p: XY, s: string, cls: string, step: number | null = null, anchor = "middle") =>
  `<text class="${cls}" x="${r2(p[0])}" y="${r2(p[1])}" text-anchor="${anchor}"${stepAttr(step, "fade")}>${s}</text>`;

/** An arc of radius r around c from angle a0 to a1 (radians, SVG orientation). */
export function arc(c: XY, r: number, a0: number, a1: number, cls: string, step: number | null = null): string {
  const p0: XY = [c[0] + r * Math.cos(a0), c[1] + r * Math.sin(a0)];
  const p1: XY = [c[0] + r * Math.cos(a1), c[1] + r * Math.sin(a1)];
  const large = Math.abs(a1 - a0) > Math.PI ? 1 : 0;
  const sweep = a1 > a0 ? 1 : 0;
  return path(`M${r2(p0[0])},${r2(p0[1])} A${r2(r)},${r2(r)} 0 ${large} ${sweep} ${r2(p1[0])},${r2(p1[1])}`, cls, step);
}

export const svg = (w: number, h: number, body: string, label: string, cls = "construction") =>
  `<svg class="${cls}" viewBox="0 0 ${w} ${h}" role="img" aria-label="${label.replace(/"/g, "&quot;")}">${body}</svg>`;

// ------------------------------------------------------------------ vector helpers

export const sub = (a: XY, b: XY): XY => [a[0] - b[0], a[1] - b[1]];
export const add = (a: XY, b: XY): XY => [a[0] + b[0], a[1] + b[1]];
export const mul = (a: XY, k: number): XY => [a[0] * k, a[1] * k];
export const len = (a: XY) => Math.hypot(a[0], a[1]);
export const unit = (a: XY): XY => mul(a, 1 / (len(a) || 1));
export const mid = (a: XY, b: XY): XY => [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
export const perp = (a: XY): XY => [-a[1], a[0]];

/** Intersection of lines ab and cd, or null when (nearly) parallel. */
export function intersect(a: XY, b: XY, c: XY, d: XY): XY | null {
  const r = sub(b, a), s = sub(d, c);
  const den = r[0] * s[1] - r[1] * s[0];
  if (Math.abs(den) < 1e-9) return null;
  const t = ((c[0] - a[0]) * s[1] - (c[1] - a[1]) * s[0]) / den;
  return add(a, mul(r, t));
}
