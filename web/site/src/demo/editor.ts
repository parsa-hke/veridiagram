// A compass-and-straightedge editor (WEBSITE_PLAN.md §3.1, §3.2), shared by Construct (blank page)
// and Upload (over the visitor's image). The page only draws previews: the model's input is made by
// the server from these coordinates with the repository's own renderer and overlay.

import type { Figure, Ref } from "./api";

export type Tool = "point" | "segment" | "circle" | "parallel" | "perpendicular" | "midpoint" | "oncircle" | "ask";
type XY = [number, number];

export interface Circle {
  center: string;
  through: string;
}

export interface EditorOptions {
  width: number;                         // viewBox (the figure's own coordinates)
  height: number;
  image?: { href: string; width: number; height: number };
  snapInk?: (p: XY) => XY;               // uploads: move a click onto the nearest dark ink
  onChange?: () => void;
}

const NAMES = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
const sub = (a: XY, b: XY): XY => [a[0] - b[0], a[1] - b[1]];
const len = (a: XY) => Math.hypot(a[0], a[1]);
const rot = (v: XY, deg: number): XY => {
  const r = (deg * Math.PI) / 180;
  return [v[0] * Math.cos(r) - v[1] * Math.sin(r), v[0] * Math.sin(r) + v[1] * Math.cos(r)];
};

/** Distance from p to segment ab. */
export function distToSegment(p: XY, a: XY, b: XY): number {
  const ab = sub(b, a), ap = sub(p, a);
  const t = Math.max(0, Math.min(1, (ap[0] * ab[0] + ap[1] * ab[1]) / (ab[0] ** 2 + ab[1] ** 2 || 1)));
  return len(sub(p, [a[0] + t * ab[0], a[1] + t * ab[1]]));
}

/** The query a selection of two parts asks, in the server's ref format, or null. */
export function queryOf(sel: Ref[]): { kind: string; refs: Ref[] } | null {
  if (sel.length !== 2) return null;
  const kinds = sel.map((r) => r[0]).sort().join("+");
  const by = (k: string) => sel.filter((r) => r[0] === k);
  if (kinds === "seg+seg") return { kind: "seg_pair", refs: sel };
  if (kinds === "circle+pt") return { kind: "pt_circle", refs: [by("pt")[0], by("circle")[0]] };
  if (kinds === "circle+seg") return { kind: "seg_circle", refs: [by("seg")[0], by("circle")[0]] };
  if (kinds === "pt+seg") {
    const s = by("seg")[0];
    return { kind: "pt_line", refs: [by("pt")[0], ["pt", s[1]], ["pt", s[2]]] };
  }
  return null;
}

export class Editor {
  points: Record<string, XY> = {};
  segments: [string, string][] = [];
  circles: Circle[] = [];
  selection: Ref[] = [];
  tool: Tool = "point";
  wobble = 0;                            // degrees for directions, percent for lengths
  readonly svg: SVGSVGElement;
  private pending: { kind: "point"; name: string } | { kind: "seg"; seg: [string, string] } | null = null;
  private history: string[] = [];
  private rng = 1;

  constructor(readonly host: HTMLElement, readonly opts: EditorOptions) {
    host.innerHTML = `<svg class="editor-svg" viewBox="0 0 ${opts.width} ${opts.height}" role="application"
      aria-label="Drawing area: click to place points and draw with the chosen tool"></svg>`;
    this.svg = host.querySelector("svg")!;
    this.svg.addEventListener("click", (e) => this.click(this.local(e)));
    this.render();
  }

  // -------------------------------------------------------------- state

  figure(): Figure {
    return {
      points: Object.fromEntries(Object.entries(this.points).map(([k, v]) => [k, [round(v[0]), round(v[1])]])),
      segments: this.segments.map((s) => [...s] as [string, string]),
      circles: this.circles.map((c) => ({ center: c.center, through: c.through })),
    };
  }

  load(f: { points: Record<string, XY>; segments: [string, string][]; circles?: Circle[] }): void {
    this.snapshot();
    this.points = { ...f.points };
    this.segments = [...f.segments];
    this.circles = [...(f.circles ?? [])];
    this.selection = [];
    this.pending = null;
    this.changed();
  }

  clear(): void {
    this.load({ points: {}, segments: [], circles: [] });
  }

  undo(): void {
    const last = this.history.pop();
    if (!last) return;
    const s = JSON.parse(last);
    this.points = s.points;
    this.segments = s.segments;
    this.circles = s.circles;
    this.selection = [];
    this.pending = null;
    this.changed();
  }

  setTool(t: Tool): void {
    this.tool = t;
    this.pending = null;
    if (t !== "ask") this.selection = [];
    this.render();
  }

  get query() {
    return queryOf(this.selection);
  }

  private snapshot(): void {
    this.history.push(JSON.stringify({ points: this.points, segments: this.segments, circles: this.circles }));
    if (this.history.length > 50) this.history.shift();
  }

  private changed(): void {
    this.render();
    this.opts.onChange?.();
  }

  private nextName(): string {
    for (let k = 0; ; k++) {
      const n = NAMES[k % 26] + (k >= 26 ? String(Math.floor(k / 26)) : "");
      if (!(n in this.points)) return n;
    }
  }

  private random(): number {                           // a small seeded generator for the wobble
    this.rng = (this.rng * 16807) % 2147483647;
    return (this.rng - 1) / 2147483646;
  }

  private jitter(): number {
    return (this.random() * 2 - 1) * this.wobble;
  }

  // -------------------------------------------------------------- hit testing

  private tol(): number {
    return Math.max(this.opts.width, this.opts.height) / 55;
  }

  private local(e: MouseEvent): XY {
    const r = this.svg.getBoundingClientRect();
    return [((e.clientX - r.left) / r.width) * this.opts.width, ((e.clientY - r.top) / r.height) * this.opts.height];
  }

  private pointAt(p: XY): string | null {
    let best: string | null = null, bd = this.tol();
    for (const [n, q] of Object.entries(this.points)) {
      const d = len(sub(p, q));
      if (d < bd) [best, bd] = [n, d];
    }
    return best;
  }

  private segmentAt(p: XY): [string, string] | null {
    let best: [string, string] | null = null, bd = this.tol() * 0.8;
    for (const s of this.segments) {
      const d = distToSegment(p, this.points[s[0]], this.points[s[1]]);
      if (d < bd) [best, bd] = [s, d];
    }
    return best;
  }

  private circleAt(p: XY): number | null {
    let best: number | null = null, bd = this.tol() * 0.8;
    this.circles.forEach((c, i) => {
      const d = Math.abs(len(sub(p, this.points[c.center])) - this.radius(c));
      if (d < bd) [best, bd] = [i, d];
    });
    return best;
  }

  radius(c: Circle): number {
    return len(sub(this.points[c.through], this.points[c.center]));
  }

  private placePoint(p: XY): string {
    const hit = this.pointAt(p);
    if (hit) return hit;
    const n = this.nextName();
    this.points[n] = this.opts.snapInk ? this.opts.snapInk(p) : p;
    return n;
  }

  private addSegment(a: string, b: string): void {
    if (a === b || this.segments.some(([x, y]) => (x === a && y === b) || (x === b && y === a))) return;
    this.segments.push([a, b]);
  }

  // -------------------------------------------------------------- tools

  private click(p: XY): void {
    const t = this.tool;
    if (t === "ask") return this.select(p);
    if (Object.keys(this.points).length >= 26 && !this.pointAt(p) && t !== "segment") return;
    if (t === "point") {
      this.snapshot();
      this.placePoint(p);
    } else if (t === "segment" || t === "circle") {
      if (this.pending?.kind !== "point") {
        this.snapshot();
        this.pending = { kind: "point", name: this.placePoint(p) };
        this.render();
        return;
      }
      const a = this.pending.name, b = this.placePoint(p);
      if (a !== b) {
        if (t === "segment") this.addSegment(a, b);
        else if (this.circles.length < 6) this.circles.push({ center: a, through: b });
      }
      this.pending = null;
    } else if (t === "midpoint") {
      const s = this.segmentAt(p);
      if (!s) return;
      this.snapshot();
      const [a, b] = [this.points[s[0]], this.points[s[1]]];
      const k = 0.5 * (1 + this.jitter() / 100);
      this.points[this.nextName()] = [a[0] + k * (b[0] - a[0]), a[1] + k * (b[1] - a[1])];
    } else if (t === "oncircle") {
      const i = this.circleAt(p);
      if (i === null) return;
      this.snapshot();
      const c = this.circles[i], o = this.points[c.center];
      const v = sub(p, o), r = this.radius(c) * (1 + this.jitter() / 100);
      this.points[this.nextName()] = [o[0] + (v[0] / len(v)) * r, o[1] + (v[1] / len(v)) * r];
    } else if (t === "parallel" || t === "perpendicular") {
      if (this.pending?.kind !== "seg") {
        const s = this.segmentAt(p);
        if (s) this.pending = { kind: "seg", seg: s };
        this.render();
        return;
      }
      this.snapshot();
      const [a, b] = this.pending.seg.map((n) => this.points[n]);
      const P = this.placePoint(p);
      const dir = sub(b, a);
      const L = len(dir);
      let v: XY = t === "parallel" ? dir : rot(dir, 90);
      v = rot(v, this.jitter());
      const scale = (t === "parallel" ? 1 : 0.6) * (1 + this.jitter() / 100);
      const q: XY = [this.points[P][0] + (v[0] / L) * L * scale, this.points[P][1] + (v[1] / L) * L * scale];
      const Q = this.nextName();
      this.points[Q] = q;
      this.addSegment(P, Q);
      this.pending = null;
    }
    this.changed();
  }

  private select(p: XY): void {
    const pt = this.pointAt(p);
    const s = pt ? null : this.segmentAt(p);
    const c = pt || s ? null : this.circleAt(p);
    const ref: Ref | null = pt ? ["pt", pt] : s ? ["seg", ...s] : c !== null ? ["circle", c] : null;
    if (!ref) return;
    const key = JSON.stringify(ref);
    const i = this.selection.findIndex((r) => JSON.stringify(r) === key);
    if (i >= 0) this.selection.splice(i, 1);
    else this.selection = [...this.selection.slice(-1), ref];
    this.changed();
  }

  // -------------------------------------------------------------- drawing

  private selClass(ref: Ref): string {
    const i = this.selection.findIndex((r) => JSON.stringify(r) === JSON.stringify(ref));
    return i < 0 ? "" : i === 0 ? " sel-red" : " sel-yellow";
  }

  render(): void {
    const k = Math.max(this.opts.width, this.opts.height) / 600;       // stroke scale for big uploads
    const parts: string[] = [];
    if (this.opts.image) {
      parts.push(`<image href="${this.opts.image.href}" x="0" y="0" width="${this.opts.image.width}" height="${this.opts.image.height}" />`);
    }
    this.circles.forEach((c, i) => {
      const o = this.points[c.center];
      parts.push(`<circle class="ed-circle${this.selClass(["circle", i])}" cx="${o[0]}" cy="${o[1]}" r="${this.radius(c)}" style="stroke-width:${1.6 * k}" />`);
    });
    for (const [a, b] of this.segments) {
      const [p, q] = [this.points[a], this.points[b]];
      const cls = this.selClass(["seg", a, b]) || this.selClass(["seg", b, a]);
      const pend = this.pending?.kind === "seg" && this.pending.seg[0] === a && this.pending.seg[1] === b ? " pending" : "";
      parts.push(`<line class="ed-seg${cls}${pend}" x1="${p[0]}" y1="${p[1]}" x2="${q[0]}" y2="${q[1]}" style="stroke-width:${1.8 * k}" />`);
    }
    for (const [n, p] of Object.entries(this.points)) {
      const pend = this.pending?.kind === "point" && this.pending.name === n ? " pending" : "";
      parts.push(`<circle class="ed-point${this.selClass(["pt", n])}${pend}" cx="${p[0]}" cy="${p[1]}" r="${(pend ? 7 : 5) * k}" />`,
        `<text class="ed-label" x="${p[0] + 7 * k}" y="${p[1] - 7 * k}" style="font-size:${15 * k}px">${n}</text>`);
    }
    if (!Object.keys(this.points).length && !this.opts.image) {
      const x = this.opts.width / 2, y = this.opts.height / 2;
      parts.push(`<text class="ed-empty" x="${x}" y="${y}" text-anchor="middle" style="font-size:${17 * k}px">
        <tspan x="${x}" dy="-0.6em">Choose a tool above, then click in here.</tspan>
        <tspan x="${x}" dy="1.5em">Segment draws a line in two clicks.</tspan></text>`);
    }
    this.svg.innerHTML = parts.join("");
    this.svg.dataset.mode = this.tool;
  }
}

const round = (v: number) => Math.round(v * 100) / 100;

/** Uploads: a snapping function that moves a click to the darkest pixel within `r` pixels. */
export function inkSnapper(img: HTMLImageElement | HTMLCanvasElement, r = 7): (p: XY) => XY {
  const c = document.createElement("canvas");
  c.width = img.width;
  c.height = img.height;
  const ctx = c.getContext("2d");
  if (!ctx) return (p) => p;
  ctx.drawImage(img, 0, 0);
  const data = ctx.getImageData(0, 0, c.width, c.height).data;
  return ([x, y]) => {
    let best: XY = [x, y], bv = Infinity;
    for (let dy = -r; dy <= r; dy++) {
      for (let dx = -r; dx <= r; dx++) {
        const px = Math.round(x + dx), py = Math.round(y + dy);
        if (px < 0 || py < 0 || px >= c.width || py >= c.height || dx * dx + dy * dy > r * r) continue;
        const i = 4 * (py * c.width + px);
        const lum = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2] + 2 * Math.hypot(dx, dy);
        if (lum < bv) [best, bv] = [[px, py], lum];
      }
    }
    return bv < 160 ? best : [x, y];                  // nothing dark nearby: keep the click
  };
}
