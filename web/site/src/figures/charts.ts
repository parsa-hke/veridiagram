// Results as constructions: horizontal Byrne bars (drawn left to right like a ruled line) and the
// Part A scaling curve plotted point by point, each point first marked with a compass arc.
// Colours follow the entity within a chart: the reference model is ink, then red, blue, yellow.
// Every chart has a legend (HTML, numbers traced), a hover tooltip, and its table beside it.

import { format, get, n } from "../data";
import { circle, dot, line, polyline, svg, text, type XY } from "../svg";

export interface Bar {
  cls: string;             // bar-ink | bar-red | bar-blue | bar-yellow
  label: string;           // series name for the tooltip
  base: string;            // JSON path holding .mean (and .sd, .n_seeds) or the value itself (valueOnly)
  valueOnly?: boolean;
}
export interface Group {
  label: string;
  bars: Bar[];
}
export interface LegendItem {
  cls: string;
  html: string;
}

const W = 380;
const LEFT = 8;
const RIGHT = 14;
const BAR = 11;
const GAP = 2;

const val = (b: Bar) => Number(get(b.valueOnly ? b.base : `${b.base}.mean`));
const sdOf = (b: Bar) => (b.valueOnly ? null : (get(`${b.base}.sd`) as number | null));

function tip(g: Group, b: Bar): string {
  const v = val(b);
  const sd = sdOf(b);
  const seeds = b.valueOnly ? null : get(`${b.base}.n_seeds`);
  return `${b.label} · ${g.label}: ${format(v, "f3")}${sd == null ? "" : ` ± ${format(sd, "f3")}`}${seeds == null ? "" : ` (${format(seeds, "seeds")})`}`;
}

export function legend(items: LegendItem[]): string {
  return `<ul class="legend">${items.map((i) => `<li><span class="key ${i.cls}"></span>${i.html}</li>`).join("")}</ul>`;
}

/** Grouped horizontal bars on a 0-1 scale (macro-F1 or precision). */
export function hbarChart(groups: Group[], items: LegendItem[], aria: string, caption = ""): string {
  const plotW = W - LEFT - RIGHT;
  const x = (v: number) => LEFT + plotW * Math.max(0, Math.min(1, v));
  const parts: string[] = [];
  const hits: string[] = [];
  let y = 6;
  let step = 1;
  for (const g of groups) {
    parts.push(text([LEFT, y + 11], g.label, "chart-group", 0, "start"));
    y += 18;
    for (const b of g.bars) {
      const v = val(b);
      const cy = y + BAR / 2;
      if (Number.isFinite(v)) {
        parts.push(line([x(0), cy], [x(v), cy], `bar chart-bar ${b.cls}`, step));
        const sd = sdOf(b);
        if (sd) {
          parts.push(`<g data-step="${step + 1}" data-kind="fade">${line([x(v - sd), cy], [x(v + sd), cy], "err")}${line(
            [x(v - sd), cy - 3.5], [x(v - sd), cy + 3.5], "err")}${line([x(v + sd), cy - 3.5], [x(v + sd), cy + 3.5], "err")}</g>`);
        }
        hits.push(`<rect class="hit" x="${LEFT}" y="${y - 1}" width="${plotW}" height="${BAR + GAP}" data-tip="${tip(g, b)}" />`);
      }
      y += BAR + GAP;
    }
    y += 12;
    step += 1;
  }
  const H = y + 16;
  const grid = [0, 0.25, 0.5, 0.75, 1]
    .map((t) => line([x(t), 2], [x(t), H - 18], t === 0 ? "axis" : "grid") + text([x(t), H - 4], t === 0 ? "0" : t === 1 ? "1" : String(t), "tick"))
    .join("");
  return `
<figure class="construction-figure chart">
  ${legend(items)}
  ${svg(W, H, `<g data-step="0" data-kind="fade">${grid}</g>${parts.join("")}${hits.join("")}`, aria)}
  ${caption ? `<figcaption>${caption}</figcaption>` : ""}
</figure>`;
}

/** Grouped bars for experiments x test units (macro-F1, geometric labels). */
export function experimentChart(
  rows: { exp: string; label: string; cls: string }[],
  units: { key: string; label: string }[],
  aria: string,
  basePath: (exp: string, unit: string) => string,
): string {
  const present = rows.filter((r) => get(`results:experiments.${r.exp}`) !== undefined);
  const groups = units.map((u) => ({
    label: u.label,
    bars: present.map((r) => ({ cls: r.cls, label: r.label, base: basePath(r.exp, u.key) })),
  }));
  const items = present.map((r) => ({
    cls: r.cls, html: `${r.label} <span class="seeds">(${n(`results:experiments.${r.exp}.n_seeds`, "seeds")})</span>`,
  }));
  return hbarChart(groups, items, aria);
}

// ------------------------------------------------------------------ Part A scaling curve

const SCALE_SERIES: Record<string, { cls: string; label: string }> = {
  geometry3k_test: { cls: "line-ink", label: "Geometry3K test" },
  pgdp5k_heldout: { cls: "line-blue", label: "PGDP5K held-out source" },
  synth_extrap: { cls: "line-red", label: "synthetic extrapolation" },
};

export function scalingChart(): string {
  const pts = (get("scaling:points") ?? []) as { unit: string; synthetic: number; mean: number; sd: number | null; n_seeds: number }[];
  const sizes = (get("scaling:sizes") ?? []) as number[];
  const order = Object.keys(SCALE_SERIES);             // ink first, so the held-out line is drawn over it
  const units = ((get("scaling:units") ?? []) as string[]).filter((u) => SCALE_SERIES[u])
    .sort((a, b) => order.indexOf(a) - order.indexOf(b));
  if (!pts.length || !sizes.length) return "";
  const Wc = 380, H = 250, L = 40, R = 16, T = 12, B = 34;
  const lo = Math.floor((Math.min(...pts.map((p) => p.mean - (p.sd ?? 0))) - 0.02) / 0.05) * 0.05;
  const hi = Math.ceil((Math.max(...pts.map((p) => p.mean + (p.sd ?? 0))) + 0.02) / 0.05) * 0.05;
  const x = (s: number) => L + ((Wc - L - R) * sizes.indexOf(s)) / Math.max(sizes.length - 1, 1);
  const y = (v: number) => T + (H - T - B) * (1 - (v - lo) / (hi - lo));
  const parts: string[] = [];
  const ticks: string[] = [];
  for (let t = lo; t <= hi + 1e-9; t += 0.05) {
    ticks.push(line([L, y(t)], [Wc - R, y(t)], Math.abs(t - lo) < 1e-9 ? "axis" : "grid"), text([L - 6, y(t) + 4], t.toFixed(2), "tick", null, "end"));
  }
  sizes.forEach((s, i) => ticks.push(text([x(s), H - 12], format(get(`scaling:sizes.${i}`), "int"), "tick")));
  const hits: string[] = [];
  units.forEach((u, k) => {
    const ser = SCALE_SERIES[u];
    const up = sizes.map((s) => pts.findIndex((p) => p.unit === u && p.synthetic === s)).filter((i) => i >= 0);
    const xy = up.map((i) => [x(pts[i].synthetic), y(pts[i].mean)] as XY);
    xy.forEach((p) => parts.push(circle(p, 11, "compass-arc", 1)));
    parts.push(polyline(xy, `series ${ser.cls}`, 2 + k));
    up.forEach((i, j) => {
      const p = pts[i];
      if (p.sd) parts.push(`<g data-step="${2 + units.length}" data-kind="fade">${line([xy[j][0], y(p.mean - p.sd)], [xy[j][0], y(p.mean + p.sd)], "err")}</g>`);
      parts.push(dot(xy[j], 4.5, `marker ${ser.cls}`, 2 + k));
      hits.push(`<circle class="hit" cx="${xy[j][0]}" cy="${xy[j][1]}" r="13" data-tip="${ser.label} · ${format(p.synthetic, "int")} synthetic: ${format(p.mean, "f3")}${p.sd == null ? "" : ` ± ${format(p.sd, "f3")}`} (${format(p.n_seeds, "seeds")})" />`);
    });
  });
  const items = units.map((u) => ({ cls: SCALE_SERIES[u].cls, html: SCALE_SERIES[u].label }));
  return `
<figure class="construction-figure chart">
  ${legend(items)}
  ${svg(Wc, H, `<g data-step="0" data-kind="fade">${ticks.join("")}</g>${parts.join("")}${hits.join("")}`,
    "Macro-F1 against the number of synthetic training diagrams, for three test sets")}
  <figcaption>Macro-F1 (geometric labels) against synthetic diagrams in training; bars are ± one standard
  deviation over seeds, where there is more than one.</figcaption>
</figure>`;
}
