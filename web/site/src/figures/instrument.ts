// Book II figures: the data as a Byrne partition of a square, and the model as an exploded figure
// (patches -> transformer layers -> head), with one patch token rising through the layers.

import { format, get, n } from "../data";
import { dot, line, polyline, svg, text, type XY } from "../svg";
import { legend } from "./charts";

export function partitionFigure(): string {
  const parts = [
    { key: "geometry3k_train_figures", label: "train", cls: "fill-red" },
    { key: "geometry3k_val_figures", label: "validation", cls: "fill-yellow" },
    { key: "geometry3k_test_figures", label: "test", cls: "fill-blue" },
    { key: "pgdp5k_heldout_figures", label: "held-out source", cls: "fill-ink" },
  ].map((p) => ({ ...p, v: Number(get(`facts:data.${p.key}`)) }));
  if (parts.some((p) => !Number.isFinite(p.v) || p.v <= 0)) return "";
  const total = parts.reduce((a, p) => a + p.v, 0);
  const S = 230, X = 65, Y = 10;
  const wTrain = (S * parts[0].v) / total;
  const rest = parts.slice(1);
  const restTotal = rest.reduce((a, p) => a + p.v, 0);
  const rects: { x: number; y: number; w: number; h: number; p: (typeof parts)[number] }[] = [
    { x: X, y: Y, w: wTrain, h: S, p: parts[0] },
  ];
  let y = Y;
  for (const p of rest) {
    const h = (S * p.v) / restTotal;
    rects.push({ x: X + wTrain, y, w: S - wTrain, h, p });
    y += h;
  }
  const body: string[] = [
    polyline([[X, Y], [X + S, Y], [X + S, Y + S], [X, Y + S], [X, Y]], "ink-line", 0),
    line([X + wTrain, Y], [X + wTrain, Y + S], "ink-line", 1),
  ];
  rects.slice(1, -1).forEach((r) => body.push(line([r.x, r.y + r.h], [r.x + r.w, r.y + r.h], "ink-line", 1)));
  rects.forEach((r) => {
    body.push(`<rect class="${r.p.cls}" x="${r.x + 1.5}" y="${r.y + 1.5}" width="${r.w - 3}" height="${r.h - 3}" data-step="2" data-kind="fade"
      data-tip="${r.p.label}: ${format(r.p.v, "int")} figures" />`);
    if (r.h > 26) body.push(text([r.x + r.w / 2, r.y + r.h / 2 + 5], format(r.p.v, "int"), `fill-label ${r.p.cls}-label`, 3));
  });
  const items = parts.map((p) => ({ cls: p.cls.replace("fill-", "bar-"), html: `${p.label}: ${n(`facts:data.${p.key}`, "int")}` }));
  return `
<figure class="construction-figure">
  ${legend(items)}
  ${svg(360, 250, body.join(""), "A square divided in proportion to the number of figures in each split")}
  <figcaption>The real figures, as a partition of a square: each region's area is its share. Synthetic diagrams
  are not drawn: the renderer can make as many as we like.</figcaption>
</figure>`;
}

export function modelFigure(): string {
  const key = get("facts:backbones.main.available") ? "main" : get("facts:backbones.demo.available") ? "demo" : null;
  if (!key) return "";
  const b = `facts:backbones.${key}`;
  const grid = Number(get(`${b}.grid`)), layers = Number(get(`${b}.layers`));
  const W = 360, H = 500;
  const img = { x: 105, y: 330, s: 150 };
  const body: string[] = [];
  const cell = img.s / grid;
  const gl: string[] = [];
  for (let i = 1; i < grid; i++) {
    gl.push(line([img.x + i * cell, img.y], [img.x + i * cell, img.y + img.s], "grid-fine"));
    gl.push(line([img.x, img.y + i * cell], [img.x + img.s, img.y + i * cell], "grid-fine"));
  }
  body.push(polyline([[img.x, img.y], [img.x + img.s, img.y], [img.x + img.s, img.y + img.s], [img.x, img.y + img.s], [img.x, img.y]], "ink-line", 0));
  body.push(`<g data-step="1" data-kind="fade">${gl.join("")}</g>`);
  const pc = Math.floor(grid * 0.62), pr = Math.floor(grid * 0.38);
  const patch: XY = [img.x + (pc + 0.5) * cell, img.y + (pr + 0.5) * cell];
  body.push(`<rect class="fill-blue" x="${img.x + pc * cell}" y="${img.y + pr * cell}" width="${cell}" height="${cell}" data-step="2" data-kind="fade" />`);
  const stackTop = 110, stackBottom = 300, sx0 = 70, sx1 = 290;
  const gapY = (stackBottom - stackTop) / Math.max(layers - 1, 1);
  const ls: string[] = [];
  for (let i = 0; i < layers; i++) ls.push(line([sx0, stackBottom - i * gapY], [sx1, stackBottom - i * gapY], "layer-line"));
  body.push(`<g data-step="3" data-kind="fade">${ls.join("")}</g>`);
  body.push(polyline([patch, [patch[0], stackBottom + 14], [patch[0], stackTop - 8]], "bar bar-blue token-path", 4));
  const head: XY = [180, 66];
  body.push(polyline([[patch[0], stackTop - 8], [head[0], head[1] + 16]], "ink-line", 5));
  body.push(`<rect class="head-box" x="${head[0] - 70}" y="${head[1] - 4}" width="140" height="22" data-step="5" data-kind="fade" />`);
  body.push(text([head[0], head[1] + 11], "CLS ⊕ mean · LayerNorm", "fig-label", 5));
  const rels = (get("facts:relations") ?? []) as string[];
  rels.forEach((_, i) => {
    const xx = head[0] + (i - (rels.length - 1) / 2) * 34;
    body.push(line([head[0], head[1] - 4], [xx, 28], "ink-line thin", 6), dot([xx, 22], 6, "output-dot", 6));
  });
  body.push(text([img.x + img.s / 2, img.y + img.s + 18], `${format(get("facts:input.long_side"), "int")} px · ${format(grid, "int")} × ${format(grid, "int")} patches`, "fig-label", 1));
  body.push(text([sx1 + 6, (stackTop + stackBottom) / 2], `${format(layers, "int")} layers`, "fig-label", 3, "start"));
  return `
<figure class="construction-figure narrow">
  ${svg(W, H, body.join(""), "The input image cut into a grid of patches; one patch token rising through the transformer layers to the head and six outputs")}
  <figcaption>The ${key === "main" ? "main model" : "demo model"} exploded: the image becomes
  ${n(`${b}.patch_tokens`, "int")} patch tokens, each rising through ${n(`${b}.layers`, "int")} layers; the head reads the
  CLS token and the mean patch token and answers with one sigmoid per relation: ${rels.join(", ").replace(/_/g, " ")}.</figcaption>
</figure>`;
}
