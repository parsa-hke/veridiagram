// The Gallery (WEBSITE_PLAN.md §3.4): twelve synthetic diagrams with every question already answered
// (web/precompute.py), so "read the whole figure" is instant without the server. Clicking a statement
// previews its parts in green over the diagram (a preview drawn by the page, not the model's input).

import type { Ref, Result, Summary } from "./api";
import { readout } from "./witnesses";

interface Geo {
  id: string;
  width: number;
  height: number;
  points: Record<string, [number, number]>;
  segments: [string, string][];
  circles: { center: string | null; cx: number; cy: number; r: number }[];
}
interface Item extends Summary {
  image: string;
  diagram: Geo;
  points: number;
  pool: string;
  results: Result[];
  queries_asked: number;
  queries_total: number;
}

export function previewSVG(d: Geo, refs: Ref[]): string {
  const P = (n: string) => d.points[n];
  const k = Math.max(d.width, d.height) / 448;
  const parts: string[] = [];
  const line = (a: [number, number], b: [number, number]) =>
    parts.push(`<line class="machine-halo" x1="${a[0]}" y1="${a[1]}" x2="${b[0]}" y2="${b[1]}" style="stroke-width:${7 * k}" />`,
      `<line class="machine-core" x1="${a[0]}" y1="${a[1]}" x2="${b[0]}" y2="${b[1]}" style="stroke-width:${3 * k}" />`);
  const pts = refs.filter((r) => r[0] === "pt");
  if (pts.length === 3) line(P(String(pts[1][1])), P(String(pts[2][1])));
  for (const r of refs) {
    if (r[0] === "seg") line(P(String(r[1])), P(String(r[2])));
    if (r[0] === "circle") {
      const c = d.circles[Number(r[1])];
      if (c) parts.push(`<circle class="machine-halo" cx="${c.cx}" cy="${c.cy}" r="${c.r}" style="stroke-width:${7 * k}" />`,
        `<circle class="machine-core" cx="${c.cx}" cy="${c.cy}" r="${c.r}" style="stroke-width:${3 * k}" />`);
    }
  }
  for (const r of pts.slice(0, 1).concat(pts.length === 3 ? [] : pts.slice(1))) {
    const [x, y] = P(String(r[1]));
    parts.push(`<circle class="machine-dot" cx="${x}" cy="${y}" r="${6 * k}" />`);
  }
  return parts.join("");
}

export function galleryPanel(host: HTMLElement, base = "data/"): () => void {
  host.innerHTML = `<p>Twelve diagrams drawn by our renderer, none of them seen by the model in training: six of
  textbook size, six with twelve to twenty points. Every question has been asked already.</p>
  <div class="gallery-grid" data-grid><p class="instrument-note">Opening the gallery…</p></div>
  <div class="gallery-detail" data-detail aria-live="polite"></div>`;
  const grid = host.querySelector<HTMLElement>("[data-grid]")!;
  const detail = host.querySelector<HTMLElement>("[data-detail]")!;
  let loaded = false;

  const open = (it: Item, btn: HTMLElement) => {
    grid.querySelectorAll(".thumb").forEach((t) => t.setAttribute("aria-pressed", String(t === btn)));
    const d = it.diagram;
    const circles = d.circles.map((c) => c.center);
    detail.innerHTML = `
<div class="gallery-view">
  <div class="gallery-figure">
    <img src="${base}${it.image}" alt="A synthetic diagram with ${it.points} points" width="${d.width}" height="${d.height}">
    <svg viewBox="0 0 ${d.width} ${d.height}" class="gallery-overlay" aria-hidden="true"></svg>
    <p class="witness-note">Click a statement to see its parts in green (a preview drawn by the page).</p>
  </div>
  ${readout(it, it.queries_asked, it.queries_total, circles, it.results)}
</div>`;
    const overlay = detail.querySelector<SVGSVGElement>(".gallery-overlay")!;
    detail.querySelectorAll<HTMLElement>("li[data-refs]").forEach((li) => {
      const showIt = () => {
        detail.querySelectorAll("li.active").forEach((x) => x.classList.remove("active"));
        li.classList.add("active");
        overlay.innerHTML = previewSVG(d, JSON.parse(li.dataset.refs!));
      };
      li.addEventListener("click", showIt);
      li.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          showIt();
        }
      });
    });
  };

  const load = async () => {
    if (loaded) return;
    loaded = true;
    try {
      const r = await fetch(`${base}gallery.json`);
      if (!r.ok) throw new Error();
      const g = (await r.json()) as { items: Item[] };
      grid.innerHTML = g.items.map((it, i) => `<button type="button" class="thumb" data-i="${i}" aria-pressed="false"
        aria-label="Diagram ${i + 1}, ${it.points} points"><img src="${base}${it.image}" alt="" loading="lazy" width="120" height="120"></button>`).join("");
      grid.querySelectorAll<HTMLButtonElement>(".thumb").forEach((b) => b.addEventListener("click", () => open(g.items[Number(b.dataset.i)], b)));
    } catch {
      grid.innerHTML = `<p class="error">The gallery could not be loaded.</p>`;
    }
  };
  return load;                                         // called when the Gallery tab is first shown
}
