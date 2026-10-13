// Book IV: Constructions (WEBSITE_PLAN.md §3): Construct, Bring your own, Beat the Transformer, Gallery.
// The server is asked only by Construct and Upload; the game and the gallery are precomputed, so they
// work while the instrument warms up or when it is offline.

import * as api from "./api";
import { constructPanel, type Panel } from "./construct";
import { galleryPanel } from "./gallery";
import { gamePanel } from "./game";

const TABS = [
  { id: "construct", label: "Construct" },
  { id: "upload", label: "Bring your own" },
  { id: "game", label: "Beat the Transformer" },
  { id: "gallery", label: "Gallery" },
];

export function book4(): string {
  return `
<section class="book" id="book-4" data-running="Book IV · Constructions">
  <h2><span class="book-no">Book IV</span> Constructions</h2>
  <p class="opening">Here you draw, and the machine judges. It cannot find points and lines by itself, so you
  supply them: draw a figure with compass and straightedge, or mark the parts of your own diagram, then
  point at two of them and ask. Three witnesses answer: the Ruler, which measures; the Eye, which only
  looks; and the picture the Eye was shown, with your question painted on it in green.</p>
  <p class="instrument-status" data-status role="status"><span class="dot"></span> <span data-status-text>Checking the instrument…</span></p>
  <div class="tabs" role="tablist" aria-label="Ways to try the model">
    ${TABS.map((t, i) => `<button type="button" role="tab" id="tab-${t.id}" aria-controls="panel-${t.id}"
      aria-selected="${i === 0}" tabindex="${i === 0 ? 0 : -1}">${t.label}</button>`).join("")}
  </div>
  ${TABS.map((t, i) => `<div class="tab-panel" role="tabpanel" id="panel-${t.id}" aria-labelledby="tab-${t.id}"${i ? " hidden" : ""}></div>`).join("")}
</section>`;
}

export function wireBook4(root: Element): void {
  const section = root.querySelector<HTMLElement>("#book-4");
  if (!section) return;
  const panel = (id: string) => section.querySelector<HTMLElement>(`#panel-${id}`)!;
  const panels: Panel[] = [constructPanel(panel("construct")), constructPanel(panel("upload"), true)];
  gamePanel(panel("game"));
  const loadGallery = galleryPanel(panel("gallery"));

  const tabs = Array.from(section.querySelectorAll<HTMLButtonElement>("[role=tab]"));
  const select = (b: HTMLButtonElement) => {
    tabs.forEach((t) => {
      const on = t === b;
      t.setAttribute("aria-selected", String(on));
      t.tabIndex = on ? 0 : -1;
      section.querySelector<HTMLElement>(`#${t.getAttribute("aria-controls")}`)!.hidden = !on;
    });
    if (b.id === "tab-gallery") loadGallery();
  };
  tabs.forEach((t, i) => {
    t.addEventListener("click", () => select(t));
    t.addEventListener("keydown", (e) => {
      const k = e.key === "ArrowRight" ? 1 : e.key === "ArrowLeft" ? -1 : 0;
      if (!k) return;
      const n = tabs[(i + k + tabs.length) % tabs.length];
      select(n);
      n.focus();
    });
  });

  const status = section.querySelector<HTMLElement>("[data-status]")!;
  const text = section.querySelector<HTMLElement>("[data-status-text]")!;
  api.watchHealth((s, h) => {
    status.dataset.state = s;
    text.textContent = s === "ready" ? `The instrument is ready (${h?.model ?? "the model"}, on ${h?.device ?? "?"}).`
      : s === "warming" ? "The instrument is warming up: the first visit after a quiet spell wakes it, which takes a minute. Meanwhile, try the game or the gallery."
      : s === "offline" ? "The instrument is offline right now. The game and the gallery work without it."
      : "Checking the instrument…";
    panels.forEach((p) => p.setStatus(s));
  });
}
