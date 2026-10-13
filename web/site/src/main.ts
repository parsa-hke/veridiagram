import "./style.css";
import { tracedSources, untracedNumbers } from "./audit";
import { loadDocs, setDocs } from "./data";
import { wirePencil } from "./cursor";
import { drawOnScroll, play, prepare } from "./draw";
import { wireRuler } from "./figures/geometry";
import { wireMotion } from "./motion";
import { renderPage } from "./page";
import { wireTooltips } from "./tooltip";
import { applyStoredTheme, wireNotes, wireRunningHead, wireTheme } from "./ui";

applyStoredTheme();
const root = document.getElementById("app")!;
const head = document.getElementById("running-head")!;
wireTheme(document.getElementById("edition") as HTMLButtonElement);
wirePencil(document.getElementById("pencil-toggle") as HTMLButtonElement | null);

async function start(): Promise<void> {
  try {
    setDocs(await loadDocs());
  } catch (e) {
    root.innerHTML = `<p class="error">The page's data could not be loaded: ${(e as Error).message}</p>`;
    return;
  }
  renderPage(root);
  wireNotes(root);
  wireRunningHead(head, root);
  const heroFig = root.querySelector(".construct-hero");
  if (heroFig) play(prepare(heroFig), 3600);
  drawOnScroll(Array.from(root.querySelectorAll(".construction-figure")).map(prepare));
  wireRuler(root);
  wireTooltips(root);
  wireMotion(root);
  const trace = new URLSearchParams(location.search).has("trace");
  if (trace || import.meta.env.DEV) {
    const bad = untracedNumbers(root);
    const { sources, missing } = tracedSources(root);
    if (bad.length || missing.length) console.warn("number trace", { untraced: bad, missing });
    if (trace) {
      document.body.classList.add("trace");
      const panel = document.createElement("aside");
      panel.className = "trace-panel";
      panel.textContent = `${sources.length} traced numbers · ${missing.length} missing · ${bad.length} untraced`;
      document.body.append(panel);
    }
  }
}

start();
