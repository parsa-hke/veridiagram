import "./style.css";
import { tracedSources, untracedNumbers } from "./audit";
import { loadDocs, setDocs } from "./data";
import { renderPage } from "./page";

const root = document.getElementById("app")!;

async function start(): Promise<void> {
  try {
    setDocs(await loadDocs());
  } catch (e) {
    root.innerHTML = `<p class="error">The page's data could not be loaded: ${(e as Error).message}</p>`;
    return;
  }
  renderPage(root);
  const trace = import.meta.env.DEV || new URLSearchParams(location.search).has("trace");
  if (trace) {
    document.body.classList.add("trace");
    const bad = untracedNumbers(root);
    const { sources, missing } = tracedSources(root);
    const panel = document.createElement("aside");
    panel.className = "trace-panel";
    panel.textContent = `${sources.length} traced numbers · ${missing.length} missing · ${bad.length} untraced`;
    document.body.append(panel);
    if (bad.length || missing.length) console.warn("number trace", { untraced: bad, missing });
  }
}

start();
