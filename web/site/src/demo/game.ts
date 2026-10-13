// Beat the Transformer (WEBSITE_PLAN.md §3.3): ten near-threshold questions, precomputed by
// web/precompute.py, so the game works without the server.

interface Item {
  image: string;
  angle_deg: number;
  parallel: boolean;
  p_parallel: number;
}
interface Game {
  tolerance_deg: number;
  band_deg: [number, number];
  items: Item[];
}

const ROUNDS = 10;

export function shuffleRounds(n: number, rounds: number, rand: () => number = Math.random): number[] {
  const idx = Array.from({ length: n }, (_, i) => i);
  for (let i = idx.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [idx[i], idx[j]] = [idx[j], idx[i]];
  }
  return idx.slice(0, Math.min(rounds, n));
}

export function gamePanel(host: HTMLElement, base = "data/"): void {
  let game: Game | null = null;
  let order: number[] = [];
  let round = 0;
  let you = 0, model = 0;
  host.innerHTML = `
<p>Ten pairs of green lines, each within a degree or two of the Ruler's tolerance. Are they parallel, as the
Ruler counts it? Answer, then see what the Ruler measured and what the Eye said.</p>
<div class="game" data-game><button type="button" class="button primary" data-start>Begin</button></div>`;
  const stage = host.querySelector<HTMLElement>("[data-game]")!;

  const load = async () => {
    if (game) return game;
    const r = await fetch(`${base}game.json`);
    if (!r.ok) throw new Error("the game could not be loaded");
    game = (await r.json()) as Game;
    return game;
  };

  const show = () => {
    const g = game!, it = g.items[order[round]];
    stage.innerHTML = `
<p class="round"><span class="smallcaps">Question</span> <span class="live">${round + 1}</span> of <span class="live">${order.length}</span></p>
<img class="game-image" src="${base}${it.image}" alt="A diagram with two lines highlighted in green" width="448" height="448">
<p>Parallel, within <span class="live">${g.tolerance_deg}°</span>?</p>
<div class="game-buttons">
  <button type="button" class="button primary" data-say="yes">Parallel</button>
  <button type="button" class="button" data-say="no">Not parallel</button>
</div>
<div class="reveal" aria-live="polite"></div>`;
    stage.querySelectorAll<HTMLButtonElement>("[data-say]").forEach((b) => b.addEventListener("click", () => answer(b.dataset.say === "yes")));
  };

  const answer = (said: boolean) => {
    const g = game!, it = g.items[order[round]];
    const eye = it.p_parallel > 0.5;
    you += Number(said === it.parallel);
    model += Number(eye === it.parallel);
    stage.querySelectorAll<HTMLButtonElement>("[data-say]").forEach((b) => (b.disabled = true));
    const mark = (ok: boolean) => `<strong class="${ok ? "right" : "wrong"}">${ok ? "right" : "wrong"}</strong>`;
    stage.querySelector(".reveal")!.innerHTML = `
<ul class="verdict-list">
  <li><span class="smallcaps">The Ruler</span> <span class="live">${it.angle_deg.toFixed(2)}°</span> from parallel:
    <strong>${it.parallel ? "parallel" : "not parallel"}</strong></li>
  <li><span class="smallcaps">The Eye</span> <span class="live">${it.p_parallel.toFixed(2)}</span>: ${eye ? "parallel" : "not parallel"}, ${mark(eye === it.parallel)}</li>
  <li><span class="smallcaps">You</span> ${said ? "parallel" : "not parallel"}, ${mark(said === it.parallel)}</li>
</ul>
<button type="button" class="button primary" data-next>${round + 1 < order.length ? "Next" : "The score"}</button>`;
    stage.querySelector<HTMLButtonElement>("[data-next]")!.addEventListener("click", () => {
      round += 1;
      if (round < order.length) show();
      else score();
    });
    stage.querySelector<HTMLButtonElement>("[data-next]")!.focus();
  };

  const score = () => {
    const n = order.length;
    const verdict = you > model ? "You beat the transformer." : you === model ? "A draw with the transformer." : "The transformer wins this time.";
    stage.innerHTML = `
<div class="scorecard">
  <p class="score-title">${verdict}</p>
  <ul class="scores">
    <li><span class="smallcaps">You</span> <span class="live">${you}</span> / <span class="live">${n}</span></li>
    <li><span class="smallcaps">The Eye</span> <span class="live">${model}</span> / <span class="live">${n}</span></li>
    <li><span class="smallcaps">The Ruler</span> <span class="live">${n}</span> / <span class="live">${n}</span>, by definition</li>
  </ul>
  <div class="game-buttons">
    <button type="button" class="button primary" data-again>Play again</button>
    <button type="button" class="button" data-share>Save the score card</button>
  </div>
</div>`;
    stage.querySelector("[data-again]")!.addEventListener("click", start);
    stage.querySelector("[data-share]")!.addEventListener("click", () => shareCard(verdict, you, model, n));
  };

  const start = async () => {
    try {
      const g = await load();
      order = shuffleRounds(g.items.length, ROUNDS);
      round = you = model = 0;
      show();
    } catch (e) {
      stage.innerHTML = `<p class="error">${(e as Error).message}</p>`;
    }
  };
  stage.querySelector("[data-start]")!.addEventListener("click", start);
}

/** Draw the score card in Byrne's colours and offer it as a PNG (or the system share sheet). */
async function shareCard(verdict: string, you: number, model: number, n: number): Promise<void> {
  const c = document.createElement("canvas");
  c.width = 1080;
  c.height = 1080;
  const g = c.getContext("2d");
  if (!g) return;
  g.fillStyle = "#F3EAD3";
  g.fillRect(0, 0, 1080, 1080);
  g.fillStyle = "#1A1A1A";
  g.textAlign = "center";
  g.font = "64px 'Libre Caslon Display', Georgia, serif";
  g.fillText("Beat the Transformer", 540, 190);
  ["#C8312B", "#E8B10E", "#1F4E8C"].forEach((col, i) => {
    g.fillStyle = col;
    g.fillRect(450 + i * 60, 230, 60, 18);
  });
  g.fillStyle = "#1A1A1A";
  g.font = "italic 44px 'Libre Caslon Text', Georgia, serif";
  g.fillText(verdict, 540, 360);
  const rows: [string, number, string][] = [["You", you, "#C8312B"], ["The Eye", model, "#00C800"], ["The Ruler", n, "#1A1A1A"]];
  rows.forEach(([label, v, col], i) => {
    const y = 500 + i * 150;
    g.font = "40px 'Libre Caslon Text', Georgia, serif";
    g.textAlign = "left";
    g.fillStyle = "#1A1A1A";
    g.fillText(label, 170, y);
    g.fillStyle = col;
    g.fillRect(430, y - 34, (480 * v) / n, 40);
    g.fillStyle = "#1A1A1A";
    g.textAlign = "right";
    g.fillText(`${v} / ${n}`, 910, y);
  });
  g.textAlign = "center";
  g.font = "28px 'Libre Caslon Text', Georgia, serif";
  g.fillText("Colour, not Letters", 540, 1000);
  const blob = await new Promise<Blob | null>((r) => c.toBlob(r, "image/png"));
  if (!blob) return;
  const file = new File([blob], "beat-the-transformer.png", { type: "image/png" });
  const nav = navigator as Navigator & { canShare?: (d: ShareData) => boolean };
  if (nav.canShare?.({ files: [file] })) {
    try {
      await nav.share({ files: [file], title: "Beat the Transformer" });
      return;
    } catch {
      /* fall through to a download */
    }
  }
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = file.name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 2000);
}
