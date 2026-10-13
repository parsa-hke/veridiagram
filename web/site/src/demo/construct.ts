// Construct (a blank page) and Upload (the visitor's picture): draw, point at two parts, ask.

import * as api from "./api";
import { Editor, inkSnapper, type Tool } from "./editor";
import { readout, witnesses } from "./witnesses";

type XY = [number, number];

const TOOLS: { tool: Tool; label: string; hint: string; helper?: boolean }[] = [
  { tool: "point", label: "Point", hint: "Click to place a point; points are lettered for you." },
  { tool: "segment", label: "Segment", hint: "Click two points (new or existing) to join them." },
  { tool: "circle", label: "Circle", hint: "Click the centre, then a point on the circle." },
  { tool: "parallel", label: "Parallel", hint: "Click a segment, then a point: a parallel through it.", helper: true },
  { tool: "perpendicular", label: "Perpendicular", hint: "Click a segment, then a point: a perpendicular through it.", helper: true },
  { tool: "midpoint", label: "Midpoint", hint: "Click a segment to mark its midpoint.", helper: true },
  { tool: "oncircle", label: "On circle", hint: "Click a circle to put a point on it.", helper: true },
  { tool: "ask", label: "Point at", hint: "Click two parts to ask about: two segments, a point and a circle, a segment and a circle, or a point and a segment." },
];

const EXAMPLES: Record<string, { points: Record<string, XY>; segments: [string, string][]; circles?: { center: string; through: string }[] }> = {
  "A trapezium": {
    points: { A: [120, 120], B: [480, 128], C: [530, 330], D: [70, 322] },
    segments: [["A", "B"], ["B", "C"], ["C", "D"], ["D", "A"], ["A", "C"]],
  },
  "A tangent": {
    points: { O: [260, 210], A: [260, 90], P: [80, 90], Q: [470, 90], B: [370, 270] },
    segments: [["P", "Q"], ["O", "A"], ["O", "B"]],
    circles: [{ center: "O", through: "A" }],
  },
  "A right triangle": {
    points: { A: [140, 330], B: [460, 330], C: [140, 110], M: [300, 220] },
    segments: [["A", "B"], ["A", "C"], ["B", "C"], ["A", "M"]],
  },
};

const thinking = `<div class="thinking" role="status"><svg viewBox="0 0 60 60" aria-hidden="true"><circle cx="30" cy="30" r="20" class="ink-line thin thinking-arc" pathLength="1" /></svg>
  <span>The instrument is looking…</span></div>`;

export interface Panel {
  setStatus(s: api.Status): void;
}

function panelHTML(id: string, upload: boolean): string {
  const tools = TOOLS.filter((t) => !upload || !t.helper);
  return `
<div class="tool-row" role="toolbar" aria-label="Drawing tools">
  ${tools.map((t) => `<button type="button" class="tool${t.tool === "point" ? " active" : ""}" data-tool="${t.tool}" aria-pressed="${t.tool === "point"}">${t.label}</button>`).join("")}
</div>
<div class="tool-row secondary">
  ${upload ? "" : `<label class="wobble">Wobble <input type="range" min="0" max="6" step="0.5" value="0" data-wobble> <span class="live" data-wobble-out>0</span>° / %</label>
  <select data-example aria-label="Load an example figure"><option value="">Examples…</option>${Object.keys(EXAMPLES).map((k) => `<option>${k}</option>`).join("")}</select>`}
  <button type="button" class="tool plain" data-undo>Undo</button>
  <button type="button" class="tool plain" data-clear>Clear</button>
</div>
${upload ? `<div class="dropzone" data-drop><input type="file" accept="image/png,image/jpeg" id="${id}-file" data-file>
  <label for="${id}-file"><strong>Choose a diagram</strong> or drop it here (PNG or JPEG, up to 5 MB).</label>
  <p class="witness-note">Your picture is processed in memory and is not stored. The model was trained on clean printed
  textbook diagrams; photographs of hand-drawn sketches are outside what it knows.</p></div>` : ""}
<div class="editor-host" data-host ${upload ? "hidden" : ""}></div>
<p class="tool-hint" data-hint>${TOOLS[0].hint}</p>
<div class="ask-row">
  <button type="button" class="button primary" data-ask disabled>Ask</button>
  <button type="button" class="button" data-read disabled>Read the whole figure</button>
  <span class="instrument-note" data-note></span>
</div>
<div class="answer" data-answer aria-live="polite"></div>`;
}

export function constructPanel(host: HTMLElement, upload = false): Panel {
  host.innerHTML = panelHTML(host.id, upload);
  const $ = <T extends Element>(sel: string) => host.querySelector<T>(sel)!;
  const answer = $<HTMLElement>("[data-answer]");
  const note = $<HTMLElement>("[data-note]");
  let status: api.Status = "checking";
  let editor: Editor | null = null;
  let image: string | null = null;                    // uploads: base64 PNG/JPEG
  let busy = false;

  const body = (): api.Body | null => {
    if (!editor) return null;
    const f = editor.figure();
    return upload ? (image ? { image, primitives: f } : null) : { construction: f };
  };
  const circleNames = () => editor?.circles.map((c) => c.center) ?? [];

  const refresh = () => {
    const ready = status === "ready" && !busy && !!editor;
    $<HTMLButtonElement>("[data-ask]").disabled = !(ready && editor!.query);
    $<HTMLButtonElement>("[data-read]").disabled = !(ready && Object.keys(editor!.points).length >= 2);
    note.textContent = status === "ready" ? (editor?.tool === "ask" && !editor.query ? "Point at two parts to ask." : "")
      : status === "offline" ? "The instrument is offline; the game and the gallery still work."
      : "The instrument is warming up; meanwhile, try the game or the gallery.";
  };

  const makeEditor = (opts: ConstructorParameters<typeof Editor>[1]) => {
    editor = new Editor($("[data-host]"), { ...opts, onChange: refresh });
    const active = host.querySelector<HTMLButtonElement>(".tool.active");
    editor.setTool((active?.dataset.tool as Tool) ?? "point");
    refresh();
  };
  if (!upload) makeEditor({ width: 600, height: 420 });

  host.querySelectorAll<HTMLButtonElement>("[data-tool]").forEach((b) =>
    b.addEventListener("click", () => {
      host.querySelectorAll<HTMLButtonElement>("[data-tool]").forEach((x) => {
        x.classList.toggle("active", x === b);
        x.setAttribute("aria-pressed", String(x === b));
      });
      editor?.setTool(b.dataset.tool as Tool);
      $("[data-hint]").textContent = TOOLS.find((t) => t.tool === b.dataset.tool)!.hint;
      refresh();
    }));
  host.querySelector("[data-undo]")?.addEventListener("click", () => editor?.undo());
  host.querySelector("[data-clear]")?.addEventListener("click", () => {
    editor?.clear();
    answer.innerHTML = "";
  });
  const wob = host.querySelector<HTMLInputElement>("[data-wobble]");
  wob?.addEventListener("input", () => {
    if (editor) editor.wobble = Number(wob.value);
    host.querySelector("[data-wobble-out]")!.textContent = wob.value;
  });
  host.querySelector<HTMLSelectElement>("[data-example]")?.addEventListener("change", (e) => {
    const sel = e.target as HTMLSelectElement;
    if (sel.value && editor) editor.load(EXAMPLES[sel.value]);
    answer.innerHTML = "";
    sel.value = "";
  });

  if (upload) {
    const file = $<HTMLInputElement>("[data-file]");
    const drop = $<HTMLElement>("[data-drop]");
    const take = async (f: File | undefined) => {
      if (!f) return;
      if (!/^image\/(png|jpeg)$/.test(f.type) || f.size > 5 * 1024 * 1024) {
        answer.innerHTML = `<p class="error">Please choose a PNG or JPEG of at most 5 MB.</p>`;
        return;
      }
      const { url, width, height, canvas } = await shrink(f, 1600);
      image = url.split(",")[1];
      $<HTMLElement>("[data-host]").hidden = false;
      makeEditor({ width, height, image: { href: url, width, height }, snapInk: inkSnapper(canvas) });
      answer.innerHTML = "";
    };
    file.addEventListener("change", () => take(file.files?.[0]));
    drop.addEventListener("dragover", (e) => {
      e.preventDefault();
      drop.classList.add("over");
    });
    drop.addEventListener("dragleave", () => drop.classList.remove("over"));
    drop.addEventListener("drop", (e) => {
      e.preventDefault();
      drop.classList.remove("over");
      take(e.dataTransfer?.files?.[0]);
    });
  }

  const fail = (e: unknown) => {
    answer.innerHTML = `<p class="error">${e instanceof Error ? e.message : String(e)}</p>`;
  };

  $<HTMLButtonElement>("[data-ask]").addEventListener("click", async () => {
    const b = body(), q = editor?.query;
    if (!b || !q) return;
    busy = true;
    refresh();
    answer.innerHTML = thinking;
    try {
      const r = await api.ask(b, [q]);
      answer.innerHTML = witnesses(r.results[0], circleNames());
    } catch (e) {
      fail(e);
    } finally {
      busy = false;
      refresh();
    }
  });

  $<HTMLButtonElement>("[data-read]").addEventListener("click", async () => {
    const b = body();
    if (!b) return;
    busy = true;
    refresh();
    const chunk = 4;
    const answers: { kind: string; refs: api.Ref[]; probs: api.Result["probs"] }[] = [];
    const results: api.Result[] = [];
    try {
      let total = Infinity, asked = 0, all = 0;
      for (let off = 0; off < total; off += chunk) {
        answer.innerHTML = `<div class="progress" role="progressbar" aria-valuemin="0" aria-valuemax="${Number.isFinite(total) ? total : 0}"
          aria-valuenow="${off}"><span style="--p:${Number.isFinite(total) ? off / total : 0}"></span></div>
          <p class="instrument-note">Reading the figure: question <span class="live">${off}</span> of
          <span class="live">${Number.isFinite(total) ? total : "…"}</span>.</p>`;
        const r = await api.readChunk(b, off, chunk);
        total = r.queries_asked;
        asked = r.queries_asked;
        all = r.queries_total;
        answers.push(...r.results.map((x) => ({ kind: x.kind, refs: x.refs, probs: x.probs })));
        results.push(...r.results);
      }
      const sum = await api.closure(b, answers);
      answer.innerHTML = readout(sum, asked, all, circleNames(), results);
    } catch (e) {
      fail(e);
    } finally {
      busy = false;
      refresh();
    }
  });

  return {
    setStatus(s) {
      status = s;
      refresh();
    },
  };
}

/** Read an image file and shrink it to at most `max` px on its long side (client-side). */
async function shrink(f: File, max: number): Promise<{ url: string; width: number; height: number; canvas: HTMLCanvasElement }> {
  const src = URL.createObjectURL(f);
  try {
    const img = new Image();
    img.src = src;
    await img.decode();
    const k = Math.min(1, max / Math.max(img.naturalWidth, img.naturalHeight));
    const c = document.createElement("canvas");
    c.width = Math.round(img.naturalWidth * k);
    c.height = Math.round(img.naturalHeight * k);
    const ctx = c.getContext("2d")!;
    ctx.fillStyle = "#fff";
    ctx.fillRect(0, 0, c.width, c.height);
    ctx.drawImage(img, 0, 0, c.width, c.height);
    let url = c.toDataURL("image/png");
    if (url.length > 6_500_000) url = c.toDataURL("image/jpeg", 0.92);
    return { url, width: c.width, height: c.height, canvas: c };
  } finally {
    URL.revokeObjectURL(src);
  }
}
