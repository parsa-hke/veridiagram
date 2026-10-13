// The reader's instrument: a Byrne pencil (day edition) or a stick of chalk (night edition) in place
// of the mouse pointer, leaving a short trace that fades. Only for fine pointers (mouse, trackpad);
// switched off by the "Plain cursor" button (remembered); no trace under prefers-reduced-motion.

export interface TracePoint {
  x: number;
  y: number;
  t: number;
}

export const TRACE_LIFE_MS = 650;

/** Drop points older than the trace's life; keep at most `max` (newest). */
export function pruneTrace(pts: TracePoint[], now: number, life = TRACE_LIFE_MS, max = 90): TracePoint[] {
  const kept = pts.filter((p) => now - p.t < life);
  return kept.length > max ? kept.slice(kept.length - max) : kept;
}

const KEY = "pencil";
const TIP = { x: 3, y: 33 };                       // the drawing tip inside the 36 x 36 cursor box

const PENCIL = `
<svg viewBox="0 0 36 36" width="36" height="36" aria-hidden="true">
  <g class="pencil-day" transform="translate(${TIP.x},${TIP.y}) rotate(-45)">
    <polygon class="p-wood" points="0,0 9,-3.6 9,3.6" />
    <polygon class="p-lead" points="0,0 3.4,-1.36 3.4,1.36" />
    <rect class="p-body" x="9" y="-3.6" width="22" height="7.2" />
    <rect class="p-stripe" x="9" y="-1" width="22" height="2" />
    <rect class="p-ferrule" x="31" y="-3.8" width="4" height="7.6" />
    <rect class="p-eraser" x="35" y="-3.6" width="4.5" height="7.2" rx="1.2" />
  </g>
  <g class="pencil-night" transform="translate(${TIP.x},${TIP.y}) rotate(-45)">
    <rect class="p-chalk" x="0" y="-3.4" width="30" height="6.8" rx="2.4" />
    <rect class="p-chalk-shade" x="0" y="1" width="30" height="2.4" rx="1.2" />
  </g>
</svg>`;

function stored(): boolean {
  try {
    return localStorage.getItem(KEY) !== "off";
  } catch {
    return true;
  }
}

export function wirePencil(toggle: HTMLButtonElement | null): void {
  const fine = typeof matchMedia === "function" && matchMedia("(pointer: fine)").matches;
  if (!fine) {
    toggle?.remove();
    return;
  }
  const calm = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const root = document.documentElement;
  const pencil = document.createElement("div");
  pencil.className = "pencil-cursor";
  pencil.innerHTML = PENCIL;
  const canvas = document.createElement("canvas");
  canvas.className = "pencil-trace";
  document.body.append(canvas, pencil);
  const ctx = canvas.getContext("2d");

  let on = stored();
  let pts: TracePoint[] = [];
  let x = -100, y = -100, raf = 0, pressed = false;

  const size = () => {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.round(innerWidth * dpr);
    canvas.height = Math.round(innerHeight * dpr);
    ctx?.setTransform(dpr, 0, 0, dpr, 0, 0);
  };
  size();
  window.addEventListener("resize", size);

  const inkColour = () => getComputedStyle(root).getPropertyValue("--ink").trim() || "#1a1a1a";
  const night = () => getComputedStyle(root).colorScheme.includes("dark");

  const frame = () => {
    raf = 0;
    pencil.style.transform = `translate(${x - TIP.x}px, ${y - TIP.y}px)`;
    if (!ctx) return;
    const now = performance.now();
    pts = pruneTrace(pts, now);
    ctx.clearRect(0, 0, innerWidth, innerHeight);
    if (pts.length > 1) {
      const colour = inkColour();
      const chalk = night();
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.strokeStyle = colour;
      for (let i = 1; i < pts.length; i++) {
        const a = pts[i - 1], b = pts[i];
        const life = 1 - (now - b.t) / TRACE_LIFE_MS;
        ctx.globalAlpha = Math.max(0, life) * (chalk ? 0.5 : 0.62);
        ctx.lineWidth = (pressed ? 2.6 : 1.5) * (0.35 + 0.65 * life) * (chalk ? 1.6 : 1);
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        ctx.stroke();
        if (chalk && i % 3 === 0) {                       // chalk dust: a few grains beside the line
          ctx.globalAlpha *= 0.6;
          ctx.fillStyle = colour;
          ctx.fillRect(b.x + ((i * 7) % 5) - 2, b.y + ((i * 11) % 5) - 2, 1.2, 1.2);
        }
      }
      ctx.globalAlpha = 1;
    }
    if (pts.length) raf = requestAnimationFrame(frame);
  };
  const queue = () => {
    if (!raf) raf = requestAnimationFrame(frame);
  };

  const interactive = "a, button, summary, [role=slider], [data-tip], .ruler-handle, label, input, select, textarea";
  const move = (e: PointerEvent) => {
    if (!on || e.pointerType !== "mouse" && e.pointerType !== "pen") return;
    x = e.clientX;
    y = e.clientY;
    if (!calm) {
      const now = performance.now();
      const evs = typeof e.getCoalescedEvents === "function" ? e.getCoalescedEvents() : [];
      (evs.length ? evs : [e]).forEach((c) => pts.push({ x: c.clientX, y: c.clientY, t: now }));
    }
    pencil.classList.toggle("over-control", !!(e.target as Element).closest?.(interactive));
    pencil.classList.remove("away");
    queue();
  };

  const apply = () => {
    root.classList.toggle("pencil", on);
    pencil.hidden = !on;
    canvas.hidden = !on || calm;
    if (toggle) {
      toggle.textContent = on ? "Plain cursor" : "Pencil";
      toggle.setAttribute("aria-pressed", String(!on));
      toggle.title = on ? "Use the ordinary mouse pointer" : "Draw with the pencil cursor";
    }
    if (!on) {
      pts = [];
      ctx?.clearRect(0, 0, innerWidth, innerHeight);
    }
  };

  window.addEventListener("pointermove", move, { passive: true });
  window.addEventListener("pointerdown", () => {
    pressed = true;
    pencil.classList.add("pressed");
  });
  window.addEventListener("pointerup", () => {
    pressed = false;
    pencil.classList.remove("pressed");
  });
  document.addEventListener("pointerleave", () => pencil.classList.add("away"));
  document.documentElement.addEventListener("mouseleave", () => pencil.classList.add("away"));
  toggle?.addEventListener("click", () => {
    on = !on;
    try {
      localStorage.setItem(KEY, on ? "on" : "off");
    } catch {
      /* private mode: lasts for this visit */
    }
    apply();
  });
  apply();
}
