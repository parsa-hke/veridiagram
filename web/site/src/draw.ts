// The drawing engine (WEBSITE_PLAN.md §1.2, §1.3): every figure is a construction drawn stroke by
// stroke, compass first, then straightedge, then fill. An SVG with class "construction" holds
// elements carrying data-step="k" (0-based order) and data-kind="stroke" (default: drawn along its
// length; strokes carry pathLength="1") or "fade" (text, dots, fills: faded in). Progress p in [0, 1]
// walks through the steps; scroll position drives it, or a clock for the frontispiece.
// prefers-reduced-motion: every figure is set finished and nothing moves.

export interface Construction {
  el: Element;                 // the figure (its SVG and caption); its position drives scroll progress
  steps: number;
  set(p: number): void;
}

const clamp = (x: number) => (x < 0 ? 0 : x > 1 ? 1 : x);

/** Which of several strokes drawn at once gets the instrument: the largest circle, else the latest line. */
const weight = (el: Element) => (el.tagName.toLowerCase() === "circle" ? 1 + Number(el.getAttribute("r")) : 0);

export function prepare(fig: Element): Construction {
  const items = Array.from(fig.querySelectorAll<SVGElement | HTMLElement>("[data-step]")).map((el) => ({
    el,
    step: Number(el.dataset.step),
    fade: el.dataset.kind === "fade",
  }));
  const steps = items.reduce((m, i) => Math.max(m, i.step + 1), 1);
  const tools = new Map<SVGSVGElement, Instrument>();
  const set = (p: number) => {
    const t = clamp(p) * steps;
    let active: { el: SVGElement; local: number } | null = null;
    for (const { el, step, fade } of items) {
      const local = clamp(t - step);
      if (!fade && local > 0 && local < 1 && el instanceof SVGElement && !el.closest(".machine-trace")
          && (!active || weight(el) >= weight(active.el))) active = { el, local };
      if (fade) {
        el.style.opacity = String(local);
      } else {
        el.style.strokeDasharray = "1 1";
        el.style.strokeDashoffset = String(1 - local);
        el.style.opacity = local > 0 ? "" : "0";      // round caps would show a dot at zero length
      }
    }
    tools.forEach((tool) => tool.hide());
    if (active) {
      const svg = active.el.ownerSVGElement;
      if (svg) {
        if (!tools.has(svg)) tools.set(svg, instrument(svg));
        tools.get(svg)!.show(active.el, active.local);
      }
    }
  };
  return { el: fig, steps, set };
}

// ------------------------------------------------------------------ the instruments

interface Instrument {
  show(el: SVGElement, local: number): void;
  hide(): void;
}

const NS = "http://www.w3.org/2000/svg";

/** A compass that sweeps circles (needle at the centre, pencil leg on the arc) and a small pencil
 * that rides the end of every other stroke while it is being drawn. */
function instrument(svg: SVGSVGElement): Instrument {
  const g = document.createElementNS(NS, "g");
  g.setAttribute("class", "instrument");
  g.setAttribute("aria-hidden", "true");
  g.innerHTML = `<g class="compass"><line class="leg" /><line class="leg" /><circle class="hinge" r="2.6" />
      <line class="grip" /></g>
    <g class="nib"><polygon class="nib-wood" points="0,0 7,-2.8 7,2.8" /><polygon class="nib-lead" points="0,0 2.6,-1 2.6,1" />
      <rect class="nib-body" x="7" y="-2.8" width="15" height="5.6" /></g>`;
  svg.append(g);
  const [compass, nib] = Array.from(g.children) as SVGGElement[];
  const [leg1, leg2, hinge, grip] = Array.from(compass.children) as SVGElement[];
  const set = (el: Element, a: Record<string, number>) => Object.entries(a).forEach(([k, v]) => el.setAttribute(k, String(v)));
  const lengths = new WeakMap<Element, number>();
  return {
    hide() {
      g.style.display = "none";
    },
    show(el, local) {
      g.style.display = "";
      if (el.tagName.toLowerCase() === "circle") {
        const [cx, cy, r] = ["cx", "cy", "r"].map((k) => Number(el.getAttribute(k)));
        const th = local * 2 * Math.PI;                    // circles are drawn from 3 o'clock, clockwise
        const px = cx + r * Math.cos(th), py = cy + r * Math.sin(th);
        const mx = (cx + px) / 2, my = (cy + py) / 2;
        const h = Math.max(18, r * 0.75);                  // the hinge stands above the two points
        const hx = mx, hy = my - h;
        set(leg1, { x1: hx, y1: hy, x2: cx, y2: cy });
        set(leg2, { x1: hx, y1: hy, x2: px, y2: py });
        set(hinge, { cx: hx, cy: hy });
        set(grip, { x1: hx, y1: hy, x2: hx, y2: hy - 9 });
        compass.style.display = "";
        nib.style.display = "none";
        return;
      }
      const geo = el as unknown as SVGGeometryElement;
      if (typeof geo.getTotalLength !== "function") {
        g.style.display = "none";
        return;
      }
      let L = lengths.get(el);
      if (L === undefined) {
        L = geo.getTotalLength();
        lengths.set(el, L);
      }
      const pt = geo.getPointAtLength(local * L);
      nib.setAttribute("transform", `translate(${pt.x},${pt.y}) rotate(-45)`);
      nib.style.display = "";
      compass.style.display = "none";
    },
  };
}

export function reducedMotion(): boolean {
  return typeof matchMedia === "function" && matchMedia("(prefers-reduced-motion: reduce)").matches;
}

/** Figures drawn as they scroll through the viewport: started when the figure's top passes 88% of
 * the viewport height, finished at 30%. Scrolling back undraws them. */
export function drawOnScroll(cons: Construction[]): void {
  if (!cons.length) return;
  if (reducedMotion()) {
    cons.forEach((c) => c.set(1));
    return;
  }
  let queued = false;
  const update = () => {
    queued = false;
    const vh = window.innerHeight;
    const start = 0.88 * vh;
    const end = 0.3 * vh;
    for (const c of cons) {
      const top = c.el.getBoundingClientRect().top;
      c.set((start - top) / (start - end));
    }
  };
  const queue = () => {
    if (!queued) {
      queued = true;
      requestAnimationFrame(update);
    }
  };
  window.addEventListener("scroll", queue, { passive: true });
  window.addEventListener("resize", queue);
  update();
}

/** Draw a construction over `ms` milliseconds (the frontispiece), eased. */
export function play(c: Construction, ms: number, delay = 250): void {
  if (reducedMotion()) {
    c.set(1);
    return;
  }
  c.set(0);
  const t0 = performance.now() + delay;
  const ease = (x: number) => (x < 0.5 ? 2 * x * x : 1 - (-2 * x + 2) ** 2 / 2);
  const frame = (now: number) => {
    const x = clamp((now - t0) / ms);
    c.set(ease(x));
    if (x < 1) requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);
}
