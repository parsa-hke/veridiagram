import { afterEach, describe, expect, it, vi } from "vitest";
import { drawOnScroll, play, prepare } from "./draw";
import { rulerVerdict } from "./figures/geometry";

function figure(): HTMLElement {
  const fig = document.createElement("figure");
  fig.innerHTML = `<svg><line data-step="0" pathLength="1" /><circle data-step="1" pathLength="1" cx="5" cy="5" r="4" />
    <text data-step="2" data-kind="fade">x</text></svg><figcaption><span data-step="2" data-kind="fade">c</span></figcaption>`;
  document.body.append(fig);
  return fig;
}

const reduce = (on: boolean) =>
  vi.stubGlobal("matchMedia", (q: string) => ({ matches: on && q.includes("reduce"), media: q, addEventListener: () => {} }));

afterEach(() => {
  vi.unstubAllGlobals();
  document.body.innerHTML = "";
});

describe("drawing engine", () => {
  it("walks through the steps in order", () => {
    const fig = figure();
    const c = prepare(fig);
    const [ln, circ, txt] = Array.from(fig.querySelectorAll<SVGElement>("svg > *"));
    const cap = fig.querySelector<HTMLElement>("figcaption span")!;
    expect(c.steps).toBe(3);
    c.set(0);
    expect(ln.style.strokeDashoffset).toBe("1");
    expect(ln.style.opacity).toBe("0");
    expect(txt.style.opacity).toBe("0");
    c.set(0.5);                                            // step 0 done, step 1 half way, step 2 not started
    expect(ln.style.strokeDashoffset).toBe("0");
    expect(Number(circ.style.strokeDashoffset)).toBeCloseTo(0.5);
    expect(cap.style.opacity).toBe("0");
    const tool = fig.querySelector<SVGGElement>(".instrument")!;     // the compass is out while the circle is drawn
    expect(tool.style.display).toBe("");
    expect((tool.querySelector(".compass") as SVGGElement).style.display).toBe("");
    c.set(1);
    expect(tool.style.display).toBe("none");                          // and put away when the figure is done
    expect(circ.style.strokeDashoffset).toBe("0");
    expect(cap.style.opacity).toBe("1");
  });

  it("shows finished figures under prefers-reduced-motion", () => {
    reduce(true);
    const a = prepare(figure()), b = prepare(figure());
    a.set(0);
    b.set(0);
    drawOnScroll([a]);
    play(b, 1000);
    for (const fig of [a.el, b.el]) {
      fig.querySelectorAll<SVGElement>("[data-step]").forEach((el) => {
        if (el.dataset.kind === "fade") expect(el.style.opacity).toBe("1");
        else expect(el.style.strokeDashoffset).toBe("0");
      });
    }
  });
});

describe("the Ruler", () => {
  it("decides parallel and equal length as the verifier states them", () => {
    const deg = (d: number) => (d * Math.PI) / 180;
    const A: [[number, number], [number, number]] = [[0, 0], [100, 0]];
    const seg = (angle: number, length: number): [[number, number], [number, number]] =>
      [[0, 50], [length * Math.cos(deg(angle)), 50 + length * Math.sin(deg(angle))]];
    expect(rulerVerdict(A, seg(3.6, 100), 3.7, 0.1).parallel).toBe(true);
    expect(rulerVerdict(A, seg(3.8, 100), 3.7, 0.1).parallel).toBe(false);
    expect(rulerVerdict(A, seg(180 - 2, 100), 3.7, 0.1).parallel).toBe(true);   // direction does not matter
    expect(rulerVerdict(A, seg(0, 91), 3.7, 0.1).equal).toBe(true);              // 9% shorter
    expect(rulerVerdict(A, seg(0, 89), 3.7, 0.1).equal).toBe(false);             // 11% shorter
    expect(rulerVerdict(A, seg(2, 100), 3.7, 0.1).angleDeg).toBeCloseTo(2, 6);
  });
});
