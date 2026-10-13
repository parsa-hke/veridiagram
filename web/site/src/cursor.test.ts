import { afterEach, describe, expect, it, vi } from "vitest";
import { TRACE_LIFE_MS, pruneTrace, wirePencil } from "./cursor";
import { REVEAL, wireMotion } from "./motion";

afterEach(() => {
  vi.unstubAllGlobals();
  document.documentElement.className = "";
  document.body.innerHTML = "";
});

const media = (fine: boolean, reduce = false) =>
  vi.stubGlobal("matchMedia", (q: string) => ({
    matches: (q.includes("fine") && fine) || (q.includes("reduce") && reduce), media: q, addEventListener: () => {},
  }));

describe("pencil trace", () => {
  it("keeps only recent points, newest last, capped", () => {
    const pts = [0, 100, 600, 700].map((t, i) => ({ x: i, y: i, t }));
    expect(pruneTrace(pts, 700).map((p) => p.t)).toEqual([100, 600, 700]);
    expect(pruneTrace(pts, 700 + TRACE_LIFE_MS)).toEqual([]);
    const many = Array.from({ length: 200 }, (_, i) => ({ x: i, y: 0, t: 1000 }));
    const kept = pruneTrace(many, 1000, TRACE_LIFE_MS, 90);
    expect(kept).toHaveLength(90);
    expect(kept[89].x).toBe(199);
  });

  it("is not installed on touch screens, and can be switched off", () => {
    media(false);
    const t1 = document.createElement("button");
    document.body.append(t1);
    wirePencil(t1);
    expect(document.documentElement.classList.contains("pencil")).toBe(false);
    expect(t1.isConnected).toBe(false);                               // no toggle without a mouse
    media(true);
    const t2 = document.createElement("button");
    document.body.append(t2);
    wirePencil(t2);
    expect(document.documentElement.classList.contains("pencil")).toBe(true);
    t2.click();
    expect(document.documentElement.classList.contains("pencil")).toBe(false);
    expect(localStorage.getItem("pencil")).toBe("off");
  });
});

describe("page motion", () => {
  it("does nothing under prefers-reduced-motion, so nothing starts hidden", () => {
    media(true, true);
    wireMotion(document.body);
    expect(document.documentElement.classList.contains("motion")).toBe(false);
    expect(REVEAL).toContain(".qed");
  });
});
