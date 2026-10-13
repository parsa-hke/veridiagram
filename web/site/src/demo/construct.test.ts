// Drawing through the real Construct panel, as clicks reach it (regression: a click in the drawing area
// used to re-select the tool and cancel a half-drawn segment).
import { describe, expect, it } from "vitest";
import { setDocs } from "../data";
import { constructPanel } from "./construct";

setDocs({ results: {}, scaling: {}, pseudo_labels: {}, facts: { meta: { api_url: null } }, figures: {} });

function panel() {
  const host = document.createElement("div");
  host.id = "panel-test";
  document.body.append(host);
  constructPanel(host);
  const svg = host.querySelector<SVGSVGElement>(".editor-svg")!;
  svg.getBoundingClientRect = () => ({ left: 0, top: 0, width: 600, height: 420, right: 600, bottom: 420, x: 0, y: 0, toJSON() {} });
  const click = (x: number, y: number) => svg.dispatchEvent(new MouseEvent("click", { bubbles: true, clientX: x, clientY: y }));
  const tool = (t: string) => host.querySelector<HTMLButtonElement>(`button[data-tool="${t}"]`)!.click();
  return { host, svg, click, tool };
}

describe("Construct", () => {
  it("draws a segment in two clicks, then a second one", () => {
    const { svg, click, tool } = panel();
    expect(svg.querySelector(".ed-empty")).not.toBeNull();         // the empty page says what to do
    tool("segment");
    click(100, 100);
    expect(svg.querySelector(".ed-point.pending")).not.toBeNull();  // A waits for its partner
    click(400, 110);
    expect(svg.querySelectorAll(".ed-seg")).toHaveLength(1);
    expect(svg.querySelector(".ed-empty")).toBeNull();
    click(100, 300);
    click(400, 310);
    expect(svg.querySelectorAll(".ed-seg")).toHaveLength(2);
    expect([...svg.querySelectorAll(".ed-label")].map((t) => t.textContent)).toEqual(["A", "B", "C", "D"]);
  });

  it("points at two segments and enables Ask once the instrument is ready", () => {
    const host = document.createElement("div");
    document.body.append(host);
    const p = constructPanel(host);
    const svg = host.querySelector<SVGSVGElement>(".editor-svg")!;
    svg.getBoundingClientRect = () => ({ left: 0, top: 0, width: 600, height: 420, right: 600, bottom: 420, x: 0, y: 0, toJSON() {} });
    const click = (x: number, y: number) => svg.dispatchEvent(new MouseEvent("click", { bubbles: true, clientX: x, clientY: y }));
    host.querySelector<HTMLButtonElement>('button[data-tool="segment"]')!.click();
    click(100, 100); click(400, 110); click(100, 300); click(400, 310);
    host.querySelector<HTMLButtonElement>('button[data-tool="ask"]')!.click();
    click(250, 105); click(250, 305);
    expect(svg.querySelectorAll(".sel-red, .sel-yellow")).toHaveLength(2);
    const ask = host.querySelector<HTMLButtonElement>("[data-ask]")!;
    expect(ask.disabled).toBe(true);                                // not before the instrument is ready
    p.setStatus("ready");
    expect(ask.disabled).toBe(false);
  });
});
