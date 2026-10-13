import { afterEach, describe, expect, it, vi } from "vitest";
import { setDocs } from "../data";
import { queryOf } from "./editor";
import { watchHealth, type Status } from "./api";

setDocs({ results: {}, scaling: {}, pseudo_labels: {}, facts: { meta: { api_url: null } }, figures: {} });
afterEach(() => {
  vi.unstubAllGlobals();
  vi.useRealTimers();
});

const respond = (seq: (number | object)[]) => {
  let i = 0;
  vi.stubGlobal("fetch", vi.fn(async () => {
    const r = seq[Math.min(i++, seq.length - 1)];
    if (typeof r === "number") return new Response("{}", { status: r });
    return new Response(JSON.stringify(r), { status: 200 });
  }));
};

async function run(seq: (number | object)[], ticks: number, stepMs = 5000): Promise<Status[]> {
  vi.useFakeTimers();
  respond(seq);
  const seen: Status[] = [];
  watchHealth((s) => seen.push(s), stepMs);
  for (let k = 0; k < ticks; k++) await vi.advanceTimersByTimeAsync(stepMs);
  return seen;
}

describe("instrument status", () => {
  it("a waking Space is warming, then ready", async () => {
    const s = await run([503, 502, { ok: true, ready: false }, { ok: true, ready: true }], 5);
    expect(s[0]).toBe("checking");
    expect(s).toContain("warming");
    expect(s.at(-1)).toBe("ready");
    expect(s).not.toContain("offline");
  });

  it("no API behind the page is offline at once", async () => {
    const s = await run([404], 3);
    expect(s).toEqual(["checking", "offline"]);
  });

  it("a model that cannot be loaded is offline", async () => {
    const s = await run([{ ok: false, ready: false }], 3);
    expect(s.at(-1)).toBe("offline");
  });
});

describe("questions from a selection", () => {
  it("names the query kind the server expects", () => {
    expect(queryOf([["seg", "A", "B"], ["seg", "C", "D"]])?.kind).toBe("seg_pair");
    expect(queryOf([["circle", 0], ["pt", "P"]])).toEqual({ kind: "pt_circle", refs: [["pt", "P"], ["circle", 0]] });
    expect(queryOf([["seg", "A", "B"], ["circle", 1]])?.kind).toBe("seg_circle");
    expect(queryOf([["pt", "P"], ["seg", "A", "B"]])).toEqual({ kind: "pt_line", refs: [["pt", "P"], ["pt", "A"], ["pt", "B"]] });
    expect(queryOf([["pt", "P"], ["pt", "Q"]])).toBeNull();
    expect(queryOf([["seg", "A", "B"]])).toBeNull();
  });
});

import { contradictionText, entity, statement } from "./witnesses";

describe("readout words", () => {
  it("writes statements and contradictions as a reader would", () => {
    expect(entity("A-B")).toBe("AB");
    expect(entity("S:C,D")).toBe("CD");
    expect(statement({ relation: "parallel", kind: "seg_pair", refs: [["seg", "A", "B"], ["seg", "C", "D"]] })).toBe("AB ∥ CD");
    expect(statement({ relation: "on_circle", kind: "pt_circle", refs: [["pt", "P"], ["circle", 0]] }, ["O"])).toBe("P on circle O");
    const r = { kind: "seg_pair", refs: [["seg", "B", "F"], ["seg", "A", "B"]], applicable: ["equal_length"], probs: { equal_length: 0.02 },
      ruler: { basis: "construction", equal_length: { holds: false, margin: 0.14, unit: "pct", tolerance: 10, measured: 14.2 } } } as never;
    const t = contradictionText({ kind: "derived_negated", fact: ["equal_length", "A-B", "B-F"] }, [r]);
    expect(t).toContain("imply AB = BF, which it denies");
    expect(t).toContain("does not hold");
    expect(t).toContain("14.2% apart");
  });
});
