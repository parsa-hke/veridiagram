import { describe, expect, it } from "vitest";
import { format, get, ms, n, setDocs } from "./data";
import { untracedNumbers } from "./audit";

const docs = {
  results: { experiments: { finetune: { n_seeds: 3, units: { g: { geometric: { mean: 0.8667, sd: 0.0071 } } } },
                            one: { n_seeds: 1, units: { g: { geometric: { mean: 0.43, sd: null } } } } } },
  scaling: { points: [{ mean: 0.5 }] },
  pseudo_labels: {},
  facts: { a: { "b": 1296 }, missing: null },
};

describe("data", () => {
  it("formats", () => {
    expect(format(0.8667, "f3")).toBe("0.867");
    expect(format(1296, "int")).toBe("1,296");
    expect(format(0.8823, "pct1")).toBe("88.2%");
    expect(format(89, "pctv")).toBe("89%");
    expect(format(7.94, "pctv")).toBe("7.94%");
    expect(format(2e-5, "sci")).toBe("2 × 10⁻⁵");
    expect(format(0.001, "sci")).toBe("1 × 10⁻³");
    expect(format(3.7, "deg")).toBe("3.7°");
    expect(format(1, "seeds")).toBe("one seed");
    expect(format(3, "seeds")).toBe("three seeds");
    expect(format(null, "f3")).toBe("—");
    expect(format(Number.NaN, "f3")).toBe("—");
  });

  it("resolves paths and traces numbers", () => {
    setDocs(docs);
    expect(get("facts:a.b")).toBe(1296);
    expect(get("scaling:points.0.mean")).toBe(0.5);
    expect(get("facts:nope.deeper")).toBeUndefined();
    expect(n("facts:a.b", "int")).toContain('data-src="facts:a.b"');
    expect(n("facts:missing")).toContain("missing");
    expect(ms("finetune", "g")).toMatch(/0\.867<\/span> ± <span[^>]*>0\.007/);
    expect(ms("one", "g")).not.toContain("±");                         // one seed: no sd
  });

  it("finds untraced numbers but not names or traced ones", () => {
    setDocs(docs);
    const el = document.createElement("div");
    el.innerHTML = `<p>Geometry3K and DINOv2 ${n("facts:a.b", "int")}</p><p>typed 0.87 here</p><code>x_s2</code>`;
    const bad = untracedNumbers(el);
    expect(bad).toHaveLength(1);
    expect(bad[0].text).toContain("0.87");
  });
});
