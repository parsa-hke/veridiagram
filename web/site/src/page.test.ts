// Milestone 1 check on the real export: every number on the page traces to a JSON field.
import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { DOC_NAMES, get, setDocs, type Docs } from "./data";
import { tracedSources, untracedNumbers } from "./audit";
import { renderPage } from "./page";

const DATA = resolve(__dirname, "../public/data");
const have = DOC_NAMES.every((d) => existsSync(resolve(DATA, `${d}.json`)));

describe.skipIf(!have)("page from the exported data", () => {
  it("renders Books I-III with every number traced", () => {
    const docs = Object.fromEntries(DOC_NAMES.map((d) => [d, JSON.parse(readFileSync(resolve(DATA, `${d}.json`), "utf8"))]));
    setDocs(docs as Docs);
    const root = document.createElement("div");
    renderPage(root);
    for (const id of ["book-1", "book-2", "book-3", "prop-4", "prop-5", "prop-6", "prop-7", "prop-8", "prop-9", "prop-10"]) {
      expect(root.querySelector(`#${id}`), id).not.toBeNull();
    }
    expect(untracedNumbers(root)).toEqual([]);
    const { sources, missing } = tracedSources(root);
    expect(missing).toEqual([]);
    expect(sources.length).toBeGreaterThan(100);
    for (const s of sources) expect(get(s), s).not.toBeUndefined();
  });
});
