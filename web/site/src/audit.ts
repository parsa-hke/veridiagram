// Milestone 1 check: every number on the page traces to a JSON field. A number in visible text must
// sit inside .num (traced), .lit (a literal such as a year) or .ref (a plan section reference).
// Digits inside names (Geometry3K, DINOv2) are part of a word, not a number.

const ALLOWED = ".num, .lit, .ref, .live, code, script, style, svg";   // .live: computed from what the reader drags

const NUMBER = /(^|[^A-Za-z\d])\d/;

export interface Untraced {
  text: string;
  context: string;
}

export function untracedNumbers(root: Element): Untraced[] {
  const out: Untraced[] = [];
  const walker = root.ownerDocument.createTreeWalker(root, NodeFilter.SHOW_TEXT);
  for (let node = walker.nextNode(); node; node = walker.nextNode()) {
    const text = node.textContent ?? "";
    if (!NUMBER.test(text)) continue;
    const parent = node.parentElement;
    if (!parent || parent.closest(ALLOWED)) continue;
    out.push({ text: text.trim(), context: (parent.textContent ?? "").trim().slice(0, 80) });
  }
  return out;
}

/** Every traced number's source, and which of them resolved to nothing. */
export function tracedSources(root: Element): { sources: string[]; missing: string[] } {
  const nums = Array.from(root.querySelectorAll<HTMLElement>(".num"));
  return {
    sources: nums.map((e) => e.dataset.src ?? ""),
    missing: nums.filter((e) => e.classList.contains("missing")).map((e) => e.dataset.src ?? ""),
  };
}
