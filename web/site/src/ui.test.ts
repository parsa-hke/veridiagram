import { beforeEach, describe, expect, it, vi } from "vitest";
import { note, resetNotes, wireNotes, wireTheme } from "./ui";

beforeEach(() => {
  vi.stubGlobal("matchMedia", (q: string) => ({ matches: false, media: q, addEventListener: () => {} }));
  delete document.documentElement.dataset.theme;
  localStorage.clear();
});

describe("margin notes", () => {
  it("number their ids and toggle open from the dagger", () => {
    resetNotes();
    const root = document.createElement("div");
    root.innerHTML = `<p>a${note("first")}</p><p>b${note("second")}</p>`;
    document.body.append(root);
    const marks = root.querySelectorAll<HTMLButtonElement>(".note-mark");
    expect(marks[1].getAttribute("aria-controls")).toBe("note-2");
    wireNotes(root);
    marks[0].click();
    expect(marks[0].getAttribute("aria-expanded")).toBe("true");
    expect(document.getElementById("note-1")!.classList.contains("open")).toBe(true);
    marks[0].click();
    expect(document.getElementById("note-1")!.classList.contains("open")).toBe(false);
    root.remove();
  });
});

describe("editions", () => {
  it("switch between day and night and remember the choice", () => {
    const b = document.createElement("button");
    wireTheme(b);
    expect(b.textContent).toBe("Night edition");
    b.click();
    expect(document.documentElement.dataset.theme).toBe("dark");
    expect(localStorage.getItem("edition")).toBe("dark");
    expect(b.textContent).toBe("Day edition");
    b.click();
    expect(document.documentElement.dataset.theme).toBe("light");
  });
});
