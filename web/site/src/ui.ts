// Small page furniture shared by the Books: margin notes, the theme switch and the running head.

let noteId = 0;

export function resetNotes(): void {
  noteId = 0;
}

/** A margin note (Tufte-style sidenote), placed inline at the end of a paragraph. On wide screens it
 * sits in the right margin; on phones it folds into a tappable † marker. Content must be inline HTML. */
export function note(html: string): string {
  const id = `note-${++noteId}`;
  return `<button class="note-mark" type="button" aria-expanded="false" aria-controls="${id}"
    aria-label="Show margin note">†</button><span class="sidenote" id="${id}" role="note"><span class="note-dagger"
    aria-hidden="true">†</span> ${html}</span>`;
}

export function wireNotes(root: Element): void {
  root.querySelectorAll<HTMLButtonElement>(".note-mark").forEach((b) => {
    b.addEventListener("click", () => {
      const open = b.getAttribute("aria-expanded") !== "true";
      b.setAttribute("aria-expanded", String(open));
      document.getElementById(b.getAttribute("aria-controls") ?? "")?.classList.toggle("open", open);
    });
  });
}

// ------------------------------------------------------------------ editions (light / dark)

type Theme = "light" | "dark";
const KEY = "edition";

function stored(): Theme | null {
  try {
    const v = localStorage.getItem(KEY);
    return v === "light" || v === "dark" ? v : null;
  } catch {
    return null;
  }
}

function current(): Theme {
  return (document.documentElement.dataset.theme as Theme | undefined)
    ?? (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
}

export function applyStoredTheme(): void {
  const t = stored();
  if (t) document.documentElement.dataset.theme = t;
}

export function wireTheme(button: HTMLButtonElement): void {
  const label = () => {
    const dark = current() === "dark";
    button.textContent = dark ? "Day edition" : "Night edition";
    button.setAttribute("aria-label", dark ? "Switch to the day edition (light)" : "Switch to the night edition (dark)");
  };
  label();
  button.addEventListener("click", () => {
    const next: Theme = current() === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try {
      localStorage.setItem(KEY, next);
    } catch {
      /* private mode: the choice lasts for this visit */
    }
    label();
  });
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", label);
}

// ------------------------------------------------------------------ running head

export function wireRunningHead(head: HTMLElement, root: Element): void {
  const slot = head.querySelector<HTMLElement>(".running-book");
  const books = Array.from(root.querySelectorAll<HTMLElement>("[data-running]"));
  if (!slot || !books.length || !("IntersectionObserver" in window)) return;
  const io = new IntersectionObserver(
    (entries) => {
      for (const e of entries) {
        if (e.isIntersecting) slot.textContent = (e.target as HTMLElement).dataset.running ?? "";
      }
    },
    { rootMargin: "-45% 0px -50% 0px" },
  );
  books.forEach((b) => io.observe(b));
}
