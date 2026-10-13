import { book1 } from "./books/book1";
import { book2 } from "./books/book2";
import { book3 } from "./books/book3";
import { get, lit, n } from "./data";

function frontispiece(): string {
  return `
<header class="frontispiece" id="top">
  <h1>${get("facts:meta.title")}</h1>
  <p class="subtitle">${get("facts:meta.subtitle")}</p>
  <p class="course">${get("facts:meta.author")} · ${lit(String(get("facts:meta.course")))} ·
    ${get("facts:meta.institution")} · ${n("facts:meta.year", "raw")}</p>
  <nav class="contents">
    <a href="#book-1">Book I: The Problem</a>
    <a href="#book-2">Book II: The Instrument</a>
    <a href="#book-3">Book III: The Propositions</a>
  </nav>
</header>`;
}

function colophon(): string {
  const repo = get("facts:meta.repo_public") && get("facts:meta.repo_url")
    ? `<a href="${get("facts:meta.repo_url")}">the code repository</a>`
    : "the code repository (private for now)";
  return `
<footer class="colophon" id="colophon">
  <h2>Colophon</h2>
  <p>Every number on this page is read from JSON written by <code>python -m web.export</code> from the
  project's run records (export of commit <code>${get("results:_export.commit") ?? "unknown"}</code>). See ${repo}.</p>
</footer>`;
}

export function renderPage(root: Element): void {
  root.innerHTML = frontispiece() + `<main>${book1()}${book2()}${book3()}</main>` + colophon();
}
