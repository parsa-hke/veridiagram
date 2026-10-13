import { book1 } from "./books/book1";
import { book2 } from "./books/book2";
import { book3 } from "./books/book3";
import { get, lit, n } from "./data";
import { book4 } from "./demo/book4";
import { hero } from "./figures/geometry";
import { resetNotes } from "./ui";

/** The frontispiece figure: a compass circle, two Byrne bars, and the machine's green trace over both.
 * Static here; milestone 3 draws it stroke by stroke. */
function emblem(): string {
  return `
<svg class="emblem" viewBox="0 0 320 240" role="img" aria-labelledby="emblem-title">
  <title id="emblem-title">A circle, a red bar and a yellow bar, and the model's green trace over both bars</title>
  <circle class="ink-line" cx="160" cy="120" r="92" />
  <circle class="ink-dot" cx="160" cy="120" r="3" />
  <line class="bar bar-red" x1="58" y1="88" x2="262" y2="62" />
  <line class="bar bar-yellow" x1="58" y1="178" x2="262" y2="152" />
  <g class="machine-trace">
    <line class="halo" x1="58" y1="88" x2="262" y2="62" />
    <line class="halo" x1="58" y1="178" x2="262" y2="152" />
    <line class="core" x1="58" y1="88" x2="262" y2="62" />
    <line class="core" x1="58" y1="178" x2="262" y2="152" />
  </g>
</svg>`;
}

function frontispiece(): string {
  return `
<header class="frontispiece" id="top">
  <p class="imprint">The Elements, a lost edition · written for a machine</p>
  <h1>${get("facts:meta.title")}</h1>
  ${hero() || `<figure class="emblem-figure">${emblem()}</figure>`}
  <p class="subtitle">${get("facts:meta.subtitle")}</p>
  <p class="course smallcaps">${get("facts:meta.author")} · ${lit(String(get("facts:meta.course")))} ·
    ${get("facts:meta.institution")} · ${n("facts:meta.year", "raw")}</p>
  <div class="actions">
    <a class="button primary" href="#book-1">Read the Elements</a>
    <a class="button" href="#book-4">Construct something</a>
  </div>
</header>`;
}

function contents(): string {
  return `
<nav class="contents" aria-label="Contents">
  <h2 class="smallcaps">Contents</h2>
  <ol>
    <li><a href="#book-1"><span class="smallcaps">Book I</span> The Problem</a></li>
    <li><a href="#book-2"><span class="smallcaps">Book II</span> The Instrument</a></li>
    <li><a href="#book-3"><span class="smallcaps">Book III</span> The Propositions</a></li>
    <li><a href="#book-4"><span class="smallcaps">Book IV</span> Constructions</a></li>
    <li><a href="#colophon"><span class="smallcaps">Colophon</span></a></li>
  </ol>
</nav>`;
}

function colophon(): string {
  const repo = get("facts:meta.repo_public") && get("facts:meta.repo_url")
    ? `<a href="${get("facts:meta.repo_url")}">the code repository</a>`
    : "the code repository, which is private for now";
  return `
<footer class="colophon" id="colophon" data-running="Colophon">
  <h2>Colophon</h2>
  <p>Set in Libre Caslon and IM Fell English, after Oliver Byrne's <em>The First Six Books of the Elements of
  Euclid</em> (${n("facts:meta.byrne_year", "raw")}), which taught geometry in red, yellow and blue instead of
  letters. Green, which Byrne never used, belongs here to the machine alone.</p>
  <p>Every number on this page is read from JSON written by <code>python -m web.export</code> from the
  project's run records (export of commit <code>${get("results:_export.commit") ?? "unknown"}</code>). See ${repo}.</p>
  <p class="smallcaps">${get("facts:meta.author")} · ${get("facts:meta.institution")} · ${n("facts:meta.year", "raw")}</p>
</footer>`;
}

export function renderPage(root: Element): void {
  resetNotes();
  root.innerHTML = frontispiece() + contents() + `<main>${book1()}${book2()}${book3()}${book4()}</main>` + colophon();
}
