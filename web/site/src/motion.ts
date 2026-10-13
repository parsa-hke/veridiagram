// Page motion beyond the constructions: drop caps bleed in like ink, the Byrne rule under each Book
// title is laid down block by block, proposition labels settle, Q.E.D. squares are stamped, margin
// notes glide in from the margin. Each element gets .in-view once, when it enters the viewport.
// Hidden starting states live under html.motion, which is set only here: without JavaScript, or with
// prefers-reduced-motion, everything is simply visible.

export const REVEAL = [".book > h2", ".opening", ".prop-no", ".qed", ".sidenote", ".legend"].join(", ");

export function wireMotion(root: Element): void {
  if (typeof matchMedia !== "function" || matchMedia("(prefers-reduced-motion: reduce)").matches) return;
  if (!("IntersectionObserver" in window)) return;
  document.documentElement.classList.add("motion");
  const io = new IntersectionObserver(
    (entries) => {
      for (const e of entries) {
        if (e.isIntersecting) {
          e.target.classList.add("in-view");
          io.unobserve(e.target);
        }
      }
    },
    { rootMargin: "0px 0px -12% 0px" },
  );
  root.querySelectorAll(REVEAL).forEach((el) => io.observe(el));
}
