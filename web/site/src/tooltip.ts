// One tooltip for every chart mark: elements with data-tip show it on hover (and on tap).

export function wireTooltips(root: Element): void {
  const tip = document.createElement("div");
  tip.className = "tooltip";
  tip.setAttribute("role", "status");
  tip.hidden = true;
  document.body.append(tip);
  const show = (el: Element, x: number, y: number) => {
    tip.textContent = el.getAttribute("data-tip") ?? "";
    tip.hidden = false;
    const w = tip.offsetWidth;
    tip.style.left = `${Math.max(8, Math.min(window.innerWidth - w - 8, x - w / 2))}px`;
    tip.style.top = `${Math.max(8, y - tip.offsetHeight - 14)}px`;
  };
  root.addEventListener("pointermove", (e) => {
    const ev = e as PointerEvent;
    const el = (ev.target as Element).closest?.("[data-tip]");
    if (el) show(el, ev.clientX, ev.clientY);
    else tip.hidden = true;
  });
  root.addEventListener("pointerleave", () => (tip.hidden = true));
  window.addEventListener("scroll", () => (tip.hidden = true), { passive: true });
}
