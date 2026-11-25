/**
 * The few pieces every chart of the page shares: SVG elements, linear scales, a tip, and the
 * wipe that draws a chart the first time it scrolls into view.
 */

const NS = "http://www.w3.org/2000/svg";

export const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)");

export function svg(tag, attributes = {}, children = []) {
  const node = document.createElementNS(NS, tag);
  for (const [name, value] of Object.entries(attributes)) {
    if (value === null || value === undefined || value === false) continue;
    // Presentation attributes do not read custom properties; the style does.
    if (typeof value === "string" && value.startsWith("var(")) node.style.setProperty(name, value);
    else node.setAttribute(name, value);
  }
  node.append(...children);
  return node;
}

export function html(tag, attributes = {}, children = []) {
  const node = document.createElement(tag);
  for (const [name, value] of Object.entries(attributes)) {
    if (value !== null && value !== undefined && value !== false) node.setAttribute(name, value);
  }
  node.append(...children);
  return node;
}

export function linear([d0, d1], [r0, r1]) {
  const scale = (value) => r0 + ((value - d0) / (d1 - d0 || 1)) * (r1 - r0);
  scale.invert = (position) => d0 + ((position - r0) / (r1 - r0 || 1)) * (d1 - d0);
  scale.domain = [d0, d1];
  return scale;
}

/** Round ticks over [low, high], about `count` of them. */
export function ticks(low, high, count = 5) {
  const raw = (high - low) / Math.max(count, 1);
  const power = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10]
    .map((factor) => factor * power)
    .find((s) => s >= raw * (1 - 1e-9));
  const values = [];
  for (let value = Math.ceil(low / step) * step; value <= high + step * 1e-9; value += step) {
    values.push(Number(value.toFixed(10)));
  }
  return values;
}

/** A path through the points, broken wherever a value is missing. */
export function line(points) {
  let path = "";
  let open = false;
  for (const [x, y] of points) {
    if (y === null || Number.isNaN(y)) {
      open = false;
      continue;
    }
    path += `${open ? "L" : "M"}${x.toFixed(1)},${y.toFixed(1)}`;
    open = true;
  }
  return path;
}

/** A closed area between two curves given as [x, top, bottom], broken at missing values. */
export function area(points) {
  const runs = [];
  let run = [];
  for (const point of points) {
    if (point[1] === null || point[2] === null) {
      if (run.length) runs.push(run);
      run = [];
    } else {
      run.push(point);
    }
  }
  if (run.length) runs.push(run);
  return runs
    .map((part) => {
      const top = part.map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join("L");
      const bottom = part
        .toReversed()
        .map(([x, , y]) => `${x.toFixed(1)},${y.toFixed(1)}`)
        .join("L");
      return `M${top}L${bottom}Z`;
    })
    .join("");
}

/** A centred moving mean that skips missing values and keeps them missing. */
export function smooth(values, half) {
  return values.map((value, index) => {
    if (value === null) return null;
    let sum = 0;
    let count = 0;
    for (let k = Math.max(0, index - half); k <= Math.min(values.length - 1, index + half); k++) {
      if (values[k] !== null) {
        sum += values[k];
        count += 1;
      }
    }
    return sum / count;
  });
}

/** A tip over the plot; `show` takes its lines as DOM nodes or text, and the pointer position. */
export function tip(plot) {
  const node = html("div", { class: "tip", role: "status", hidden: "" });
  plot.append(node);
  return {
    show(children, x, y) {
      node.replaceChildren(...children);
      node.hidden = false;
      const width = node.offsetWidth;
      const height = node.offsetHeight;
      const room = plot.clientWidth;
      const left = x + 14 + width > room ? Math.max(0, x - 14 - width) : x + 14;
      const top = Math.max(0, Math.min(y - height / 2, plot.clientHeight - height));
      node.style.transform = `translate(${left}px, ${top}px)`;
    },
    hide() {
      node.hidden = true;
    },
  };
}

/** A tip line with the colored dot of its series. */
export function tipLine(color, text) {
  const dot = html("i");
  dot.style.background = color;
  return html("span", {}, [dot, text]);
}

const pending = new IntersectionObserver(
  (entries) => {
    for (const entry of entries) {
      if (!entry.isIntersecting) continue;
      pending.unobserve(entry.target);
      entry.target.dataset.draw = "running";
      // Done once every animation inside has ended, so that a redraw does not replay them.
      entry.target.addEventListener("animationend", function settle() {
        if (entry.target.getAnimations({ subtree: true }).length) return;
        entry.target.dataset.draw = "done";
        entry.target.removeEventListener("animationend", settle);
      });
    }
  },
  { rootMargin: "0px 0px -12% 0px" },
);

/** Marks `node` to be drawn the first time it comes into view; the CSS runs the animation. */
export function drawOnView(node) {
  if (node.dataset.draw || reducedMotion.matches) return;
  node.dataset.draw = "pending";
  pending.observe(node);
}

/**
 * Draws a chart into `plot` now and again whenever its width changes. `render` gets the width
 * and returns the SVG. The marks are wiped in once, the first time the plot comes into view.
 */
export function responsive(plot, render) {
  let width = 0;
  const draw = () => {
    width = Math.round(plot.clientWidth);
    plot.querySelector("svg")?.remove();
    plot.prepend(render(width));
  };
  draw();
  drawOnView(plot);
  new ResizeObserver(() => {
    if (Math.round(plot.clientWidth) !== width) draw();
  }).observe(plot);
  return draw;
}

/**
 * A design token as a CSS value. Marks are painted with it through their style, so that they
 * follow the theme without being drawn again.
 */
export function paint(name) {
  return `var(${name})`;
}

/** Radio buttons drawn as a segmented control, in the `.options` element named `name`. */
export function options(name, choices, chosen, onChange) {
  const group = document.querySelector(`.options[data-name="${name}"]`);
  group.replaceChildren(
    ...Object.entries(choices).map(([value, label]) =>
      html("label", {}, [
        html("input", { type: "radio", name, value, checked: value === chosen }),
        html("span", {}, [label]),
      ]),
    ),
  );
  group.addEventListener("change", (event) => onChange(event.target.value));
}
