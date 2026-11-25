/**
 * The hero: the solar output that mainland France held back at negative prices in 2025, beside a
 * ridgeline of every year since 2019, where the held-back output stands hatched above the
 * published one.
 */

import { daily, load, YEARS } from "./data.js";
import { day, energy, MONTHS, number, THIN } from "./format.js";
import {
  area,
  html,
  line,
  linear,
  paint,
  reducedMotion,
  smooth,
  svg,
  tip,
  tipLine,
} from "./plot.js";

const LATEST = YEARS.at(-1);
/** Days on each side of the moving mean of the ridgeline. */
const HALF_WINDOW = 3;
/** The height of the tallest ridge, in rows: above one, a year overlaps the one before it. */
const RISE = 1.7;

export async function renderHero() {
  const [runs, quality, days] = await Promise.all([
    load("modulation.json"),
    load("quality.json"),
    daily("FR", "solar"),
  ]);
  renderFigures(runs, quality);
  renderRidges(days);
}

function renderFigures(runs, quality) {
  const solar = runs
    .find((run) => run.country === "FR" && run.technology === "solar")
    .periods.find((period) => period.period === String(LATEST));
  const twh = solar.estimate_gwh / 1000;
  countUp(document.getElementById("hero-figure"), twh);
  document.getElementById("hero-reference").textContent = energy(solar.reference_gwh);
  document.getElementById("hero-reference-note").textContent =
    `l’estimation en vaut ${number(solar.estimate_gwh / solar.reference_gwh, 2)} fois ; ` +
    `intervalle à 90${THIN}% : ${number(solar.low_gwh / 1000, 2)} à ` +
    `${number(solar.high_gwh / 1000, 2)}${THIN}TWh`;

  const held = quality.filter(
    (row) => row.model === "calibrated_level" && row.period === "2023-2025",
  );
  const low = Math.min(...held.map((row) => row.correlation));
  const high = Math.max(...held.map((row) => row.correlation));
  document.getElementById("hero-correlation").textContent =
    `${number(low, 2)} à ${number(high, 2)}`;
  const errors = held.map((row) => row.rmse_cf);
  document.getElementById("hero-correlation-note").textContent =
    "avec la production publiée, heure par heure, sur les années exclues du calage ; erreur " +
    `de ${number(Math.min(...errors), 3)} à ${number(Math.max(...errors), 3)} de la capacité`;
}

function countUp(node, value) {
  const text = number(value, 2);
  if (reducedMotion.matches) {
    node.textContent = text;
    return;
  }
  const start = performance.now();
  const from = value * 0.9;
  const step = (now) => {
    const t = Math.min(1, (now - start) / 1100);
    const eased = 1 - (1 - t) ** 4;
    node.textContent = number(from + (value - from) * eased, 2);
    if (t < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

function renderRidges(days) {
  const plot = document.getElementById("ridges-plot");
  const years = YEARS.map((year) => {
    const rows = days.filter((row) => row.year === year);
    const published = smooth(
      rows.map((row) => row.published),
      HALF_WINDOW,
    );
    const held = smooth(
      rows.map((row) => (row.published === null ? null : row.heldBack)),
      HALF_WINDOW,
    );
    return { year, rows, published, held };
  });
  const peak = Math.max(
    ...years.flatMap(({ published, held }) => published.map((value, i) => value + (held[i] ?? 0))),
  );

  let geometry;

  const render = () => {
    const width = plot.clientWidth;
    const height = plot.clientHeight;
    const margin = { top: 6, right: 4, bottom: 26, left: 40 };
    const spacing = (height - margin.top - margin.bottom) / (years.length - 1 + RISE);
    const amplitude = spacing * RISE;
    const x = linear([0, 365], [margin.left, width - margin.right]);
    const base = (index) => margin.top + amplitude + index * spacing;
    const y = (index, value) => base(index) - (value / peak) * amplitude;
    geometry = { x, base, y, spacing, amplitude, margin };

    const rows = years.map(({ year, published, held }, index) => {
      const latest = year === LATEST;
      const top = published.map((value, i) => (value === null ? null : value + (held[i] ?? 0)));
      const at = (values) =>
        values.map((value, i) => [x(i), value === null ? null : y(index, value)]);
      const heldOnly = top.map((value, i) => ((held[i] ?? 0) > 0.05 ? value : null));
      return svg("g", { class: "ridge", "data-year": year }, [
        svg("path", {
          d: area(
            top.map((value, i) => [x(i), value === null ? null : y(index, value), base(index)]),
          ),
          fill: paint("--paper"),
        }),
        svg("path", {
          class: "held",
          d: area(
            top.map((value, i) => [
              x(i),
              (held[i] ?? 0) > 0.05 ? y(index, value) : null,
              (held[i] ?? 0) > 0.05 ? y(index, published[i]) : null,
            ]),
          ),
          fill: "url(#hatch)",
        }),
        svg("path", {
          class: "held-edge",
          d: line(at(heldOnly)),
          fill: "none",
          stroke: paint("--model"),
          "stroke-width": 1,
        }),
        svg("path", {
          class: "published",
          d: line(at(published)),
          fill: "none",
          stroke: latest ? paint("--published") : paint("--dim"),
          "stroke-width": latest ? 1.75 : 1.15,
          "stroke-linejoin": "round",
          pathLength: 1,
        }),
        svg(
          "text",
          {
            x: 0,
            y: base(index) - 2,
            class: latest ? "label-strong" : "label",
          },
          [String(year)],
        ),
      ]);
    });

    const months = MONTHS.map((letter, month) =>
      svg("text", { x: x(month * 30.4 + 15), y: height - 6, "text-anchor": "middle" }, [letter]),
    );
    const crosshair = svg("g", { class: "focus", visibility: "hidden" }, [
      svg("line", { class: "crosshair", y1: margin.top, y2: height - margin.bottom }),
      svg("circle", { r: 3.5, fill: paint("--ink"), stroke: paint("--paper"), "stroke-width": 2 }),
    ]);
    return svg(
      "svg",
      {
        width,
        height,
        viewBox: `0 0 ${width} ${height}`,
        role: "img",
        "aria-label":
          "Production solaire journalière publiée pour la France continentale, une ligne par année " +
          "de 2019 à 2025. La production non injectée aux prix négatifs apparaît dès 2020, grossit " +
          "en 2024 et culmine au printemps et à l’été 2025.",
      },
      [svg("g", { class: "marks" }, rows), svg("g", { class: "axis" }, months), crosshair],
    );
  };

  const draw = () => {
    plot.querySelector("svg")?.remove();
    plot.prepend(render());
  };
  draw();
  if (!reducedMotion.matches) play(plot);
  let width = plot.clientWidth;
  new ResizeObserver(() => {
    if (plot.clientWidth !== width) {
      width = plot.clientWidth;
      draw();
    }
  }).observe(plot);

  const tipper = tip(plot);
  plot.addEventListener("pointermove", (event) => {
    const box = plot.getBoundingClientRect();
    const px = event.clientX - box.left;
    const py = event.clientY - box.top;
    const { x, base, y, spacing, amplitude, margin } = geometry;
    const index = Math.max(
      0,
      Math.min(years.length - 1, Math.round((py - margin.top - amplitude) / spacing + 0.35)),
    );
    const { rows, published } = years[index];
    const position = Math.round(x.invert(px));
    const row = rows[position];
    const focus = plot.querySelector(".focus");
    if (!row || row.published === null || px < margin.left) {
      focus.setAttribute("visibility", "hidden");
      tipper.hide();
      return;
    }
    const cx = x(position);
    focus.setAttribute("visibility", "visible");
    focus.querySelector("line").setAttribute("x1", cx);
    focus.querySelector("line").setAttribute("x2", cx);
    focus.querySelector("circle").setAttribute("cx", cx);
    focus.querySelector("circle").setAttribute("cy", y(index, published[position]));
    const lines = [
      html("strong", {}, [day(row.day)]),
      tipLine(paint("--published"), `Publiée ${energy(row.published)}`),
      tipLine(paint("--model"), `Potentiel ${energy(row.potential)}`),
    ];
    if (row.negativeHours > 0) {
      lines.push(
        html("span", {}, [
          `${energy(row.heldBack)} non injectés sur ${row.negativeHours}${THIN}h à prix négatif`,
        ]),
      );
    }
    tipper.show(lines, cx, base(index) - amplitude / 2);
  });
  plot.addEventListener("pointerleave", () => {
    plot.querySelector(".focus")?.setAttribute("visibility", "hidden");
    tipper.hide();
  });
}

/** Draws the years one after another, then lays the held-back output over them. */
function play(plot) {
  const ridges = [...plot.querySelectorAll(".ridge")];
  ridges.forEach((ridge, index) => {
    const delay = index * 90;
    ridge.querySelector(".published").animate(
      [
        { strokeDasharray: "1 1", strokeDashoffset: 1 },
        { strokeDasharray: "1 1", strokeDashoffset: 0 },
      ],
      { duration: 1100, delay, easing: "cubic-bezier(0.23, 1, 0.32, 1)", fill: "backwards" },
    );
    for (const part of ridge.querySelectorAll(".held, .held-edge")) {
      part.animate([{ opacity: 0 }, { opacity: 1 }], {
        duration: 700,
        delay: ridges.length * 90 + 500,
        easing: "ease-out",
        fill: "backwards",
      });
    }
  });
}
