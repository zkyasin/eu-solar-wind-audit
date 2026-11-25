/** Chapter 01: published output against the modelled potential, and the error of each model. */

import { COUNTRIES, daily, load, TECHNOLOGIES, YEARS } from "./data.js";
import { day, energy, MONTHS, number, percent } from "./format.js";
import {
  area,
  drawOnView,
  html,
  line,
  linear,
  options,
  paint,
  responsive,
  svg,
  ticks,
  tip,
  tipLine,
} from "./plot.js";

const state = { country: "FR", technology: "solar", year: "2025" };

const LADDER = [
  ["raw", "Physics, nominal parameters"],
  ["climatology", "Climatology by month and hour"],
  ["calibrated", "Calibrated"],
  ["calibrated_level", "With stratum ratios and yearly level"],
];

export async function renderModel() {
  const quality = await load("quality.json");
  renderLadder(quality);

  const plot = document.getElementById("model-plot");
  const tipper = tip(plot);
  let rows = [];
  let geometry = null;

  const update = async () => {
    const all = await daily(state.country, state.technology);
    rows = all.filter((row) => row.year === Number(state.year));
    renderReadout(rows);
    redraw();
  };

  const redraw = responsive(plot, (width) => {
    const chart = drawDaily(rows, width);
    geometry = chart.geometry;
    return chart.node;
  });

  options("country", COUNTRIES, state.country, (value) => {
    state.country = value;
    update();
  });
  options("technology", TECHNOLOGIES, state.technology, (value) => {
    state.technology = value;
    update();
  });
  options(
    "year",
    Object.fromEntries(YEARS.map((year) => [year, String(year)])),
    state.year,
    (value) => {
      state.year = value;
      update();
    },
  );

  plot.addEventListener("pointermove", (event) => {
    if (!geometry) return;
    const box = plot.getBoundingClientRect();
    const index = Math.round(geometry.x.invert(event.clientX - box.left));
    const row = rows[index];
    const focus = plot.querySelector(".focus");
    if (!row || row.published === null) {
      focus?.setAttribute("visibility", "hidden");
      tipper.hide();
      return;
    }
    const cx = geometry.x(index);
    focus.setAttribute("visibility", "visible");
    focus.querySelector("line").setAttribute("x1", cx);
    focus.querySelector("line").setAttribute("x2", cx);
    const lines = [
      html("strong", {}, [day(row.day)]),
      tipLine(paint("--published"), `Published ${energy(row.published)}`),
      tipLine(paint("--model"), `Potential ${energy(row.potential)}`),
    ];
    if (row.negativeHours > 0) {
      lines.push(html("span", {}, [`${row.negativeHours} h at negative prices`]));
      lines.push(html("span", {}, [`${energy(row.heldBack)} held back`]));
    }
    tipper.show(lines, cx, geometry.y(Math.max(row.published, row.potential)));
  });
  plot.addEventListener("pointerleave", () => {
    plot.querySelector(".focus")?.setAttribute("visibility", "hidden");
    tipper.hide();
  });

  await update();
}

function renderReadout(rows) {
  const both = rows.filter((row) => row.published !== null && row.potential !== null);
  const published = both.reduce((sum, row) => sum + row.published, 0);
  const potential = both.reduce((sum, row) => sum + row.potential, 0);
  const hours = rows.reduce((sum, row) => sum + row.negativeHours, 0);
  const held = rows.reduce((sum, row) => sum + (row.heldBack ?? 0), 0);
  const empty = both.length === 0;
  const cells = [
    ["Published", empty ? "–" : energy(published), `${both.length} days`],
    ["Over the potential", empty ? "–" : percent(published / potential, 1), "same days"],
    ["Negative-price hours", number(hours), "day-ahead market"],
    ["Held back at them", empty ? "–" : energy(held), "estimated"],
  ];
  document
    .getElementById("model-readout")
    .replaceChildren(
      ...cells.map(([label, value, detail]) =>
        html("div", {}, [
          html("dt", {}, [label]),
          html("dd", {}, [value, html("small", {}, [detail])]),
        ]),
      ),
    );
  const caption = document.getElementById("model-caption");
  caption.textContent = empty
    ? "In 2023 the published French wind series includes offshore farms, so the year is left out of the onshore model."
    : "Daily output, GWh. Below, the hours with a negative day-ahead price on each day.";
}

function drawDaily(rows, width) {
  const height = width < 560 ? 260 : 320;
  const strip = 26;
  const margin = { top: 8, right: 6, bottom: 22 + strip + 10, left: 34 };
  const values = rows.flatMap((row) => [row.published, row.potential]).filter((v) => v !== null);
  const top = values.length ? Math.max(...values) * 1.05 : 1;
  const x = linear([0, 365], [margin.left, width - margin.right]);
  const y = linear([0, top], [height - margin.bottom, margin.top]);
  const stripTop = height - strip - 18;
  const hoursScale = linear([0, 24], [0, strip]);

  const gridValues = ticks(0, top, width < 560 ? 3 : 4);
  const grid = svg(
    "g",
    { class: "grid" },
    gridValues.map((value) =>
      svg("line", { x1: margin.left, x2: width - margin.right, y1: y(value), y2: y(value) }),
    ),
  );
  const yAxis = svg(
    "g",
    { class: "axis" },
    gridValues.map((value) =>
      svg("text", { x: margin.left - 8, y: y(value) + 4, "text-anchor": "end" }, [number(value)]),
    ),
  );
  const months = svg(
    "g",
    { class: "axis" },
    MONTHS.map((letter, month) =>
      svg("text", { x: x(month * 30.4 + 15), y: height - 2, "text-anchor": "middle" }, [letter]),
    ),
  );

  const at = (key) => rows.map((row, i) => [x(i), row[key] === null ? null : y(row[key])]);
  const held = area(
    rows.map((row, i) =>
      row.heldBack > 0 && row.published !== null
        ? [x(i), y(row.published + row.heldBack), y(row.published)]
        : [x(i), null, null],
    ),
  );
  const barWidth = Math.max(1, (x(1) - x(0)) * 0.8);
  const bars = rows
    .filter((row) => row.negativeHours > 0)
    .map((row) => {
      const i = rows.indexOf(row);
      const h = hoursScale(row.negativeHours);
      return svg("rect", {
        x: x(i) - barWidth / 2,
        y: stripTop + strip - h,
        width: barWidth,
        height: h,
      });
    });

  const marks = svg("g", { class: "marks" }, [
    svg("path", { d: held, fill: "url(#hatch)" }),
    svg("path", {
      d: line(at("potential")),
      fill: "none",
      stroke: paint("--model"),
      "stroke-width": 1.25,
      "stroke-linejoin": "round",
    }),
    svg("path", {
      d: line(at("published")),
      fill: "none",
      stroke: paint("--published"),
      "stroke-width": 1.25,
      "stroke-linejoin": "round",
    }),
    svg("g", { fill: paint("--model"), "fill-opacity": 0.85 }, bars),
  ]);

  const node = svg(
    "svg",
    {
      width,
      height,
      viewBox: `0 0 ${width} ${height}`,
      role: "img",
      "aria-label": `Daily published output and modelled potential, ${COUNTRIES[state.country]}, ${TECHNOLOGIES[state.technology].toLowerCase()}, ${state.year}.`,
    },
    [
      grid,
      svg("line", {
        class: "baseline",
        x1: margin.left,
        x2: width - margin.right,
        y1: y(0),
        y2: y(0),
      }),
      svg("line", {
        class: "baseline",
        x1: margin.left,
        x2: width - margin.right,
        y1: stripTop + strip,
        y2: stripTop + strip,
      }),
      svg("text", { class: "label", x: margin.left, y: stripTop - 3 }, ["Negative-price hours"]),
      yAxis,
      months,
      marks,
      svg("g", { class: "focus", visibility: "hidden" }, [
        svg("line", { class: "crosshair", y1: margin.top, y2: stripTop + strip }),
      ]),
    ],
  );
  return { node, geometry: { x, y } };
}

function renderLadder(quality) {
  const series = [
    ["FR", "solar"],
    ["FR", "wind_onshore"],
    ["DE", "solar"],
    ["DE", "wind_onshore"],
  ];
  const scope = quality.filter((row) => row.period === "2023-2025");
  const largest = Math.max(...scope.map((row) => row.rmse_cf));
  const container = document.getElementById("model-ladder");
  container.replaceChildren(
    ...series.map(([country, technology]) => {
      const rows = scope.filter((row) => row.country === country && row.technology === technology);
      const final = rows.find((row) => row.model === "calibrated_level");
      return html("article", { class: "rung" }, [
        html("h4", {}, [`${COUNTRIES[country]}, ${TECHNOLOGIES[technology].toLowerCase()}`]),
        html(
          "ol",
          {},
          LADDER.map(([model, label], index) => {
            const row = rows.find((r) => r.model === model);
            const bar = html("span", { class: "bar" });
            bar.style.setProperty("--share", row.rmse_cf / largest);
            bar.style.setProperty("--step", index);
            return html("li", { class: model === "calibrated_level" ? "final" : null }, [
              html("span", { class: "rung-label" }, [label]),
              html("span", { class: "track" }, [bar]),
              html("span", { class: "rung-value" }, [number(row.rmse_cf, 3)]),
            ]);
          }),
        ),
        html("p", {}, [`Hourly correlation ${number(final.correlation, 3)}`]),
      ]);
    }),
  );
  drawOnView(container);
}
