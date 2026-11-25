/** Chapter 03: what each published series contains, from its level and its producer's word. */

import { load } from "./data.js";
import { number, percent } from "./format.js";
import { html, line, linear, paint, responsive, svg, ticks } from "./plot.js";

/** The series of the chart, as pairs of a feed-in series and a total over the same fleet. */
const SERIES = {
  smard: { label: "SMARD", country: "Germany", note: "feed-in", color: "--published" },
  energy_charts_total: {
    label: "Energy-Charts",
    country: "Germany",
    note: "total",
    color: "--third",
  },
  energinet: {
    label: "Energinet",
    country: "Denmark",
    note: "size classes",
    color: "--published",
    dashed: true,
  },
  energinet_total: {
    label: "Energinet",
    country: "Denmark",
    note: "with self-consumption",
    color: "--third",
    dashed: true,
  },
  eco2mix: { label: "éCO2mix", country: "France", note: "", color: "--ink" },
  entsoe_nl: { label: "Transparency Platform", country: "Netherlands", note: "", color: "--muted" },
};

const CHARTED = ["smard", "energy_charts_total", "energinet", "energinet_total", "eco2mix"];
const ORDER = [
  "smard",
  "energy_charts_total",
  "energinet",
  "energinet_total",
  "eco2mix",
  "entsoe_nl",
];

const WORDS = {
  included: "Included",
  excluded: "Excluded",
  inconclusive: "Inconclusive",
  covered: "Covered",
  not_covered: "Not covered",
};

export async function renderContent() {
  const perimeter = await load("perimeter.json");
  renderLegend();
  responsive(document.getElementById("content-plot"), (width) =>
    drawRatios(perimeter.years, width),
  );
  renderMatrix(perimeter);
}

function renderLegend() {
  document.getElementById("content-legend").replaceChildren(
    ...CHARTED.map((name) => {
      const { label, country, note, color, dashed } = SERIES[name];
      return html("li", {}, [
        svg("svg", { class: "key", viewBox: "0 0 24 12", "aria-hidden": "true" }, [
          svg("line", {
            x1: 0,
            y1: 6,
            x2: 24,
            y2: 6,
            stroke: paint(color),
            "stroke-dasharray": dashed ? "4 3" : null,
          }),
        ]),
        `${label}${note ? `, ${note}` : ""} (${country})`,
      ]);
    }),
  );
}

function drawRatios(years, width) {
  const narrow = width < 560;
  const height = narrow ? 280 : 320;
  const margin = { top: 12, right: 12, bottom: 24, left: 36 };
  const first = 2019;
  const last = 2024;
  const x = linear([first, last], [margin.left, width - margin.right]);
  const y = linear([0.5, 1.12], [height - margin.bottom, margin.top]);
  const gridValues = ticks(0.5, 1.1, 6);

  const series = CHARTED.map((name) => {
    const rows = years.filter((row) => row.series === name).sort((a, b) => a.year - b.year);
    return { name, rows, ...SERIES[name] };
  });

  const lines = series.map(({ rows, color, dashed }) =>
    svg("g", {}, [
      svg("path", {
        d: line(rows.map((row) => [x(row.year), y(row.solar_ratio)])),
        fill: "none",
        stroke: paint(color),
        "stroke-width": 1.75,
        "stroke-dasharray": dashed ? "5 4" : null,
        "stroke-linejoin": "round",
      }),
      ...rows.map((row) =>
        svg("circle", {
          cx: x(row.year),
          cy: y(row.solar_ratio),
          r: 2.75,
          fill: dashed ? paint("--paper") : paint(color),
          stroke: paint(color),
          "stroke-width": 1.5,
        }),
      ),
    ]),
  );

  return svg(
    "svg",
    {
      width,
      height,
      viewBox: `0 0 ${width} ${height}`,
      role: "img",
      "aria-label":
        "Published solar output over Eurostat gross production, 2019 to 2024: the German and Danish totals stay near 1, their feed-in series fall below it, and éCO2mix stays near 1.",
    },
    [
      svg(
        "g",
        { class: "grid" },
        gridValues.map((v) =>
          svg("line", { x1: margin.left, x2: width - margin.right, y1: y(v), y2: y(v) }),
        ),
      ),
      svg("line", {
        class: "baseline",
        x1: margin.left,
        x2: width - margin.right,
        y1: y(1),
        y2: y(1),
      }),
      svg("g", { class: "axis" }, [
        ...gridValues.map((v) =>
          svg("text", { x: margin.left - 8, y: y(v) + 4, "text-anchor": "end" }, [number(v, 1)]),
        ),
        ...[2019, 2020, 2021, 2022, 2023, 2024].map((year) =>
          svg("text", { x: x(year), y: height - 4, "text-anchor": "middle" }, [
            narrow ? `’${String(year).slice(2)}` : String(year),
          ]),
        ),
      ]),
      svg("g", { class: "marks" }, lines),
    ],
  );
}

function renderMatrix({ cells, years, detection_threshold: threshold }) {
  const body = document.querySelector("#matrix tbody");
  body.replaceChildren(
    ...ORDER.map((name) => {
      const cell = cells.find((row) => row.series === name);
      const { label, country, note } = SERIES[name];
      return html("tr", {}, [
        html("th", { scope: "row" }, [
          `${label}${note ? `, ${note}` : ""}`,
          html("small", {}, [country]),
        ]),
        html("td", { "data-label": "Self-consumption" }, [state(cell.self_consumption)]),
        html("td", { "data-label": "Distribution grids" }, [state(cell.distribution)]),
        html("td", { "data-label": "Producer" }, [verdict(cell.validation)]),
      ]);
    }),
  );
  const french = years.filter((row) => row.series === "eco2mix");
  const shares = french.map((row) => row.self_consumption_share);
  const dutch = years.filter((row) => row.series === "entsoe_nl").map((row) => row.wind_ratio);
  document.getElementById("matrix-note").textContent =
    `A self-consumed share is detected from ${percent(threshold.self_consumption, 1)} of solar ` +
    `output, and a share of wind capacity left out from ${percent(threshold.distribution, 1)}. ` +
    `In France it is ${percent(Math.min(...shares), 1)} to ${percent(Math.max(...shares), 1)} of ` +
    "solar output: RTE counts it, but the share is too small for yearly statistics to confirm, so " +
    `éCO2mix stays inconclusive. The Dutch series holds ${percent(Math.min(...dutch))} to ` +
    `${percent(Math.max(...dutch))} of Eurostat's wind output; no statement of its producer is ` +
    "archived yet to confirm why.";
}

function state(value) {
  return html("span", { class: `state state-${value}` }, [WORDS[value] ?? value]);
}

function verdict(value) {
  const words = { pass: "Agrees", fail: "Disagrees", inconclusive: "Not decided" };
  return html("span", { class: `verdict-chip verdict-${value}` }, [words[value]]);
}
