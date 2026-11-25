/** Chapter 02: the findings of the detectors on a timeline, one row per group of series. */

import { load } from "./data.js";
import { day, number } from "./format.js";
import { html, linear, paint, responsive, svg, tip, tipLine } from "./plot.js";

const FAMILIES = {
  time: { label: "Missing or shifted hours", color: "--published" },
  values: { label: "Impossible values", color: "--flag" },
  model: { label: "Against the model", color: "--model" },
  consistency: { label: "Against other series", color: "--third" },
};

const KINDS = {
  gap: ["time", "Missing hours"],
  timestamp_shift: ["time", "Shifted timestamps"],
  clock_change_jump: ["time", "Jump at a clock change"],
  clock_change_convention: ["time", "Clock-change convention"],
  negative: ["values", "Below zero"],
  above_capacity: ["values", "Above installed capacity"],
  frozen: ["values", "Frozen value"],
  residual_outlier: ["model", "Outlier against the model"],
  level_break: ["model", "Break in level"],
  parts_gap: ["consistency", "Total against the sum of its parts"],
  cross_source_gap: ["consistency", "Against a second source"],
};

const REGIONS = {
  11: "Île-de-France",
  24: "Centre-Val de Loire",
  27: "Bourgogne-Franche-Comté",
  28: "Normandie",
  32: "Hauts-de-France",
  44: "Grand Est",
  52: "Pays de la Loire",
  53: "Bretagne",
  75: "Nouvelle-Aquitaine",
  76: "Occitanie",
  84: "Auvergne-Rhône-Alpes",
  93: "Provence-Alpes-Côte d'Azur",
};

const TECHNOLOGY = {
  solar: "solar",
  wind: "wind",
  wind_onshore: "onshore wind",
  wind_offshore: "offshore wind",
};

const ROWS = [
  ["FR", "national", "France"],
  ["FR", "regional", "12 regions"],
  ["DE", "national", "Germany"],
  ["DE", "regional", "4 control areas"],
];

const START = Date.UTC(2019, 0, 1);
const END = Date.UTC(2026, 0, 1);

/** The row, the readable name and the technology of a series such as `odre.FR-44.solar`. */
function describe(series) {
  const [source, area, technology = ""] = series.split(".");
  if (source === "energy_charts") {
    const tech = series.split(".").at(-1);
    return {
      scope: "national",
      name: `Germany ${TECHNOLOGY[tech]}, Energy-Charts`,
      technology: tech,
    };
  }
  if (!technology) return { scope: null };
  const name =
    source === "odre"
      ? area === "FR"
        ? `France ${TECHNOLOGY[technology]}`
        : `${REGIONS[area.slice(3)]} ${TECHNOLOGY[technology]}`
      : area === "DE"
        ? `Germany ${TECHNOLOGY[technology]}`
        : `${area} ${TECHNOLOGY[technology]}`;
  const national = area === "FR" || area === "DE";
  return { scope: national ? "national" : "regional", name, technology };
}

export async function renderAnomalies() {
  const findings = (await load("findings.json")).map((finding) => ({
    ...finding,
    ...describe(finding.series),
    family: KINDS[finding.kind][0],
    from: Math.max(START, Date.parse(finding.start)),
    to: Math.min(END, Date.parse(finding.end)),
  }));
  renderReadout(findings);
  renderLegend(findings);

  const placed = findings.filter((finding) => finding.scope);
  const plot = document.getElementById("anomaly-plot");
  const tipper = tip(plot);
  responsive(plot, (width) => drawTimeline(placed, width));

  plot.addEventListener("pointermove", (event) => {
    const mark = event.target.closest?.("[data-index]");
    if (!mark) {
      tipper.hide();
      return;
    }
    const finding = placed[Number(mark.dataset.index)];
    const box = plot.getBoundingClientRect();
    const first = finding.start.slice(0, 10);
    const last = finding.end.slice(0, 10);
    tipper.show(
      [
        html("strong", {}, [finding.name]),
        tipLine(paint(FAMILIES[finding.family].color), KINDS[finding.kind][1]),
        html("span", {}, [first === last ? day(first) : `${day(first)} to ${day(last)}`]),
      ],
      event.clientX - box.left,
      event.clientY - box.top,
    );
  });
  plot.addEventListener("pointerleave", () => tipper.hide());
}

function renderReadout(findings) {
  const count = (country) => findings.filter((finding) => finding.country === country).length;
  const cells = [
    [
      "Findings, 2019–2025",
      number(findings.length),
      `France ${count("FR")}, Germany ${count("DE")}`,
    ],
    ["Kinds of finding", String(Object.keys(KINDS).length), "from gaps to breaks in level"],
    ["Checked as tickets", "11", "7 in France, 4 in Germany"],
  ];
  document
    .getElementById("anomaly-readout")
    .replaceChildren(
      ...cells.map(([label, value, detail]) =>
        html("div", {}, [
          html("dt", {}, [label]),
          html("dd", {}, [value, html("small", {}, [detail])]),
        ]),
      ),
    );
}

function renderLegend(findings) {
  document.getElementById("anomaly-legend").replaceChildren(
    ...Object.entries(FAMILIES).map(([family, { label, color }]) => {
      const total = findings.filter((finding) => finding.family === family).length;
      return html("li", {}, [
        svg("svg", { class: "key", viewBox: "0 0 24 12", "aria-hidden": "true" }, [
          svg("rect", { x: 9, y: 0, width: 6, height: 12, rx: 1, fill: paint(color) }),
        ]),
        `${label} · ${total}`,
      ]);
    }),
  );
}

function drawTimeline(findings, width) {
  const narrow = width < 560;
  const margin = { top: 4, right: 6, bottom: 24, left: narrow ? 0 : 118 };
  const band = narrow ? 52 : 46;
  const labelSpace = narrow ? 18 : 0;
  const lanes = { solar: 0, wind: 1, wind_onshore: 1, wind_offshore: 1 };
  const height = margin.top + ROWS.length * (band + labelSpace) + margin.bottom;
  const x = linear([START, END], [margin.left, width - margin.right]);
  const top = (row) => margin.top + row * (band + labelSpace) + labelSpace;
  const lane = (band - 10) / 2;

  const years = [];
  for (let year = 2019; year <= 2026; year++) years.push(Date.UTC(year, 0, 1));
  const grid = svg(
    "g",
    { class: "grid" },
    years.map((t) =>
      svg("line", { x1: x(t), x2: x(t), y1: margin.top, y2: height - margin.bottom }),
    ),
  );
  const axis = svg(
    "g",
    { class: "axis" },
    years
      .slice(0, -1)
      .map((t) =>
        svg(
          "text",
          { x: (x(t) + x(t + 365.25 * 864e5)) / 2, y: height - 6, "text-anchor": "middle" },
          [
            narrow
              ? `’${String(new Date(t).getUTCFullYear()).slice(2)}`
              : String(new Date(t).getUTCFullYear()),
          ],
        ),
      ),
  );
  const labels = ROWS.flatMap(([, , label], row) => {
    if (narrow) return [svg("text", { class: "label", x: 0, y: top(row) - 5 }, [label])];
    return [
      svg("text", { class: "label", x: 0, y: top(row) + band / 2 + 4 }, [label]),
      ...["solar", "wind"].map((technology, k) =>
        svg(
          "text",
          {
            class: "axis-note",
            x: margin.left - 8,
            y: top(row) + 3 + k * (lane + 4) + lane / 2 + 3.5,
            "text-anchor": "end",
          },
          [technology],
        ),
      ),
    ];
  });
  const lanesBackground = ROWS.flatMap((_, row) =>
    [0, 1].map((k) =>
      svg("rect", {
        class: "lane",
        x: margin.left,
        y: top(row) + 3 + k * (lane + 4),
        width: width - margin.left - margin.right,
        height: lane,
        rx: 2,
      }),
    ),
  );
  const marks = findings.map((finding, index) => {
    const row = ROWS.findIndex(
      ([country, scope]) => country === finding.country && scope === finding.scope,
    );
    const k = lanes[finding.technology] ?? 0;
    const x0 = x(finding.from);
    const w = Math.max(2, x(finding.to) - x0);
    return svg("rect", {
      "data-index": index,
      x: x0 - (w === 2 ? 1 : 0),
      y: top(row) + 3 + k * (lane + 4),
      width: w,
      height: lane,
      rx: 1,
      fill: paint(FAMILIES[finding.family].color),
      "fill-opacity": 0.85,
    });
  });
  return svg(
    "svg",
    {
      width,
      height,
      viewBox: `0 0 ${width} ${height}`,
      role: "img",
      "aria-label": `${findings.length} findings of the detectors in the solar and wind series of France and Germany, 2019 to 2025, by group of series and kind.`,
    },
    [
      grid,
      svg("g", {}, lanesBackground),
      svg("g", { class: "marks" }, marks),
      axis,
      svg("g", { class: "axis" }, labels),
    ],
  );
}
