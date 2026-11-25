/** Chapter 02: the findings of the detectors on a timeline, one row per group of series. */

import { load } from "./data.js";
import { day, number } from "./format.js";
import { html, linear, paint, responsive, svg, tip, tipLine } from "./plot.js";

const FAMILIES = {
  time: { label: "Heures manquantes ou décalées", color: "--published" },
  values: { label: "Valeurs impossibles", color: "--flag" },
  model: { label: "Écart au modèle", color: "--model" },
  consistency: { label: "Écart à d’autres séries", color: "--third" },
};

const KINDS = {
  gap: ["time", "Heures manquantes"],
  timestamp_shift: ["time", "Horodatage décalé"],
  clock_change_jump: ["time", "Saut au changement d’heure"],
  clock_change_convention: ["time", "Convention de changement d’heure"],
  negative: ["values", "Valeur négative"],
  above_capacity: ["values", "Au-dessus de la capacité installée"],
  frozen: ["values", "Valeur figée"],
  residual_outlier: ["model", "Valeur aberrante par rapport au modèle"],
  level_break: ["model", "Rupture de niveau"],
  parts_gap: ["consistency", "Total différent de la somme de ses parties"],
  cross_source_gap: ["consistency", "Écart à une seconde source"],
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
  solar: "Solaire",
  wind: "Éolien",
  wind_onshore: "Éolien terrestre",
  wind_offshore: "Éolien en mer",
};

const ROWS = [
  ["FR", "national", "France"],
  ["FR", "regional", "12 régions"],
  ["DE", "national", "Allemagne"],
  ["DE", "regional", "4 zones de réglage"],
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
      name: `${TECHNOLOGY[tech]}, Allemagne (Energy-Charts)`,
      technology: tech,
    };
  }
  if (!technology) return { scope: null };
  const name =
    source === "odre"
      ? area === "FR"
        ? `${TECHNOLOGY[technology]}, France`
        : `${TECHNOLOGY[technology]}, ${REGIONS[area.slice(3)]}`
      : area === "DE"
        ? `${TECHNOLOGY[technology]}, Allemagne`
        : `${TECHNOLOGY[technology]}, ${area}`;
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
        html("span", {}, [first === last ? day(first) : `du ${day(first)} au ${day(last)}`]),
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
      "Constats, 2019–2025",
      number(findings.length),
      `France ${count("FR")}, Allemagne ${count("DE")}`,
    ],
    ["Types de constat", String(Object.keys(KINDS).length), "des trous aux ruptures de niveau"],
    ["Vérifiés en tickets", "11", "7 en France, 4 en Allemagne"],
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
      ...["solaire", "éolien"].map((technology, k) =>
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
      "aria-label": `${findings.length} constats des détecteurs dans les séries solaires et éoliennes de France et d’Allemagne, de 2019 à 2025, par groupe de séries et par type.`,
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
