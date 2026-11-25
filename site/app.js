import * as Plot from "https://cdn.jsdelivr.net/npm/@observablehq/plot@0.6.17/+esm";

const REPOSITORY = "https://github.com/zkyasin/eu-solar-wind-audit";
const COUNTRIES = { FR: "France", DE: "Germany", DK1: "Denmark DK1", DK2: "Denmark DK2" };
const TECHNOLOGIES = { solar: "Solar", wind_onshore: "Onshore wind" };
const PERIODS = { 2019: "2019", 2020: "2020", 2021: "2021", 2022: "2022", 2023: "2023", 2024: "2024", 2025: "2025", all: "All" };
const MODELS = {
  raw: "Physics, nominal parameters",
  climatology: "Climatology of the published output",
  calibrated: "Calibrated",
  calibrated_strata: "Calibrated, with strata",
  calibrated_level: "Calibrated, with strata and yearly level",
};
const QUALITY_PERIODS = ["2019-2022", "2023-2025"];
const SERIES = {
  eco2mix: "éCO2mix, mainland France",
  smard: "SMARD, Germany",
  energy_charts_total: "Energy-Charts total power, Germany",
  energinet: "Energinet size classes, Denmark",
  energinet_total: "Energinet with self-consumption, Denmark",
  entsoe_nl: "ENTSO-E Transparency Platform, Netherlands",
};
const CONTENTS = {
  included: "Included",
  excluded: "Excluded",
  covered: "Covered",
  not_covered: "Not covered",
  inconclusive: "Inconclusive",
  ambiguous: "Ambiguous",
};
const CELLS = {
  complete: "Complete",
  partial: "Partial",
  feed_in_only: "Feed-in only",
  partial_feed_in: "Partial feed-in",
  inconclusive: "Inconclusive",
  ambiguous: "Ambiguous",
};
const KINDS = {
  gap: "Gap",
  negative: "Negative values",
  above_capacity: "Above capacity",
  frozen: "Frozen values",
  timestamp_shift: "Timestamp shift",
  clock_change_convention: "Clock change convention",
  clock_change_jump: "Clock change jump",
  level_break: "Level break",
  residual_outlier: "Outlier against the model",
  parts_gap: "Parts do not add up",
  cross_source_gap: "Two sources disagree",
};
// Regions of the éCO2mix regional series, by INSEE code.
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
const SOURCES = { odre: "éCO2mix", smard: "SMARD", energy_charts: "Energy-Charts", energinet: "Energinet" };
const SERIES_TECHNOLOGIES = {
  solar: "solar",
  wind: "wind",
  wind_onshore: "onshore wind",
  wind_offshore: "offshore wind",
};

const DANISH_SOLAR =
  "Danish solar adds the self-consumed output that Energinet estimates to the metered output of " +
  "the three size classes; before 2020 the whole output is booked as self-consumed.";
const SERIES_NOTES = {
  "FR wind_onshore":
    "French onshore wind has no values from June 2022 to the end of 2023: the national wind " +
    "series of éCO2mix includes offshore farms from June 2022, and its onshore series starts in 2024.",
  "DK1 solar": DANISH_SOLAR,
  "DK2 solar": DANISH_SOLAR,
};

const state = { country: "FR", technology: "solar", period: "2025" };
const documents = new Map();

const decimals = (digits) =>
  new Intl.NumberFormat("en-GB", { minimumFractionDigits: digits, maximumFractionDigits: digits });
const integer = new Intl.NumberFormat("en-GB", { maximumFractionDigits: 0 });
const dayFormat = new Intl.DateTimeFormat("en-GB", {
  day: "numeric",
  month: "short",
  year: "numeric",
  timeZone: "UTC",
});

function energy(gwh) {
  if (gwh === null || gwh === undefined || Number.isNaN(gwh)) return "–";
  if (Math.abs(gwh) >= 1000) return `${decimals(2).format(gwh / 1000)} TWh`;
  return `${decimals(Math.abs(gwh) >= 100 ? 0 : 1).format(gwh)} GWh`;
}

function percent(share, digits = 1) {
  return Number.isFinite(share) ? `${decimals(digits).format(share * 100)} %` : "–";
}

function value(number, digits) {
  return number === null || number === undefined ? "–" : decimals(digits).format(number);
}

function sum(rows, key) {
  return rows.reduce((total, row) => total + (row[key] ?? 0), 0);
}

function load(name) {
  if (!documents.has(name)) {
    documents.set(
      name,
      fetch(`data/${name}`).then((response) => {
        if (!response.ok) throw new Error(`${name}: HTTP ${response.status}`);
        return response.json();
      }),
    );
  }
  return documents.get(name);
}

function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}

function tiles(target, items) {
  target.replaceChildren(
    ...items.map(({ label, value: shown, detail }) => {
      const tile = element("div", undefined, "tile");
      tile.append(element("p", label, "label"), element("p", shown, "value"));
      if (detail) tile.append(element("p", detail, "detail"));
      return tile;
    }),
  );
}

// Writes a table; `columns` give a label, a function of the row and whether it is a number.
function table(target, columns, rows) {
  target.replaceChildren();
  const head = target.createTHead().insertRow();
  for (const column of columns) {
    const header = element("th", column.label, column.number ? "number" : undefined);
    header.scope = "col";
    head.append(header);
  }
  const body = target.createTBody();
  for (const row of rows) {
    const line = body.insertRow();
    for (const column of columns) {
      const cell = line.insertCell();
      const shown = column.value(row);
      if (shown instanceof Node) cell.append(shown);
      else cell.textContent = shown ?? "";
      if (column.number) cell.className = "number";
    }
  }
}

function colors() {
  const style = getComputedStyle(document.documentElement);
  const read = (name) => style.getPropertyValue(name).trim();
  return {
    published: read("--published"),
    model: read("--model"),
    price: read("--price"),
    ink: read("--ink-2"),
    axis: read("--axis"),
    surface: read("--surface"),
  };
}

function chartStyle(palette) {
  return { background: "transparent", color: palette.ink, fontSize: "12px", fontFamily: "inherit" };
}

function segmented(name, options) {
  const container = document.querySelector(`.segmented[data-name="${name}"]`);
  for (const [option, label] of Object.entries(options)) {
    const input = document.createElement("input");
    Object.assign(input, { type: "radio", name, value: option, id: `${name}-${option}` });
    input.checked = state[name] === option;
    const tag = element("label", label);
    tag.htmlFor = input.id;
    container.append(input, tag);
  }
}

async function dailyRows() {
  const daily = await load(`daily-${state.country}-${state.technology}.json`);
  const rows = daily.day.map((day, index) => ({
    day,
    date: new Date(`${day}T00:00:00Z`),
    published: daily.published_gwh[index],
    model: daily.potential_gwh[index],
    hours: daily.negative_price_hours[index],
    heldBack: daily.held_back_gwh[index],
  }));
  return state.period === "all" ? rows : rows.filter((row) => row.day.startsWith(state.period));
}

async function renderExplorer() {
  const rows = await dailyRows();
  renderTotals(rows);
  renderDailyCharts(rows);
  renderDailyTable(rows);
  await renderQuality();
  const note = document.getElementById("series-note");
  const text = SERIES_NOTES[`${state.country} ${state.technology}`];
  note.hidden = !text;
  note.textContent = text ?? "";
}

function renderTotals(rows) {
  const both = rows.filter((row) => row.published !== null && row.model !== null);
  const published = sum(both, "published");
  const model = sum(both, "model");
  tiles(document.getElementById("totals"), [
    { label: "Published", value: both.length ? energy(published) : "–", detail: `${integer.format(both.length)} days with both values` },
    { label: "Model", value: both.length ? energy(model) : "–", detail: "Same days" },
    { label: "Published over model", value: both.length ? percent(published / model) : "–", detail: "Same days" },
    {
      label: "Negative-price hours",
      value: integer.format(sum(rows, "hours")),
      // Without a published reference, years below the minimal detectable effect have no estimate.
      detail: rows.some((row) => row.hours > 0 && row.heldBack === null)
        ? "Output held back at them: not estimated below the minimal detectable effect"
        : `Output held back at them: ${energy(sum(rows, "heldBack"))}`,
    },
  ]);
}

function renderDailyCharts(rows) {
  const palette = colors();
  const daily = document.getElementById("daily-chart");
  const prices = document.getElementById("price-chart");
  if (!rows.length) {
    daily.replaceChildren();
    prices.replaceChildren();
    return;
  }
  const width = daily.clientWidth || 800;
  const domain = [rows[0].date, new Date(rows.at(-1).date.getTime() + 86400000)];
  const strokeWidth = rows.length > 400 ? 1 : 2;
  const describe = (row) =>
    [
      dayFormat.format(row.date),
      `Published ${row.published === null ? "–" : `${decimals(1).format(row.published)} GWh`}`,
      `Model ${row.model === null ? "–" : `${decimals(1).format(row.model)} GWh`}`,
      row.hours ? `Negative-price hours ${row.hours}` : null,
    ]
      .filter(Boolean)
      .join("\n");
  daily.replaceChildren(
    Plot.plot({
      width,
      height: 320,
      marginLeft: 52,
      marginRight: 12,
      style: chartStyle(palette),
      x: { type: "utc", domain, label: null },
      y: { label: "GWh per day", grid: true, nice: true, zero: true },
      marks: [
        Plot.ruleY([0], { stroke: palette.axis }),
        Plot.lineY(rows, { x: "date", y: "model", stroke: palette.model, strokeWidth }),
        Plot.lineY(rows, { x: "date", y: "published", stroke: palette.published, strokeWidth }),
        Plot.ruleX(rows, Plot.pointerX({ x: "date", stroke: palette.axis })),
        Plot.tip(
          rows,
          Plot.pointerX({ x: "date", y: (row) => row.model ?? row.published ?? 0, title: describe }),
        ),
      ],
    }),
  );
  prices.replaceChildren(
    Plot.plot({
      width,
      height: 110,
      marginLeft: 52,
      marginRight: 12,
      style: chartStyle(palette),
      x: { type: "utc", domain, label: null },
      y: { label: "Hours", grid: true, ticks: 3, domain: [0, 24], clamp: true },
      marks: [
        Plot.rectY(rows, { x: "date", interval: "day", y: "hours", fill: palette.price }),
        Plot.ruleY([0], { stroke: palette.axis }),
        Plot.tip(
          rows.filter((row) => row.hours > 0),
          Plot.pointerX({ x: "date", y: "hours", title: describe }),
        ),
      ],
    }),
  );
}

function renderDailyTable(rows) {
  const details = document.getElementById("daily-table");
  const draw = () => {
    const target = element("table");
    table(
      target,
      [
        { label: "Day", value: (row) => row.day },
        { label: "Published, GWh", value: (row) => value(row.published, 1), number: true },
        { label: "Model, GWh", value: (row) => value(row.model, 1), number: true },
        { label: "Negative-price hours", value: (row) => integer.format(row.hours), number: true },
        { label: "Held back, GWh", value: (row) => value(row.heldBack, 1), number: true },
      ],
      rows,
    );
    details.querySelector(".table-scroll").replaceChildren(target);
  };
  details.ontoggle = () => details.open && draw();
  if (details.open) draw();
}

async function renderQuality() {
  const quality = await load("quality.json");
  const pick = (model, period, metric) =>
    quality.find(
      (row) =>
        row.country === state.country &&
        row.technology === state.technology &&
        row.model === model &&
        row.period === period,
    )?.[metric];
  table(
    document.getElementById("quality"),
    [
      { label: "Model", value: (model) => MODELS[model] },
      ...QUALITY_PERIODS.flatMap((period) => [
        { label: `RMSE, ${period}`, value: (model) => value(pick(model, period, "rmse_cf"), 3), number: true },
        { label: `Correlation, ${period}`, value: (model) => value(pick(model, period, "correlation"), 3), number: true },
      ]),
    ],
    Object.keys(MODELS),
  );
}

function seriesLabel(series) {
  const parts = series.split(".");
  const source = SOURCES[parts[0]] ?? parts[0];
  // A finding on a whole dataset, such as its clock-change convention.
  if (parts.length === 2) return `${source}, dataset ${parts[1]}`;
  const area = parts[1];
  const region = area.startsWith("FR-") ? REGIONS[area.slice(3)] : null;
  const place = region ?? COUNTRIES[area] ?? area;
  const technology = SERIES_TECHNOLOGIES[parts.at(-1)] ?? parts.at(-1);
  return `${source}, ${place}, ${technology}`;
}

function utc(timestamp) {
  return `${timestamp.slice(0, 16).replace("T", " ")} UTC`;
}

async function renderFindings() {
  const findings = await load("findings.json");
  const kinds = Object.keys(KINDS).filter((kind) => findings.some((row) => row.kind === kind));
  table(
    document.getElementById("finding-counts"),
    [
      { label: "Kind", value: (kind) => KINDS[kind] },
      ...Object.entries(COUNTRIES).map(([country, name]) => ({
        label: name,
        value: (kind) => integer.format(findings.filter((row) => row.kind === kind && row.country === country).length),
        number: true,
      })),
    ],
    kinds,
  );
  const form = document.getElementById("finding-filters");
  const options = (select, entries) =>
    select.replaceChildren(
      ...entries.map(([option, label]) => Object.assign(element("option", label), { value: option })),
    );
  options(form.elements.country, [["all", "All"], ...Object.entries(COUNTRIES)]);
  options(form.elements.kind, [["all", "All"], ...kinds.map((kind) => [kind, KINDS[kind]])]);
  const draw = () => {
    const country = form.elements.country.value;
    const kind = form.elements.kind.value;
    const shown = findings
      .filter((row) => (country === "all" || row.country === country) && (kind === "all" || row.kind === kind))
      .sort((a, b) => a.start.localeCompare(b.start));
    table(
      document.getElementById("findings"),
      [
        { label: "Start", value: (row) => utc(row.start) },
        { label: "End", value: (row) => utc(row.end) },
        { label: "Series", value: (row) => seriesLabel(row.series) },
        { label: "Kind", value: (row) => KINDS[row.kind] ?? row.kind },
        { label: "Score", value: (row) => value(row.score, 2), number: true },
      ],
      shown,
    );
  };
  form.addEventListener("change", draw);
  draw();
}

function status(validation) {
  if (!validation) return "";
  return element("span", validation[0].toUpperCase() + validation.slice(1), `status ${validation}`);
}

function content(row, dimension) {
  const cell = element("div");
  cell.append(element("div", CONTENTS[row[dimension]] ?? row[dimension]));
  const documented = row[`${dimension}_documented`];
  const detail = `Level: ${(CONTENTS[row[`${dimension}_level`]] ?? "–").toLowerCase()}; producer: ${
    documented ? CONTENTS[documented].toLowerCase() : "no statement"
  }`;
  cell.append(element("div", detail, "meta"));
  return cell;
}

async function renderMatrix() {
  const perimeter = await load("perimeter.json");
  document.getElementById("content-intro").textContent =
    "Whether each series includes the output that producers consume on site, and whether it covers " +
    "the plants on the distribution grids. A dimension is decided only when the yearly level against " +
    "Eurostat and the statement of the producer agree. A self-consumed share is detected from " +
    `${percent(perimeter.detection_threshold.self_consumption)} of solar output, a share of wind ` +
    `capacity left out from ${percent(perimeter.detection_threshold.distribution)}.`;
  table(
    document.getElementById("matrix"),
    [
      { label: "Series", value: (row) => SERIES[row.series] ?? row.series },
      { label: "Self-consumption", value: (row) => content(row, "self_consumption") },
      { label: "Distribution grids", value: (row) => content(row, "distribution") },
      { label: "Cell", value: (row) => CELLS[row.cell] ?? row.cell },
      {
        label: "Expected",
        value: (row) =>
          (row.expected_cell ?? "")
            .split(" or ")
            .map((cell, index) => (index ? (CELLS[cell] ?? cell).toLowerCase() : (CELLS[cell] ?? cell)))
            .join(" or "),
      },
      { label: "Validation", value: (row) => status(row.validation) },
    ],
    [...perimeter.cells].sort(
      (a, b) => Object.keys(SERIES).indexOf(a.series) - Object.keys(SERIES).indexOf(b.series),
    ),
  );
}

async function renderModulation() {
  const runs = await load("modulation.json");
  const palette = colors();
  const container = document.getElementById("modulation");
  const yearly = runs.flatMap((run) =>
    run.periods
      .filter((period) => period.period.length === 4)
      .map((period) => ({ ...period, country: run.country, technology: run.technology, reference: run.reference })),
  );
  container.replaceChildren(
    ...runs.map((run) => {
      const panel = element("figure", undefined, "chart");
      panel.append(element("h3", `${COUNTRIES[run.country]}, ${TECHNOLOGIES[run.technology].toLowerCase()}`));
      const years = yearly.filter((row) => row.country === run.country && row.technology === run.technology);
      if (years.length === 0) {
        panel.append(element("figcaption", "No published reference, and no year above the minimal detectable effect."));
        return panel;
      }
      const plot = element("div");
      panel.append(plot);
      if (run.verdict) panel.append(element("figcaption", `Verdict of the evaluation plan: ${run.verdict}.`));
      else if (run.reference) panel.append(element("figcaption", `Reference: ${run.reference}.`));
      else panel.append(element("figcaption", "No published reference. Years below the minimal detectable effect are left out."));
      requestAnimationFrame(() =>
        plot.replaceChildren(
          Plot.plot({
            width: plot.clientWidth || 360,
            height: 220,
            marginLeft: 52,
            marginRight: 8,
            style: chartStyle(palette),
            x: { type: "band", label: null, padding: 0.4 },
            y: { label: "GWh", grid: true, nice: true, zero: true },
            marks: [
              Plot.ruleY([0], { stroke: palette.axis }),
              Plot.ruleX(years, { x: "period", y1: "low_gwh", y2: "high_gwh", stroke: palette.model, strokeWidth: 2 }),
              Plot.tickY(
                years.filter((row) => row.reference_gwh !== null),
                { x: "period", y: "reference_gwh", stroke: palette.published, strokeWidth: 2 },
              ),
              Plot.dot(years, {
                x: "period",
                y: "estimate_gwh",
                r: 4,
                fill: palette.model,
                stroke: palette.surface,
                strokeWidth: 2,
              }),
              Plot.tip(
                years,
                Plot.pointerX({
                  x: "period",
                  y: "estimate_gwh",
                  title: (row) =>
                    [
                      row.period,
                      `Estimate ${energy(row.estimate_gwh)}`,
                      `90 % interval ${energy(row.low_gwh)} to ${energy(row.high_gwh)}`,
                      row.reference_gwh === null ? null : `${row.reference} ${energy(row.reference_gwh)}`,
                    ]
                      .filter(Boolean)
                      .join("\n"),
                }),
              ),
            ],
          }),
        ),
      );
      return panel;
    }),
  );
  const target = element("table");
  table(
    target,
    [
      { label: "Country", value: (row) => COUNTRIES[row.country] },
      { label: "Technology", value: (row) => TECHNOLOGIES[row.technology] },
      { label: "Year", value: (row) => row.period },
      { label: "Estimate", value: (row) => energy(row.estimate_gwh), number: true },
      { label: "90 % interval", value: (row) => `${energy(row.low_gwh)} to ${energy(row.high_gwh)}`, number: true },
      { label: "Reference", value: (row) => (row.reference_gwh === null ? "–" : `${row.reference} ${energy(row.reference_gwh)}`), number: true },
    ],
    yearly,
  );
  document.querySelector("#modulation-table .table-scroll").replaceChildren(target);
}

async function renderResults() {
  const [quality, findings, perimeter, runs] = await Promise.all([
    load("quality.json"),
    load("findings.json"),
    load("perimeter.json"),
    load("modulation.json"),
  ]);
  const errors = quality
    .filter((row) => row.model === "calibrated_level" && row.period === "2023-2025")
    .map((row) => row.rmse_cf);
  const validated = perimeter.cells.filter((row) => row.validation);
  const count = (outcome) => validated.filter((row) => row.validation === outcome).length;
  const solar = runs
    .find((run) => run.country === "FR" && run.technology === "solar")
    .periods.find((period) => period.period === "2025");
  tiles(document.getElementById("results"), [
    {
      label: "Hourly error of the model, 2023-2025",
      value: `${value(Math.min(...errors), 3)} to ${value(Math.max(...errors), 3)}`,
      detail: `RMSE in capacity factor, held-out years: ${Object.values(COUNTRIES).join(", ")}`,
    },
    {
      label: "Findings in the published series, 2019-2025",
      value: integer.format(findings.length),
      detail: Object.entries(COUNTRIES)
        .map(([country, name]) => `${name} ${findings.filter((row) => row.country === country).length}`)
        .join(", "),
    },
    {
      label: "Validation series of the content matrix",
      value: `${count("pass")} of ${validated.length} pass`,
      detail: `${count("fail")} fail, ${count("inconclusive")} inconclusive`,
    },
    {
      label: "Solar output held back in France, 2025",
      value: energy(solar.estimate_gwh),
      detail: `90 % interval ${energy(solar.low_gwh)} to ${energy(solar.high_gwh)}; RTE ${energy(solar.reference_gwh)}`,
    },
  ]);
}

async function renderMeta() {
  const meta = await load("meta.json");
  const line = document.getElementById("meta");
  const commit = element("a", meta.commit);
  commit.href = `${REPOSITORY}/commit/${meta.commit}`;
  const code = element("a", "Code and method on GitHub");
  code.href = REPOSITORY;
  line.replaceChildren(`Data exported on ${meta.exported} from commit `, commit, ". ", code, ".");
}

let resizing;
function redraw() {
  clearTimeout(resizing);
  resizing = setTimeout(() => {
    renderExplorer();
    renderModulation();
  }, 150);
}

async function main() {
  // A link can open the explorer on a series, such as ?country=DK1&technology=solar&period=2025.
  const params = new URLSearchParams(location.search);
  for (const [name, options] of [["country", COUNTRIES], ["technology", TECHNOLOGIES], ["period", PERIODS]]) {
    const chosen = params.get(name);
    if (chosen !== null && Object.hasOwn(options, chosen)) state[name] = chosen;
  }
  segmented("country", COUNTRIES);
  segmented("technology", TECHNOLOGIES);
  segmented("period", PERIODS);
  document.getElementById("filters").addEventListener("change", (event) => {
    state[event.target.name] = event.target.value;
    renderExplorer();
  });
  window.addEventListener("resize", redraw);
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", redraw);
  await Promise.all([
    renderMeta(),
    renderResults(),
    renderExplorer(),
    renderFindings(),
    renderMatrix(),
    renderModulation(),
  ]);
}

main().catch((error) => {
  document.getElementById("meta").textContent = `The data could not be loaded: ${error.message}`;
});
