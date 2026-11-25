/** Chapter 04: output held back at negative prices, each year, beside the published references. */

import { COUNTRIES, load, TECHNOLOGIES, YEARS } from "./data.js";
import { energy, number, THIN } from "./format.js";
import { html, linear, options, paint, responsive, svg, ticks } from "./plot.js";

const REFERENCES = { FR: "RTE", DE: "Hirth (2026)" };

/** The verdicts of the evaluation plan, as the data documents name them. */
const VERDICTS = { "concordant in volume only": "Concordant en volume seulement" };

export async function renderModulation() {
  const runs = await load("modulation.json");
  let country = "FR";
  const container = document.getElementById("modulation-plots");
  const panels = Object.keys(TECHNOLOGIES).map((technology) => {
    const plot = html("div", { class: "plot" });
    const panel = html("div", { class: "panel" }, [
      html("p", { class: "panel-title" }, [TECHNOLOGIES[technology]]),
      plot,
    ]);
    return { technology, plot, panel };
  });
  container.replaceChildren(...panels.map(({ panel }) => panel));

  const draws = panels.map(({ technology, plot }) =>
    responsive(plot, (width) => {
      const run = runs.find((r) => r.country === country && r.technology === technology);
      return drawYears(run, width);
    }),
  );

  const update = () => {
    for (const draw of draws) draw();
    renderReadout(runs, country);
    renderVerdict(runs, country);
    document.getElementById("modulation-caption").textContent =
      country === "FR"
        ? "Production non injectée chaque année, en GWh. Aucun prix négatif en France en 2022, et la série éolienne française de 2023 inclut l’éolien en mer."
        : "Production non injectée chaque année, en GWh. Les volumes éoliens de Hirth incluent l’éolien en mer.";
    document.getElementById("modulation-reference-key").textContent =
      country === "FR" ? "Volume publié par RTE" : "Écrêtement par le marché, Hirth (2026)";
  };
  options("modulation-country", COUNTRIES, country, (value) => {
    country = value;
    update();
  });
  update();
}

function yearly(run) {
  return run.periods.filter((period) => period.period.length === 4);
}

function drawYears(run, width) {
  const height = width < 420 ? 220 : 250;
  const margin = { top: 22, right: 4, bottom: 22, left: 40 };
  const periods = yearly(run);
  const top = Math.max(...periods.flatMap((p) => [p.high_gwh, p.reference_gwh ?? 0]), 1) * 1.06;
  const band = (width - margin.left - margin.right) / YEARS.length;
  const x = (year) => margin.left + (YEARS.indexOf(year) + 0.5) * band;
  const y = linear([0, top], [height - margin.bottom, margin.top]);
  const gridValues = ticks(0, top, 4);
  const half = Math.min(14, band * 0.22);

  const marks = YEARS.map((year) => {
    const period = periods.find((p) => Number(p.period) === year);
    const cx = x(year);
    if (!period) {
      return svg("text", { class: "axis-note", x: cx, y: y(0) - 6, "text-anchor": "middle" }, [
        "–",
      ]);
    }
    const low = Math.max(0, period.low_gwh);
    const children = [
      svg("rect", {
        x: cx - half,
        y: y(period.high_gwh),
        width: half * 2,
        height: Math.max(1, y(low) - y(period.high_gwh)),
        fill: "url(#hatch)",
        rx: 1.5,
      }),
      svg("line", {
        x1: cx - half - 2,
        x2: cx + half + 2,
        y1: y(period.estimate_gwh),
        y2: y(period.estimate_gwh),
        stroke: paint("--model"),
        "stroke-width": 2.5,
      }),
    ];
    if (period.reference_gwh !== null) {
      children.push(
        svg("line", {
          x1: cx + half + 5,
          x2: cx + half + 5 + Math.min(14, band * 0.18),
          y1: y(period.reference_gwh),
          y2: y(period.reference_gwh),
          stroke: paint("--published"),
          "stroke-width": 2.5,
          "stroke-linecap": "round",
        }),
      );
    }
    if (year === YEARS.at(-1)) {
      children.push(
        svg(
          "text",
          { class: "label-strong", x: cx, y: y(period.high_gwh) - 8, "text-anchor": "middle" },
          [energy(period.estimate_gwh)],
        ),
      );
    }
    return svg("g", {}, children);
  });

  return svg(
    "svg",
    {
      width,
      height,
      viewBox: `0 0 ${width} ${height}`,
      role: "img",
      "aria-label": `${TECHNOLOGIES[run.technology]} : production non injectée aux prix négatifs, ${COUNTRIES[run.country]}, chaque année de 2019 à 2025, avec son intervalle à 90${THIN}% et la référence publiée.`,
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
        y1: y(0),
        y2: y(0),
      }),
      svg("g", { class: "axis" }, [
        ...gridValues.map((v) =>
          svg("text", { x: margin.left - 8, y: y(v) + 4, "text-anchor": "end" }, [number(v)]),
        ),
        ...YEARS.map((year) =>
          svg("text", { x: x(year), y: height - 4, "text-anchor": "middle" }, [
            width < 420 ? `’${String(year).slice(2)}` : String(year),
          ]),
        ),
      ]),
      svg("g", { class: "marks" }, marks),
    ],
  );
}

function latest(runs, country, technology) {
  const run = runs.find((r) => r.country === country && r.technology === technology);
  return { run, period: yearly(run).find((p) => p.period === String(YEARS.at(-1))) };
}

function renderReadout(runs, country) {
  const cells = Object.keys(TECHNOLOGIES).map((technology) => {
    const { period } = latest(runs, country, technology);
    const reference = period.reference_gwh;
    const detail =
      reference === null
        ? "sans référence"
        : `${REFERENCES[country]} ${energy(reference)}, ratio ${number(period.estimate_gwh / reference, 2)}`;
    return [`${TECHNOLOGIES[technology]}, ${YEARS.at(-1)}`, energy(period.estimate_gwh), detail];
  });
  document
    .getElementById("modulation-readout")
    .replaceChildren(
      ...cells.map(([label, value, detail]) =>
        html("div", {}, [
          html("dt", {}, [label]),
          html("dd", {}, [value, html("small", {}, [detail])]),
        ]),
      ),
    );
}

function renderVerdict(runs, country) {
  const node = document.getElementById("modulation-verdict");
  if (country === "FR") {
    const ratios = Object.keys(TECHNOLOGIES).map((technology) => {
      const { run } = latest(runs, country, technology);
      return ["2024", "2025"].map((year) => {
        const p = yearly(run).find((period) => period.period === year);
        return number(p.estimate_gwh / p.reference_gwh, 2);
      });
    });
    const verdict = runs.find((r) => r.country === "FR").verdict;
    node.replaceChildren(
      html("p", { class: "verdict-title" }, [VERDICTS[verdict] ?? verdict]),
      html("p", {}, [
        "Les ratios annuels à RTE de 2024 et 2025 restent entre 0,8 et 1,25, comme l’exige le " +
          `plan : ${ratios[0].join(" et ")} pour le solaire, ${ratios[1].join(" et ")} pour ` +
          "l’éolien terrestre. Les tests mensuels échouent, et le placebo sur les heures à 5 à " +
          `15${THIN}€/MWh, où le solaire publié dépasse le potentiel de 1,4 à 4,6${THIN}% cinq ` +
          "années sur six, indique que l’estimation solaire est peut-être sous-évaluée de " +
          "quelques pour cent.",
      ]),
    );
    return;
  }
  node.replaceChildren(
    html("p", { class: "verdict-title" }, [
      "Pas de verdict pour l’Allemagne, comme le prévoit le plan",
    ]),
    html("p", {}, [
      "La part du potentiel éolien terrestre non injectée reste à moins de 3 points du taux " +
        "d’écrêtement de Hirth chaque année depuis 2022, et les volumes représentent 79 à " +
        `90${THIN}% des siens, qui incluent aussi l’éolien en mer. Le solaire de 2025 vaut 2,4 ` +
        `fois ses 0,6${THIN}TWh : l’estimation retient aussi l’écrêtement dû aux congestions du ` +
        "réseau, et l’injection publiée baisse quand les batteries domestiques et " +
        "l’autoconsommation absorbent la production de midi.",
    ]),
  );
}
