/** Number and date formats of the page: English, with a thin space before units. */

const formats = new Map();

function format(digits) {
  if (!formats.has(digits)) {
    formats.set(
      digits,
      new Intl.NumberFormat("en-GB", {
        minimumFractionDigits: digits,
        maximumFractionDigits: digits,
      }),
    );
  }
  return formats.get(digits);
}

export function number(value, digits = 0) {
  return value === null || value === undefined ? "–" : format(digits).format(value);
}

/** An energy in GWh, written in TWh from 1,000 GWh. */
export function energy(gwh) {
  if (gwh === null || gwh === undefined) return "–";
  const size = Math.abs(gwh);
  if (size >= 1000) return `${number(gwh / 1000, 2)} TWh`;
  if (size >= 100) return `${number(gwh)} GWh`;
  if (size >= 10) return `${number(gwh, 1)} GWh`;
  return `${number(gwh, 2)} GWh`;
}

export function percent(ratio, digits = 0) {
  return `${number(ratio * 100, digits)} %`;
}

const dayFormat = new Intl.DateTimeFormat("en-GB", {
  day: "numeric",
  month: "short",
  year: "numeric",
  timeZone: "UTC",
});

/** A local day, `YYYY-MM-DD`, written out. */
export function day(text) {
  return dayFormat.format(new Date(`${text}T00:00:00Z`));
}

export const MONTHS = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"];
