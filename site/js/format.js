/**
 * Number and date formats of the page, in French: a decimal comma, and a narrow no-break space
 * between thousands and before units.
 */

/** Narrow no-break space, the French thin space before a unit. */
export const THIN = "\u202f";

const formats = new Map();

function format(digits) {
  if (!formats.has(digits)) {
    formats.set(
      digits,
      new Intl.NumberFormat("fr-FR", {
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

/** An energy in GWh, written in TWh from 1 000 GWh. */
export function energy(gwh) {
  if (gwh === null || gwh === undefined) return "–";
  const size = Math.abs(gwh);
  if (size >= 1000) return `${number(gwh / 1000, 2)}${THIN}TWh`;
  if (size >= 100) return `${number(gwh)}${THIN}GWh`;
  if (size >= 10) return `${number(gwh, 1)}${THIN}GWh`;
  return `${number(gwh, 2)}${THIN}GWh`;
}

export function percent(ratio, digits = 0) {
  return `${number(ratio * 100, digits)}${THIN}%`;
}

const dayFormat = new Intl.DateTimeFormat("fr-FR", {
  day: "numeric",
  month: "long",
  year: "numeric",
  timeZone: "UTC",
});

/** A local day, `YYYY-MM-DD`, written out. */
export function day(text) {
  return dayFormat.format(new Date(`${text}T00:00:00Z`));
}

/** Initials of the months, January to December. */
export const MONTHS = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"];
