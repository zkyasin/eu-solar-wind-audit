/** Loading of the data documents that `eu-solar-wind-audit export-site-data` writes. */

const documents = new Map();

export function load(name) {
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

/** A daily series as one record per day; years are local years, as the days are. */
export async function daily(country, technology) {
  const columns = await load(`daily-${country}-${technology}.json`);
  return columns.day.map((day, index) => ({
    day,
    year: Number(day.slice(0, 4)),
    published: columns.published_gwh[index],
    potential: columns.potential_gwh[index],
    negativeHours: columns.negative_price_hours[index],
    heldBack: columns.held_back_gwh[index],
  }));
}

export const COUNTRIES = {
  FR: "France",
  DE: "Allemagne",
  DK1: "Danemark DK1",
  DK2: "Danemark DK2",
};
export const TECHNOLOGIES = { solar: "Solaire", wind_onshore: "Éolien terrestre" };
export const YEARS = [2019, 2020, 2021, 2022, 2023, 2024, 2025];
