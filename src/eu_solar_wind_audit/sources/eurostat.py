"""Annual gross electricity production from solar photovoltaics and wind, by country, from Eurostat
(dataset nrg_bal_peh, licence CC BY 4.0, source "Eurostat, nrg_bal_peh").

Gross production is the total electricity produced (Eurostat glossary, "Gross electricity
generation"), so it includes the output that producers consume on site. It is the reference level
of axis 3 of the evaluation plan, against which published series are checked for self-consumption
and for the distribution grids they cover. Wind is onshore and offshore together.

France is the national territory with Monaco and the overseas departments of Guadeloupe,
Martinique, Guyane, La Réunion and Mayotte (Eurostat metadata of France for energy production,
nrg_prd_simsqu_fr, section 3.7, read 2026-11-24), so it includes Corsica too.
"""

import io
import time
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import httpx
import pandas as pd
import pandera.pandas as pa
from pandera.typing import DataFrame, Series

URL = "https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/nrg_bal_peh"
CACHE_DIR = Path("data/raw/eurostat")
TIMEOUT_SECONDS = 60.0
RETRY_DELAYS_SECONDS = (5.0, 30.0)

# Countries of the evaluation plan, Scope.
COUNTRIES = ("AT", "BE", "DE", "DK", "ES", "FR", "IT", "NL", "PL", "PT")
# Standard International Energy Product Classification codes.
FUELS = {"solar": "RA420", "wind": "RA300"}
FIRST_YEAR = 2019
KEY = f"A.GEP.{'+'.join(FUELS.values())}.GWH.{'+'.join(COUNTRIES)}"
# Annual gross electricity production in GWh.
FIXED_DIMENSIONS = {"freq": "A", "nrg_bal": "GEP", "unit": "GWH"}


class GrossProduction(pa.DataFrameModel):
    """Gross production in GWh of one country, technology and year, with its Eurostat observation
    flag ("p" provisional, "e" estimated, ...), null when it has none."""

    country: Series[str] = pa.Field(isin=COUNTRIES)
    technology: Series[str] = pa.Field(isin=list(FUELS))
    year: Series[int] = pa.Field(ge=FIRST_YEAR)
    gwh: Series[float] = pa.Field(ge=0)
    flag: Series[str] = pa.Field(nullable=True)

    class Config:
        strict = True
        unique: ClassVar[list[str]] = ["country", "technology", "year"]


@dataclass(frozen=True)
class Release:
    """The values of a download and the time Eurostat last updated the dataset, naive, in
    Luxembourg time as the file gives it."""

    production: DataFrame[GrossProduction]
    updated: pd.Timestamp


def fetch_gross_production(
    *,
    client: httpx.Client | None = None,
    cache_dir: Path = CACHE_DIR,
    refresh: bool = False,
) -> Release:
    """Gross solar and wind production of the countries of the plan from FIRST_YEAR, from the
    cached download unless `refresh` is set."""
    path = cache_dir / "nrg_bal_peh.csv"
    if path.exists() and not refresh:
        return parse_gross_production(path.read_bytes())

    params = {"format": "SDMX-CSV", "startPeriod": str(FIRST_YEAR), "compressed": "false"}
    if client is None:
        with httpx.Client(timeout=TIMEOUT_SECONDS) as own_client:
            content = _get(own_client, f"{URL}/{KEY}/", params)
    else:
        content = _get(client, f"{URL}/{KEY}/", params)
    # Parsed before caching, so that a download that does not parse is never kept.
    release = parse_gross_production(content)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    partial.write_bytes(content)
    partial.replace(path)
    return release


def parse_gross_production(content: bytes) -> Release:
    """`Release` from an SDMX-CSV download; observations without a value are left out."""
    raw = pd.read_csv(io.BytesIO(content), dtype={"OBS_FLAG": "str", "geo": "str"})
    for column, expected in FIXED_DIMENSIONS.items():
        found = set(raw[column])
        if found != {expected}:
            raise ValueError(f"expected {column} {expected!r} only, found {sorted(found)}")
    technologies = raw["siec"].map({code: name for name, code in FUELS.items()})
    if technologies.isna().any():
        raise ValueError(f"unexpected fuels: {sorted(set(raw['siec'][technologies.isna()]))}")

    observed = raw["OBS_VALUE"].notna()
    production = pd.DataFrame(
        {
            "country": raw["geo"],
            "technology": technologies,
            "year": raw["TIME_PERIOD"].astype("int64"),
            "gwh": raw["OBS_VALUE"].astype("float64"),
            "flag": raw["OBS_FLAG"],
        }
    )[observed]
    updated = pd.to_datetime(raw["LAST UPDATE"], format="%d/%m/%y %H:%M:%S").max()
    return Release(GrossProduction.validate(production.reset_index(drop=True)), updated)


def _get(client: httpx.Client, url: str, params: dict[str, str]) -> bytes:
    """Retries timeouts, connection errors and server errors; other HTTP errors raise at once."""
    for delay in RETRY_DELAYS_SECONDS:
        try:
            response = client.get(url, params=params)
            if response.status_code < 500:
                break
        except httpx.TransportError:
            pass
        time.sleep(delay)
    else:
        response = client.get(url, params=params)
    response.raise_for_status()
    return response.content
