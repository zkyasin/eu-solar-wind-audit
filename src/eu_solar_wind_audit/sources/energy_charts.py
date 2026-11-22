"""Installed capacity from the Energy-Charts API of Fraunhofer ISE (licence CC BY 4.0, attribution
"energy-charts.info", as stated in the `license` field of each response).

A month's value is the capacity installed at the end of that month. Monthly values are published
for Germany only.
"""

import json
import time
from pathlib import Path
from typing import Any, ClassVar

import httpx
import pandas as pd
import pandera.pandas as pa
from pandera.typing import DataFrame, Series

URL = "https://api.energy-charts.info/v2/installed_power"
CACHE_DIR = Path("data/raw/energy_charts")
TIMEOUT_SECONDS = 30.0
# The API allows about 2 requests per minute and answers 429 with a Retry-After in seconds.
RETRY_DELAYS_SECONDS = (10.0, 30.0, 60.0)

TECHNOLOGIES = ("solar_ac", "solar_dc", "wind_onshore", "wind_offshore")
MW_PER_UNIT = {"MW": 1.0, "GW": 1000.0}


class InstalledCapacity(pa.DataFrameModel):
    month: Series[pa.DateTime]
    technology: Series[str] = pa.Field(isin=TECHNOLOGIES)
    capacity_mw: Series[float] = pa.Field(ge=0)

    @pa.check("month")
    @classmethod
    def first_day_of_month(cls, month: "pd.Series[pd.Timestamp]") -> "pd.Series[bool]":
        return month.dt.is_month_start & (month == month.dt.normalize())

    class Config:
        strict = True
        unique: ClassVar[list[str]] = ["month", "technology"]


def fetch_installed_capacity(
    country: str,
    *,
    client: httpx.Client | None = None,
    cache_dir: Path = CACHE_DIR,
    refresh: bool = False,
) -> DataFrame[InstalledCapacity]:
    """Monthly installed capacity, from the cached response unless `refresh` is set.

    The API answers 404 for any country but "de".
    """
    path = cache_dir / f"installed_power_{country}_monthly.json"
    if path.exists() and not refresh:
        return parse_installed_power(json.loads(path.read_bytes()))

    params = {"country": country, "time_step": "monthly"}
    if client is None:
        with httpx.Client(timeout=TIMEOUT_SECONDS) as own_client:
            content = _get(own_client, params)
    else:
        content = _get(client, params)
    # Parsed before caching, so that a response that does not parse is never kept.
    capacity = parse_installed_power(json.loads(content))
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write under a temporary name so that an interrupted write never looks complete.
    partial = path.with_name(path.name + ".part")
    partial.write_bytes(content)
    partial.replace(path)
    return capacity


def parse_installed_power(payload: dict[str, Any]) -> DataFrame[InstalledCapacity]:
    """Tidy panel from a /v2/installed_power response; null values are left out.

    The month in progress at `generated_at` is dropped: its value is not yet an end-of-month
    capacity.
    """
    # The API states the licence per response and some of its data is not open.
    if not payload["license"].startswith("CC BY 4.0"):
        raise ValueError(f"unexpected licence: {payload['license']}")
    units = {series["id"]: series.get("unit") or payload["unit"] for series in payload["series"]}
    technologies = [technology for technology in TECHNOLOGIES if technology in units]
    unknown = {units[technology] for technology in technologies} - MW_PER_UNIT.keys()
    if unknown:
        raise ValueError(f"unexpected units: {sorted(unknown)}")

    # Timestamps are local midnights with a seasonal UTC offset.
    timezone = payload["timezone"]
    stamps = pd.to_datetime([row["timestamp"] for row in payload["data"]], utc=True)
    months = stamps.tz_convert(timezone).tz_localize(None).as_unit("ns")
    generated_at = pd.Timestamp(payload["generated_at"]).tz_convert(timezone).tz_localize(None)
    ended = months + pd.offsets.MonthBegin(1) <= generated_at

    values = pd.DataFrame(
        [row["values"] for row in payload["data"]],
        index=pd.Index(months, name="month"),
        columns=pd.Index(technologies, name="technology"),
        dtype=float,
    )
    capacity_mw = values[ended] * [MW_PER_UNIT[units[technology]] for technology in technologies]
    tidy = capacity_mw.reset_index().melt(id_vars="month", value_name="capacity_mw").dropna()
    return InstalledCapacity.validate(tidy.reset_index(drop=True))


def _get(client: httpx.Client, params: dict[str, str]) -> bytes:
    """Retries rate limits, timeouts, connection errors and server errors; other HTTP errors
    raise at once."""
    for delay in RETRY_DELAYS_SECONDS:
        try:
            response = client.get(URL, params=params)
        except httpx.TransportError:
            time.sleep(delay)
            continue
        if response.status_code != 429 and response.status_code < 500:
            break
        time.sleep(float(response.headers.get("Retry-After", delay)))
    else:
        response = client.get(URL, params=params)
    response.raise_for_status()
    return response.content
