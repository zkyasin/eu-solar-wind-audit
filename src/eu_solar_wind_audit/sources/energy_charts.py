"""German production and installed capacity from the Energy-Charts API of Fraunhofer ISE (licence
CC BY 4.0, attribution "energy-charts.info", as stated in the `license` field of each response).

Each capacity value is the capacity installed at the end of a month and is labelled with that
month's last day. Monthly values are published for Germany only.

Production is mean power in MW over quarter-hours labelled at their start. `public_power` is the
net production fed into the public grid; `total_power` adds industrial self-supply and, for solar,
self-consumption. `total_power` is published for Germany only. Their hourly wind energy was equal
in every hour of 2019-2025 (checked on 2026-11-22).
"""

import json
import time
from pathlib import Path
from typing import Any, ClassVar

import httpx
import pandas as pd
import pandera.pandas as pa
from pandera.typing import DataFrame, Series

from eu_solar_wind_audit import grid

URL = "https://api.energy-charts.info/v2/installed_power"
POWER_URL = "https://api.energy-charts.info/v2/{endpoint}"
CACHE_DIR = Path("data/raw/energy_charts")
TIMEOUT_SECONDS = 30.0
# The API allows about 2 requests per minute and answers 429 with a Retry-After in seconds.
RETRY_DELAYS_SECONDS = (10.0, 30.0, 60.0)

TECHNOLOGIES = ("solar_ac", "solar_dc", "wind_onshore", "wind_offshore")
MW_PER_UNIT = {"MW": 1.0, "GW": 1000.0}

PERIMETERS = {"public_power": "public", "total_power": "total"}
POWER_SERIES = ("solar", "wind_onshore", "wind_offshore")
TIMEZONE = "Europe/Berlin"
QUARTER_HOUR = pd.Timedelta(minutes=15)


class InstalledCapacity(pa.DataFrameModel):
    month_end: Series[pa.DateTime]
    technology: Series[str] = pa.Field(isin=TECHNOLOGIES)
    capacity_mw: Series[float] = pa.Field(ge=0)

    @pa.check("month_end")
    @classmethod
    def last_day_of_month(cls, month_end: "pd.Series[pd.Timestamp]") -> "pd.Series[bool]":
        return month_end.dt.is_month_end & (month_end == month_end.dt.normalize())

    class Config:
        strict = True
        unique: ClassVar[list[str]] = ["month_end", "technology"]


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
            content = _get(own_client, URL, params)
    else:
        content = _get(client, URL, params)
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
    _check_licence(payload)
    units = {series["id"]: series.get("unit") or payload["unit"] for series in payload["series"]}
    technologies = [technology for technology in TECHNOLOGIES if technology in units]
    unknown = {units[technology] for technology in technologies} - MW_PER_UNIT.keys()
    if unknown:
        raise ValueError(f"unexpected units: {sorted(unknown)}")

    # Timestamps are the local midnights starting each month, with a seasonal UTC offset.
    timezone = payload["timezone"]
    stamps = pd.to_datetime([row["timestamp"] for row in payload["data"]], utc=True)
    starts = stamps.tz_convert(timezone).tz_localize(None).as_unit("ns")
    generated_at = pd.Timestamp(payload["generated_at"]).tz_convert(timezone).tz_localize(None)
    ended = starts + pd.offsets.MonthBegin(1) <= generated_at

    values = pd.DataFrame(
        [row["values"] for row in payload["data"]],
        index=pd.Index(starts + pd.offsets.MonthEnd(0), name="month_end"),
        columns=pd.Index(technologies, name="technology"),
        dtype=float,
    )
    capacity_mw = values[ended] * [MW_PER_UNIT[units[technology]] for technology in technologies]
    tidy = capacity_mw.reset_index().melt(id_vars="month_end", value_name="capacity_mw").dropna()
    return InstalledCapacity.validate(tidy.reset_index(drop=True))


def fetch_hourly_power(
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    client: httpx.Client | None = None,
    cache_dir: Path = CACHE_DIR,
) -> dict[str, DataFrame[grid.Hourly]]:
    """German solar and wind production on the UTC hours of [start, end).

    Keys are `<public|total>.<technology>` and `self_consumption.solar` (total minus public).
    Whole local months are requested.
    """
    grid.check_period(start, end)
    if client is None:
        with httpx.Client(timeout=TIMEOUT_SECONDS) as own_client:
            return fetch_hourly_power(start, end, client=own_client, cache_dir=cache_dir)

    months = pd.period_range(_local(start), _local(end - grid.HOUR), freq="M")
    series: dict[str, DataFrame[grid.Hourly]] = {}
    for endpoint, perimeter in PERIMETERS.items():
        monthly = [fetch_power_month(endpoint, month, client, cache_dir) for month in months]
        power = pd.concat(monthly)
        for technology in POWER_SERIES:
            energy = power[technology] * (QUARTER_HOUR / grid.HOUR)
            hourly = grid.hourly_from_period_start_energy(energy)
            name = f"{perimeter}.{technology}"
            series[name] = grid.Hourly.validate(grid.on_hours(hourly, start, end))

    public, total = series["public.solar"], series["total.solar"]
    series["self_consumption.solar"] = grid.Hourly.validate(
        total.assign(
            energy_mwh=total["energy_mwh"] - public["energy_mwh"],
            complete=total["complete"] & public["complete"],
        )
    )
    return series


def fetch_power_month(
    endpoint: str, month: pd.Period, client: httpx.Client, cache_dir: Path
) -> pd.DataFrame:
    """Solar and wind power of one local month, from the cache once the month is settled and
    complete."""
    path = cache_dir / f"{endpoint}_de_{month}.json"
    if path.exists():
        return parse_power(json.loads(path.read_bytes()))

    params = {
        "country": "de",
        "start": str(month.start_time.date()),
        "end": str(month.end_time.date()),
    }
    content = _get(client, POWER_URL.format(endpoint=endpoint), params)
    # Parsed before caching, so that a response that does not parse is never kept.
    power = parse_power(json.loads(content))
    # Values of the current and previous month can still be filled in or revised. The API leaves
    # out the quarter-hours it has no values for yet, and its documentation reads null as "not
    # (yet) available", so a month is kept only when no value of any quarter-hour is missing.
    this_month = _local(pd.Timestamp.now(tz="UTC")).to_period("M")
    quarter_hours = pd.date_range(month.start_time, month.end_time, freq=QUARTER_HOUR, tz=TIMEZONE)
    if month < this_month - 1 and quarter_hours.isin(power.dropna().index).all():
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(path.name + ".part")
        partial.write_bytes(content)
        partial.replace(path)
    return power


def parse_power(payload: dict[str, Any]) -> pd.DataFrame:
    """Mean power in MW of each series of POWER_SERIES, indexed by the UTC start of each
    quarter-hour; null values become NaN."""
    _check_licence(payload)
    if payload["interval_minutes"] != QUARTER_HOUR // pd.Timedelta(minutes=1):
        raise ValueError(f"unexpected interval: {payload['interval_minutes']} minutes")
    units = {series["id"]: series.get("unit") or payload["unit"] for series in payload["series"]}
    unknown = {units.get(technology) for technology in POWER_SERIES} - {"MW"}
    if unknown:
        raise ValueError(f"unexpected units: {unknown}")

    rows = payload["data"]
    stamps = pd.to_datetime([row["timestamp"] for row in rows], utc=True)
    return pd.DataFrame(
        [row["values"] for row in rows],
        index=pd.DatetimeIndex(stamps, name="time"),
        columns=list(POWER_SERIES),
        dtype=float,
    )


def _local(stamp: pd.Timestamp) -> pd.Timestamp:
    return stamp.tz_convert(TIMEZONE).tz_localize(None)


def _check_licence(payload: dict[str, Any]) -> None:
    # The API states the licence per response and some of its data is not open.
    if not payload["license"].startswith("CC BY 4.0"):
        raise ValueError(f"unexpected licence: {payload['license']}")


def _get(client: httpx.Client, url: str, params: dict[str, str]) -> bytes:
    """Retries rate limits, timeouts, connection errors and server errors; other HTTP errors
    raise at once."""
    for delay in RETRY_DELAYS_SECONDS:
        try:
            response = client.get(url, params=params)
        except httpx.TransportError:
            time.sleep(delay)
            continue
        if response.status_code != 429 and response.status_code < 500:
            break
        time.sleep(float(response.headers.get("Retry-After", delay)))
    else:
        response = client.get(url, params=params)
    response.raise_for_status()
    return response.content
