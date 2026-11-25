"""Client for Energi Data Service, the open data portal of Energinet, the Danish transmission
system operator (licence CC BY 4.0, attribution "Source: Energinet (www.energidataservice.dk)").

ProductionConsumptionSettlement holds hourly settlement data from Datahub for the price areas DK1
and DK2: solar output by plant size, with the estimated self-consumption of solar output apart,
onshore and offshore wind output, and gross consumption. Its metadata says of
SolarPowerSelfConMWh: "The estimated production is based on real-time measurements, occasionally
containing small errors that are not corrected retroactively", and of GrossConsumptionMWh: "Total
consumption through the grid is GrossConsumptionMWh - SolarPowerSelfConMWh -
LocalPowerSelfConMWh". CapacityPerMunicipality holds the capacity of each municipality in MW,
"calculated at midnight on the first day of the month".

The size classes leave out the self-consumed output: with SolarPowerSelfConMWh they add up to the
gross solar production that Eurostat gives for Denmark in 2020 and 2022-2024. In 2019 they are
zero and the whole output is booked as self-consumption; self-consumption then takes 42 % of the
output of 2020, and from 9 to 17 % of each year's output from 2021 to 2025.

The API selects periods in local Danish time and stamps hours in UTC. Its rate limits are set per
dataset, and it answers HTTP 429 with the number of seconds to wait (API guide,
https://www.energidataservice.dk/guides/api-guides). Each request therefore takes a whole year,
and settled years are cached.
"""

import json
import re
import time
from pathlib import Path
from typing import Any, ClassVar, Literal

import httpx
import numpy as np
import pandas as pd
import pandera.pandas as pa
from pandera.typing import DataFrame, Series

from eu_solar_wind_audit import grid

BASE_URL = "https://api.energidataservice.dk/dataset"
CACHE_DIR = Path("data/raw/energinet")
TIMEZONE = "Europe/Copenhagen"
TIMEOUT_SECONDS = 120.0
ATTEMPTS = 3
RETRY_DELAY_SECONDS = 10.0
# Wait after HTTP 429 when the message gives no number of seconds.
RATE_LIMIT_SECONDS = 300.0
# Settlement data are corrected after delivery, so recent years are downloaded again on each call.
SETTLED_AFTER = pd.Timedelta(days=90)
# Years still corrected are kept for the process: the API allows about one request per dataset
# every five minutes, and a run reads each year for several quantities.
_unsettled: dict[tuple[Path, int], bytes] = {}

SETTLEMENT = "ProductionConsumptionSettlement"
# First year whose solar output is split between the size classes and self-consumption: "Before
# 2020 all solar power is placed in SolarPowerSelfConMWh" (comment of the dataset).
SOLAR_SPLIT_YEAR = 2020
CAPACITY = "CapacityPerMunicipality"

Area = Literal["DK1", "DK2", "DK"]
PriceArea = Literal["DK1", "DK2"]
PRICE_AREAS: tuple[PriceArea, ...] = ("DK1", "DK2")
Quantity = Literal[
    "solar",
    "solar_self_consumption",
    "solar_total",
    "wind_onshore",
    "wind_offshore",
    "gross_consumption",
]
SIZE_CLASSES = ("SolarPowerLt10kW_MWh", "SolarPowerGe10Lt40kW_MWh", "SolarPowerGe40kW_MWh")
# Settlement columns whose sum gives each quantity. The total adds the self-consumed output to the
# size classes, which leave it out.
QUANTITIES: dict[Quantity, tuple[str, ...]] = {
    "solar": SIZE_CLASSES,
    "solar_self_consumption": ("SolarPowerSelfConMWh",),
    "solar_total": (*SIZE_CLASSES, "SolarPowerSelfConMWh"),
    "wind_onshore": ("OnshoreWindLt50kW_MWh", "OnshoreWindGe50kW_MWh"),
    "wind_offshore": ("OffshoreWindLt100MW_MWh", "OffshoreWindGe100MW_MWh"),
    "gross_consumption": ("GrossConsumptionMWh",),
}
ENERGY_COLUMNS = tuple(
    dict.fromkeys(column for columns in QUANTITIES.values() for column in columns)
)
CAPACITY_COLUMNS = {
    "SolarPowerCapacity": "solar_mw",
    "OnshoreWindCapacity": "wind_onshore_mw",
    "OffshoreWindCapacity": "wind_offshore_mw",
}


class MunicipalityCapacity(pa.DataFrameModel):
    """Capacity in MW of each municipality at midnight, local time, on `month_start`. The price
    area follows from the municipality number: 101 to 400 in DK2, from 410 in DK1 (metadata of
    CapacityPerMunicipality); other numbers have none."""

    month_start: Series[pa.DateTime]
    municipality: Series[int]
    price_area: Series[str] = pa.Field(isin=list(PRICE_AREAS), nullable=True)
    solar_mw: Series[float] = pa.Field(ge=0, nullable=True)
    wind_onshore_mw: Series[float] = pa.Field(ge=0, nullable=True)
    wind_offshore_mw: Series[float] = pa.Field(ge=0, nullable=True)

    @pa.check("month_start")
    @classmethod
    def first_day_of_month(cls, month_start: "pd.Series[pd.Timestamp]") -> "pd.Series[bool]":
        return month_start.dt.is_month_start & (month_start == month_start.dt.normalize())

    class Config:
        strict = True
        unique: ClassVar[list[str]] = ["month_start", "municipality"]


def fetch_hourly_settlement(
    area: Area,
    quantity: Quantity,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    client: httpx.Client | None = None,
    cache_dir: Path = CACHE_DIR,
) -> DataFrame[grid.Hourly]:
    """Energy of `quantity` in `area` on the UTC hours of [start, end); DK is the sum of both
    price areas. An hour is complete when every column of the quantity is published, in both
    areas for DK. Negative values are kept."""
    grid.check_period(start, end)
    if client is None:
        with httpx.Client(timeout=TIMEOUT_SECONDS) as own_client:
            return fetch_hourly_settlement(
                area, quantity, start, end, client=own_client, cache_dir=cache_dir
            )
    records = _records(client, start, end, cache_dir)
    columns = list(QUANTITIES[quantity])
    by_area = (
        records.set_index(["time", "PriceArea"])[columns]
        .sum(axis="columns", min_count=len(columns))
        .unstack("PriceArea")
    )
    areas = list(PRICE_AREAS) if area == "DK" else [area]
    energy = by_area.reindex(columns=areas).sum(axis="columns", min_count=len(areas))
    hourly = pd.DataFrame({"energy_mwh": energy, "complete": energy.notna()})
    return grid.Hourly.validate(grid.on_hours(hourly, start, end))


def fetch_settlement_values(
    area: PriceArea,
    quantity: Quantity,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    client: httpx.Client | None = None,
    cache_dir: Path = CACHE_DIR,
) -> "pd.Series[float]":
    """Values of `quantity` in the price area `area` as published: one per record of the UTC
    hours of [start, end), indexed by its hour in time order, repeated hours kept, and NaN where a
    column of the quantity is missing."""
    grid.check_period(start, end)
    if client is None:
        with httpx.Client(timeout=TIMEOUT_SECONDS) as own_client:
            return fetch_settlement_values(
                area, quantity, start, end, client=own_client, cache_dir=cache_dir
            )
    records = _records(client, start, end, cache_dir)
    columns = list(QUANTITIES[quantity])
    rows = records[records["PriceArea"].eq(area)].set_index("time")[columns]
    values: pd.Series[float] = rows.sum(axis="columns", min_count=len(columns))
    in_period = (values.index >= start) & (values.index < end)
    return values[in_period].sort_index(kind="stable").rename("energy_mwh")


def hourly_capacity(
    capacity: pd.DataFrame,
    area: PriceArea,
    technology: Literal["solar", "wind_onshore"],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> "pd.Series[float]":
    """Capacity in MW of `technology` in the municipalities of the price area `area` at the start
    of each hour of [start, end), both hour starts in UTC, from a `MunicipalityCapacity` frame:
    linear in time between the first days of months at midnight, Danish time. Raises ValueError for
    hours outside the panel."""
    grid.check_period(start, end)
    column = f"{technology}_mw"
    monthly = capacity[capacity["price_area"].eq(area)].groupby("month_start")[column].sum()
    instants = pd.DatetimeIndex(monthly.index).tz_localize(TIMEZONE).tz_convert("UTC")
    hours = pd.date_range(start, end, freq="h", inclusive="left", unit="ns", name="time")
    capacity_mw = np.interp(
        hours.asi8, instants.as_unit("ns").asi8, monthly.to_numpy(), left=np.nan, right=np.nan
    )
    if np.isnan(capacity_mw).any():
        raise ValueError(f"no {technology} capacity of {area} for some hours from {start} to {end}")
    return pd.Series(capacity_mw, index=hours, name="capacity_mw")


def fetch_capacity(
    *,
    client: httpx.Client | None = None,
    cache_dir: Path = CACHE_DIR,
    refresh: bool = False,
) -> DataFrame[MunicipalityCapacity]:
    """Monthly capacity of every municipality, from the cached response unless `refresh` is
    set."""
    path = cache_dir / f"{CAPACITY}.json"
    if refresh or not path.exists():
        if client is None:
            with httpx.Client(timeout=TIMEOUT_SECONDS) as own_client:
                return fetch_capacity(client=own_client, cache_dir=cache_dir, refresh=refresh)
        _write(path, _get(client, f"{BASE_URL}/{CAPACITY}", {"limit": "0"}))
    return parse_capacity(json.loads(path.read_bytes()))


def parse_capacity(payload: dict[str, Any]) -> DataFrame[MunicipalityCapacity]:
    """`MunicipalityCapacity` of a CapacityPerMunicipality response."""
    records = pd.DataFrame(payload["records"])
    municipality = records["MunicipalityNo"].astype("int64")
    frame = pd.DataFrame(
        {
            "month_start": pd.to_datetime(records["Month"]).dt.as_unit("ns"),
            "municipality": municipality,
            "price_area": price_area(municipality),
        }
        | {name: records[column].astype(float) for column, name in CAPACITY_COLUMNS.items()}
    )
    ordered = frame.sort_values(["month_start", "municipality"]).reset_index(drop=True)
    return MunicipalityCapacity.validate(ordered)


def price_area(municipality: pd.Series) -> "pd.Series[str]":
    """Price area of each municipality number: DK2 from 101 to 400 and DK1 from 410 (metadata of
    CapacityPerMunicipality), null for other or unknown numbers."""
    number = pd.to_numeric(municipality).astype(float)
    areas = pd.Series(None, index=municipality.index, dtype="str")
    areas[number.between(101, 400)] = "DK2"
    areas[number >= 410] = "DK1"
    return areas


def _records(
    client: httpx.Client, start: pd.Timestamp, end: pd.Timestamp, cache_dir: Path
) -> pd.DataFrame:
    """Settlement records of the local Danish years that the UTC hours [start, end) reach."""
    first = start.tz_convert(TIMEZONE).year
    last = (end - grid.HOUR).tz_convert(TIMEZONE).year
    return pd.concat(
        [_year(client, year, cache_dir) for year in range(first, last + 1)], ignore_index=True
    )


def _year(client: httpx.Client, year: int, cache_dir: Path) -> pd.DataFrame:
    """Settlement records of a local Danish year, with their UTC hour as `time`."""
    path = cache_dir / f"{SETTLEMENT}_{year}.json"
    if path.exists():
        content = path.read_bytes()
    elif (cache_dir, year) in _unsettled:
        content = _unsettled[cache_dir, year]
    else:
        params = {"start": f"{year}-01-01", "end": f"{year + 1}-01-01", "limit": "0"}
        content = _get(client, f"{BASE_URL}/{SETTLEMENT}", params)
        if pd.Timestamp(year + 1, 1, 1, tz=TIMEZONE) + SETTLED_AFTER <= pd.Timestamp.now(tz="UTC"):
            _write(path, content)
        else:
            _unsettled[cache_dir, year] = content
    records = pd.DataFrame(
        json.loads(content)["records"], columns=["HourUTC", "PriceArea", *ENERGY_COLUMNS]
    )
    hours = pd.to_datetime(records["HourUTC"], utc=True).dt.as_unit("ns")
    return (
        records.drop(columns="HourUTC")
        .astype(dict.fromkeys(ENERGY_COLUMNS, float))
        .assign(time=hours)
    )


def _get(client: httpx.Client, url: str, params: dict[str, str]) -> bytes:
    """Waits the time the API asks for after a rate limit, and retries timeouts, connection
    errors and server errors; other HTTP errors raise at once."""
    for _ in range(ATTEMPTS):
        try:
            response = client.get(url, params=params)
        except httpx.TransportError:
            time.sleep(RETRY_DELAY_SECONDS)
            continue
        if response.status_code == 429:
            time.sleep(_rate_limit_wait(response))
        elif response.status_code >= 500:
            time.sleep(RETRY_DELAY_SECONDS)
        else:
            break
    else:
        response = client.get(url, params=params)
    response.raise_for_status()
    return response.content


def _rate_limit_wait(response: httpx.Response) -> float:
    """Seconds to wait after HTTP 429, read from "Try again in X seconds"."""
    found = re.search(r"Try again in (\d+) seconds", response.text)
    return float(found.group(1)) if found else RATE_LIMIT_SECONDS


def _write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write under a temporary name so that an interrupted write never looks complete.
    partial = path.with_name(path.name + ".part")
    partial.write_bytes(content)
    partial.replace(path)
