"""éCO2mix production and consumption from ODRÉ, the open data portal of the French networks.

Values are in MW, as published, indexed by their UTC timestamp. Consolidated and definitive values
are 30-minute averages labelled at the end of the period: on a 15-minute grid where every other row
is empty in the national dataset, on a 30-minute grid in the regional one. Real-time values are
15-minute instantaneous values. On spring clock-change days the source repeats UTC stamps; on
autumn days it has no rows for the repeated local hour.

`national_hourly` and `regional_hourly` put these series on the common grid of `grid`, with the
publication status of each hour.
"""

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Literal

import httpx
import numpy as np
import pandas as pd
import pandera.pandas as pa
from pandera.typing import DataFrame, Index, Series

from eu_solar_wind_audit import grid

BASE_URL = "https://odre.opendatasoft.com/api/explore/v2.1/catalog/datasets"
CACHE_DIR = Path("data/raw/odre")

NATIONAL_CONSOLIDATED = "eco2mix-national-cons-def"
NATIONAL_REALTIME = "eco2mix-national-tr"
REGIONAL_CONSOLIDATED = "eco2mix-regional-cons-def"

STATUS = {
    "Données définitives": "definitive",
    "Données consolidées": "consolidated",
    "Données temps réel": "realtime",
}

# From the most to the least final.
STATUS_ORDER = list(STATUS.values())

# The 12 éCO2mix regions, as named in the regional dataset. Corsica is in neither the regional nor
# the national data: hourly national solar equals the sum of these regions within 3 MWh (checked
# in tests/test_odre_network.py).
REGIONS = {
    "11": "Île-de-France",
    "24": "Centre-Val de Loire",
    "27": "Bourgogne-Franche-Comté",
    "28": "Normandie",
    "32": "Hauts-de-France",
    "44": "Grand Est",
    "52": "Pays de la Loire",
    "53": "Bretagne",
    "75": "Nouvelle-Aquitaine",
    "76": "Occitanie",
    "84": "Auvergne-Rhône-Alpes",
    "93": "Provence-Alpes-Côte d'Azur",
}

# The regional onshore and offshore fields are null before 2021, then zero while regional wind is
# not, until 31 December 2023 (ODRÉ export read on 2026-11-22). The national consolidated dataset
# has no split.
WIND_SPLIT_START = pd.Timestamp("2024-01-01", tz="Europe/Paris").tz_convert("UTC")

NationalSeries = Literal["load", "solar", "wind", "wind_onshore", "wind_offshore"]
RegionalSeries = Literal["solar", "wind", "wind_onshore", "wind_offshore"]

# ODRÉ field -> column name, in export order.
NATIONAL_CONSOLIDATED_FIELDS = {
    "date_heure": "time",
    "nature": "status",
    "consommation": "load_mw",
    "solaire": "solar_mw",
    "eolien": "wind_mw",
}
NATIONAL_REALTIME_FIELDS = {
    **NATIONAL_CONSOLIDATED_FIELDS,
    "eolien_terrestre": "wind_onshore_mw",
    "eolien_offshore": "wind_offshore_mw",
}
REGIONAL_CONSOLIDATED_FIELDS = {
    "date_heure": "time",
    "code_insee_region": "region_code",
    "libelle_region": "region",
    "nature": "status",
    "consommation": "load_mw",
    "solaire": "solar_mw",
    "eolien": "wind_mw",
    "eolien_terrestre": "wind_onshore_mw",
    "eolien_offshore": "wind_offshore_mw",
}


# Production is not checked for sign: the source publishes small negative values, which the
# anomaly detectors must see.
class NationalConsolidated(pa.DataFrameModel):
    time: Index[grid.UtcTime]
    status: Series[str] = pa.Field(isin=["definitive", "consolidated"])
    load_mw: Series[float] = pa.Field(ge=0, nullable=True)
    solar_mw: Series[float] = pa.Field(nullable=True)
    wind_mw: Series[float] = pa.Field(nullable=True)

    class Config:
        strict = True


class NationalRealtime(pa.DataFrameModel):
    time: Index[grid.UtcTime]
    status: Series[str] = pa.Field(isin=["realtime"])
    load_mw: Series[float] = pa.Field(ge=0, nullable=True)
    solar_mw: Series[float] = pa.Field(nullable=True)
    wind_mw: Series[float] = pa.Field(nullable=True)
    wind_onshore_mw: Series[float] = pa.Field(nullable=True)
    wind_offshore_mw: Series[float] = pa.Field(nullable=True)

    class Config:
        strict = True


class RegionalConsolidated(pa.DataFrameModel):
    time: Index[grid.UtcTime]
    region_code: Series[str] = pa.Field(isin=list(REGIONS))
    region: Series[str]
    status: Series[str] = pa.Field(isin=["definitive", "consolidated"])
    load_mw: Series[float] = pa.Field(ge=0, nullable=True)
    solar_mw: Series[float] = pa.Field(nullable=True)
    wind_mw: Series[float] = pa.Field(nullable=True)
    # Published from 2021 on.
    wind_onshore_mw: Series[float] = pa.Field(nullable=True)
    wind_offshore_mw: Series[float] = pa.Field(nullable=True)

    class Config:
        strict = True


class Hourly(grid.Hourly):
    """The status is the least final one among the published rows of the hour, null when the hour
    has none."""

    status: Series[str] = pa.Field(isin=STATUS_ORDER, nullable=True)


class RegionalHourly(pa.DataFrameModel):
    time: Index[grid.UtcTime]
    region_code: Series[str] = pa.Field(isin=list(REGIONS))
    region: Series[str]
    energy_mwh: Series[float] = pa.Field(nullable=True)
    complete: Series[bool]
    status: Series[str] = pa.Field(isin=STATUS_ORDER, nullable=True)

    class Config:
        strict = True


@dataclass(frozen=True)
class Snapshot:
    path: Path
    retrieved_at: datetime
    sha256: str


def make_client() -> httpx.Client:
    # Transport retries cover failed connections only; HTTP error statuses are raised.
    return httpx.Client(
        transport=httpx.HTTPTransport(retries=3),
        timeout=httpx.Timeout(30, read=300),
        follow_redirects=True,
    )


def export_url(dataset: str, export_format: str) -> str:
    return f"{BASE_URL}/{dataset}/exports/{export_format}"


def utc_days(start: date, end: date) -> str:
    """ODSQL filter on the timestamps from `start` 00:00 UTC to `end` + 1 day 00:00 UTC, both
    included, so that every UTC day can be rebuilt from end-labelled or instantaneous values."""
    if end < start:
        raise ValueError(f"end {end} is before start {start}")
    first = datetime.combine(start, time(), UTC).isoformat()
    last = datetime.combine(end + timedelta(days=1), time(), UTC).isoformat()
    return f"date_heure >= date'{first}' AND date_heure <= date'{last}'"


def download_export(
    client: httpx.Client,
    dataset: str,
    *,
    select: str | None = None,
    where: str | None = None,
    export_format: str = "parquet",
    cache_dir: Path = CACHE_DIR,
    refresh: bool = False,
) -> Path:
    """Download an export once and return its path under `cache_dir`/`dataset`.

    The file name is a hash of the query, so a different selection is downloaded again.
    """
    params = {name: value for name, value in [("select", select), ("where", where)] if value}
    key = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:16]
    dest = cache_dir / dataset / f"{key}.{export_format}"
    if refresh or not dest.exists():
        _stream_to(client, export_url(dataset, export_format), params, dest)
    return dest


def fetch_national_consolidated(
    client: httpx.Client, start: date, end: date, *, cache_dir: Path = CACHE_DIR
) -> DataFrame[NationalConsolidated]:
    return _consolidated_export(
        client,
        NATIONAL_CONSOLIDATED,
        NATIONAL_CONSOLIDATED_FIELDS,
        NationalConsolidated,
        start,
        end,
        cache_dir,
    )


def fetch_national_realtime(
    client: httpx.Client, start: date, end: date, *, cache_dir: Path = CACHE_DIR
) -> DataFrame[NationalRealtime]:
    # Always downloaded again: real-time values are revised until consolidation.
    path = download_export(
        client,
        NATIONAL_REALTIME,
        select=",".join(NATIONAL_REALTIME_FIELDS),
        where=utc_days(start, end),
        cache_dir=cache_dir,
        refresh=True,
    )
    return NationalRealtime.validate(_tidy(path, NATIONAL_REALTIME_FIELDS))


def fetch_regional_consolidated(
    client: httpx.Client, start: date, end: date, *, cache_dir: Path = CACHE_DIR
) -> DataFrame[RegionalConsolidated]:
    return _consolidated_export(
        client,
        REGIONAL_CONSOLIDATED,
        REGIONAL_CONSOLIDATED_FIELDS,
        RegionalConsolidated,
        start,
        end,
        cache_dir,
    )


def national_hourly(
    client: httpx.Client,
    series: NationalSeries,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    cache_dir: Path = CACHE_DIR,
) -> DataFrame[Hourly]:
    """Every hour of [start, end), both hour starts in UTC. Consolidated and definitive values are
    used up to the last hour they cover, real-time values after it.

    Before real time, national onshore and offshore wind are the sums of the 12 regions.
    """
    first, last = _days(start, end)
    if series in ("wind_onshore", "wind_offshore"):
        regional = regional_hourly(client, series, start, end, cache_dir=cache_dir)
        hourly = _sum_of_regions(regional)
    else:
        published = fetch_national_consolidated(client, first, last, cache_dir=cache_dir)
        hourly = _consolidated_hourly(published[f"{series}_mw"], published["status"], start, end)

    covered = hourly.index[hourly["status"].notna()]
    if covered.empty or covered[-1] < end - grid.HOUR:
        after = start if covered.empty else covered[-1] + grid.HOUR
        realtime = fetch_national_realtime(client, after.date(), last, cache_dir=cache_dir)
        later = _realtime_hourly(realtime[f"{series}_mw"], after, end)
        hourly = pd.concat([hourly[hourly.index < after], later])
    return Hourly.validate(hourly)


def regional_hourly(
    client: httpx.Client,
    series: RegionalSeries,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    cache_dir: Path = CACHE_DIR,
) -> DataFrame[RegionalHourly]:
    """Every hour of [start, end), both hour starts in UTC, for each of the 12 regions, from
    consolidated and definitive values only."""
    first, last = _days(start, end)
    published = fetch_regional_consolidated(client, first, last, cache_dir=cache_dir)
    frames = []
    for code, name in REGIONS.items():
        rows = published[published["region_code"] == code]
        hourly = _consolidated_hourly(rows[f"{series}_mw"], rows["status"], start, end)
        if series in ("wind_onshore", "wind_offshore"):
            unpublished = hourly.index < WIND_SPLIT_START
            hourly.loc[unpublished, "energy_mwh"] = np.nan
            hourly.loc[unpublished, "complete"] = False
        frames.append(hourly.assign(region_code=code, region=name))
    return RegionalHourly.validate(pd.concat(frames))


def snapshot_realtime(client: httpx.Client, directory: Path) -> Snapshot:
    """Save the full current real-time export, with a JSON file holding its retrieval time and
    SHA-256 checksum."""
    retrieved_at = datetime.now(UTC).replace(microsecond=0)
    path = directory / f"{NATIONAL_REALTIME}_{retrieved_at:%Y%m%dT%H%M%SZ}.parquet"
    url = export_url(NATIONAL_REALTIME, "parquet")
    _stream_to(client, url, {}, path)
    with path.open("rb") as file:
        sha256 = hashlib.file_digest(file, "sha256").hexdigest()
    metadata = {
        "dataset": NATIONAL_REALTIME,
        "url": url,
        "retrieved_at": retrieved_at.isoformat(),
        "sha256": sha256,
    }
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    return Snapshot(path, retrieved_at, sha256)


def _stream_to(client: httpx.Client, url: str, params: dict[str, str], dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Write under a temporary name so that an interrupted download never looks complete.
    partial = dest.with_name(dest.name + ".part")
    with client.stream("GET", url, params=params) as response:
        response.raise_for_status()
        with partial.open("wb") as file:
            for chunk in response.iter_bytes():
                file.write(chunk)
    partial.replace(dest)


def _consolidated_export[Schema: pa.DataFrameModel](
    client: httpx.Client,
    dataset: str,
    fields: dict[str, str],
    schema: type[Schema],
    start: date,
    end: date,
    cache_dir: Path,
) -> DataFrame[Schema]:
    """Cached export, downloaded again when it stops before the last hour of `end`: consolidated
    months are added in the middle of the next month (ODRÉ dataset description) and then leave
    the real-time dataset."""
    select, where = ",".join(fields), utc_days(start, end)
    path = download_export(client, dataset, select=select, where=where, cache_dir=cache_dir)
    frame = schema.validate(_tidy(path, fields))
    if frame.empty or frame.index[-1] < pd.Timestamp(end, tz="UTC") + pd.Timedelta(hours=23):
        path = download_export(
            client, dataset, select=select, where=where, cache_dir=cache_dir, refresh=True
        )
        frame = schema.validate(_tidy(path, fields))
    return frame


def _days(start: pd.Timestamp, end: pd.Timestamp) -> tuple[date, date]:
    """First and last UTC days of the hours [start, end), refusing an empty period before any
    download."""
    if end <= start:
        raise ValueError(f"empty period from {start} to {end}")
    return start.date(), (end - grid.HOUR).date()


def _consolidated_hourly(
    power_mw: pd.Series, status: pd.Series, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    hourly = grid.hourly_from_period_end_means(power_mw)
    # Same shift as the values: a row labelled 00:00 belongs to the hour starting at 23:00.
    rank = status.map(STATUS_ORDER.index).shift(-1, freq="30min").resample("h").max()
    hourly["status"] = rank.reindex(hourly.index).map(dict(enumerate(STATUS_ORDER)))
    return grid.on_hours(hourly, start, end)


def _realtime_hourly(power_mw: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    hourly = grid.hourly_from_instantaneous(power_mw).assign(status="realtime")
    return grid.on_hours(hourly, start, end)


def _sum_of_regions(regional: pd.DataFrame) -> pd.DataFrame:
    by_hour = regional.groupby(level="time")
    complete = by_hour["complete"].all()
    rank = regional["status"].map(STATUS_ORDER.index, na_action="ignore")
    return pd.DataFrame(
        {
            "energy_mwh": by_hour["energy_mwh"].sum().where(complete),
            "complete": complete,
            "status": rank.groupby(level="time").max().map(dict(enumerate(STATUS_ORDER))),
        }
    )


def _tidy(path: Path, fields: dict[str, str]) -> pd.DataFrame:
    frame = pd.read_parquet(path, columns=list(fields)).rename(columns=fields)
    frame["status"] = frame["status"].replace(STATUS)
    values = [column for column in frame.columns if column.endswith("_mw")]
    # The regional wind field is text, with "-" or "ND" for some missing values.
    frame[values] = (
        frame[values].replace(["-", "ND"], np.nan).apply(pd.to_numeric).astype("float64")
    )
    index = pd.DatetimeIndex(frame.pop("time"), name="time").as_unit("ns")
    return frame.set_index(index).sort_index(kind="stable")
