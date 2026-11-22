"""éCO2mix production and consumption from ODRÉ, the open data portal of the French networks.

Values are in MW, as published, indexed by their UTC timestamp. Consolidated and definitive values
are 30-minute averages labelled at the end of the period: on a 15-minute grid where every other row
is empty in the national dataset, on a 30-minute grid in the regional one. Real-time values are
15-minute instantaneous values. On spring clock-change days the source repeats UTC stamps; on
autumn days it has no rows for the repeated local hour.
"""

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Annotated

import httpx
import numpy as np
import pandas as pd
import pandera.pandas as pa
from pandera.typing import DataFrame, Index, Series

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

UtcTime = Annotated[pd.DatetimeTZDtype, "ns", "UTC"]


# Production is not checked for sign: the source publishes small negative values, which the
# anomaly detectors must see.
class NationalConsolidated(pa.DataFrameModel):
    time: Index[UtcTime]
    status: Series[str] = pa.Field(isin=["definitive", "consolidated"])
    load_mw: Series[float] = pa.Field(ge=0, nullable=True)
    solar_mw: Series[float] = pa.Field(nullable=True)
    wind_mw: Series[float] = pa.Field(nullable=True)

    class Config:
        strict = True


class NationalRealtime(pa.DataFrameModel):
    time: Index[UtcTime]
    status: Series[str] = pa.Field(isin=["realtime"])
    load_mw: Series[float] = pa.Field(ge=0, nullable=True)
    solar_mw: Series[float] = pa.Field(nullable=True)
    wind_mw: Series[float] = pa.Field(nullable=True)
    wind_onshore_mw: Series[float] = pa.Field(nullable=True)
    wind_offshore_mw: Series[float] = pa.Field(nullable=True)

    class Config:
        strict = True


class RegionalConsolidated(pa.DataFrameModel):
    time: Index[UtcTime]
    region_code: Series[str]
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
    path = download_export(
        client,
        NATIONAL_CONSOLIDATED,
        select=",".join(NATIONAL_CONSOLIDATED_FIELDS),
        where=utc_days(start, end),
        cache_dir=cache_dir,
    )
    return NationalConsolidated.validate(_tidy(path, NATIONAL_CONSOLIDATED_FIELDS))


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
    path = download_export(
        client,
        REGIONAL_CONSOLIDATED,
        select=",".join(REGIONAL_CONSOLIDATED_FIELDS),
        where=utc_days(start, end),
        cache_dir=cache_dir,
    )
    return RegionalConsolidated.validate(_tidy(path, REGIONAL_CONSOLIDATED_FIELDS))


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
