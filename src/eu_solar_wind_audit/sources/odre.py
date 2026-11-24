"""éCO2mix production and consumption from ODRÉ, the open data portal of the French networks.

Values are in MW, as published, indexed by their UTC timestamp. Consolidated and definitive values
are 30-minute averages labelled at the end of the period: on a 15-minute grid where every other row
is empty in the national dataset, on a 30-minute grid in the regional one. Real-time values are
15-minute instantaneous values. On spring clock-change days the source repeats UTC stamps; on
autumn days it has no rows for the repeated local hour.

`national_hourly` and `regional_hourly` put these series on the common grid of `grid`, with the
publication status of each hour; `consolidated_hourly`, `realtime_hourly` and `regional_from_rows`
do the same with rows already fetched. `onshore_hourly` gives onshore wind on the hours where the
published series tell it apart from offshore wind.
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
# National register of production and storage installations at the end of 2024 (RTE, Enedis, EDF
# SEI and the local distribution operators), installations of 36 kW or less aggregated.
REGISTER = "registre-national-installation-production-stockage-electricite-agrege-311224"
REGISTER_FIELDS = {
    "coderegion": "region_code",
    "filiere": "sector",
    "codegestionnaire": "grid_operator",
    "puismaxinstallee": "capacity_kw",
}
# The EIC code of RTE, the transmission system operator, in the register.
RTE_EIC = "10XFR-RTE------Q"
# INSEE codes of the regions that the national éCO2mix series leave out: the five overseas regions
# and Corsica.
OUTSIDE_NATIONAL = ("01", "02", "03", "04", "06", "94")

# Layout of the rows. RTE's "Spécification des fichiers de données en puissance pour éCO2mix"
# (2025, linked from https://www.rte-france.com/en/data-publications/eco2mix/download-indicators)
# gives consolidated and definitive values as mean power every 30 minutes and real-time values as
# instantaneous power every 15 minutes (sections 1.1 and 3). It does not say where a mean is
# stamped: the end of its period is the convention of docs/evaluation-plan.md.
CONSOLIDATED_STEP = pd.Timedelta(minutes=30)
CONSOLIDATED_LABEL: grid.Label = "end"
REALTIME_STEP = pd.Timedelta(minutes=15)
REALTIME_LABEL: grid.Label = "instant"
# The specification gives 96 rows for every day in Paris time (section 3.1): in UTC, a
# clock-change day repeats or lacks the stamps of one hour.
TIMEZONE = "Europe/Paris"

STATUS = {
    "Données définitives": "definitive",
    "Données consolidées": "consolidated",
    "Données temps réel": "realtime",
}

# From the most to the least final.
STATUS_ORDER = list(STATUS.values())

# The 12 éCO2mix regions, as named in the regional dataset. Corsica is in neither the regional nor
# the national data: hourly national solar and wind equal the sum of these regions to the rounding
# of published values (checked in tests/test_odre_network.py).
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

# Under a day, so that a daily run always downloads again an export that stopped early the day
# before.
SHORT_EXPORT_MAX_AGE = timedelta(hours=12)

# The regional onshore and offshore fields are null before 2021, then zero while regional wind is
# not, until 31 December 2023 (ODRÉ export read on 2026-11-22). The national consolidated dataset
# has no split.
WIND_SPLIT_START = pd.Timestamp("2024-01-01", tz="Europe/Paris").tz_convert("UTC")

# Saint-Nazaire, the first offshore wind farm in France, delivered its first power in early June
# 2022 (EDF; AFP, 2022-06-10). Earlier offshore output, from the 2 MW Floatgen demonstrator
# connected in 2018, is ignored.
FIRST_OFFSHORE_POWER = pd.Timestamp("2022-06-01", tz="Europe/Paris").tz_convert("UTC")

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
        hourly = sum_of_regions(regional)
    else:
        published = fetch_national_consolidated(client, first, last, cache_dir=cache_dir)
        hourly = consolidated_hourly(published[f"{series}_mw"], published["status"], start, end)

    after = consolidated_end(hourly, start)
    if after < end:
        realtime = fetch_national_realtime(client, after.date(), last, cache_dir=cache_dir)
        later = realtime_hourly(realtime[f"{series}_mw"], after, end)
        hourly = pd.concat([hourly[hourly.index < after], later])
    return Hourly.validate(hourly)


def regional_hourly(
    client: httpx.Client,
    series: RegionalSeries,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    cache_dir: Path = CACHE_DIR,
) -> dict[str, DataFrame[Hourly]]:
    """Every hour of [start, end), both hour starts in UTC, for each of the 12 regions, from
    consolidated and definitive values only. Keyed by region code, in `REGIONS` order."""
    first, last = _days(start, end)
    published = fetch_regional_consolidated(client, first, last, cache_dir=cache_dir)
    return regional_from_rows(published, series, start, end)


def onshore_hourly(
    client: httpx.Client,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    cache_dir: Path = CACHE_DIR,
) -> DataFrame[Hourly]:
    """Onshore wind for every hour of [start, end), both hour starts in UTC: national wind before
    `FIRST_OFFSHORE_POWER`, national onshore wind from `WIND_SPLIT_START`. Hours in between are
    incomplete and have no status: published wind then includes offshore output, which no series
    gives separately."""
    grid.check_period(start, end)
    parts = []
    if start < FIRST_OFFSHORE_POWER:
        wind_end = min(end, FIRST_OFFSHORE_POWER)
        parts.append(national_hourly(client, "wind", start, wind_end, cache_dir=cache_dir))
    if end > WIND_SPLIT_START:
        onshore_start = max(start, WIND_SPLIT_START)
        parts.append(
            national_hourly(client, "wind_onshore", onshore_start, end, cache_dir=cache_dir)
        )
    hourly = pd.concat(parts) if parts else Hourly.empty()
    return Hourly.validate(grid.on_hours(hourly, start, end))


def regional_from_rows(
    published: pd.DataFrame, series: RegionalSeries, start: pd.Timestamp, end: pd.Timestamp
) -> dict[str, DataFrame[Hourly]]:
    """`regional_hourly` from `published`, rows of `fetch_regional_consolidated`."""
    by_region: dict[str, DataFrame[Hourly]] = {}
    for code in REGIONS:
        rows = published[published["region_code"] == code]
        hourly = consolidated_hourly(rows[f"{series}_mw"], rows["status"], start, end)
        if series in ("wind_onshore", "wind_offshore"):
            unpublished = hourly.index < WIND_SPLIT_START
            hourly.loc[unpublished, "energy_mwh"] = np.nan
            hourly.loc[unpublished, "complete"] = False
        by_region[code] = Hourly.validate(hourly)
    return by_region


def consolidated_hourly(
    power_mw: pd.Series, status: pd.Series, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    """Every hour of [start, end), both hour starts in UTC, of the consolidated and definitive
    values `power_mw` of one series, with the status of each hour from the `status` of the rows,
    both as fetched. The result is not validated."""
    hourly = grid.hourly_from_period_end_means(power_mw)
    # Same shift as the values: a row labelled 00:00 belongs to the hour starting at 23:00.
    rank = status.map(STATUS_ORDER.index).shift(-1, freq="30min").resample("h").max()
    hourly["status"] = rank.reindex(hourly.index).map(dict(enumerate(STATUS_ORDER)))
    return grid.on_hours(hourly, start, end)


def realtime_hourly(power_mw: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Every hour of [start, end), both hour starts in UTC, of the real-time values `power_mw` of
    one series as fetched; the hours with rows have the status realtime. The result is not
    validated."""
    hourly = grid.hourly_from_instantaneous(power_mw)
    published = hourly.index.isin(pd.DatetimeIndex(power_mw.index).floor("h"))
    hourly["status"] = pd.Series("realtime", index=hourly.index).where(published)
    return grid.on_hours(hourly, start, end)


def sum_of_regions(regional: dict[str, DataFrame[Hourly]]) -> pd.DataFrame:
    """National hours as the sum of the regional `Hourly` frames of `regional`: complete when
    every region is, with the least final status of the regions. The result is not validated."""
    stacked = pd.concat(regional.values())
    by_hour = stacked.groupby(level="time")
    complete = by_hour["complete"].all()
    rank = stacked["status"].map(STATUS_ORDER.index, na_action="ignore")
    return pd.DataFrame(
        {
            "energy_mwh": by_hour["energy_mwh"].sum().where(complete),
            "complete": complete,
            "status": rank.groupby(level="time").max().map(dict(enumerate(STATUS_ORDER))),
        }
    )


def consolidated_end(hourly: pd.DataFrame, start: pd.Timestamp) -> pd.Timestamp:
    """The hour after the last hour with rows in `hourly`, the consolidated and definitive hours
    of a series from `start`, or `start` when no hour has rows. Real-time values fill the hours
    from it."""
    covered = hourly.index[hourly["status"].notna()]
    return start if covered.empty else covered[-1] + grid.HOUR


def complete_end(hourly: pd.DataFrame, start: pd.Timestamp) -> pd.Timestamp:
    """The hour after the last complete hour of `hourly`, the consolidated and definitive hours of
    a series from `start`, or `start` without one. Rows that stop at a date leave their last hour
    incomplete: each value is labelled at the end of its half-hour, so the last half-hour of a day
    has the date of the next one."""
    complete = hourly.index[hourly["complete"].to_numpy()]
    return start if complete.empty else complete[-1] + grid.HOUR


def fetch_register(client: httpx.Client, *, cache_dir: Path = CACHE_DIR) -> pd.DataFrame:
    """Wind ("Eolien") and solar ("Solaire") rows of the register at the end of 2024: region code,
    sector, EIC code of the grid operator and maximum installed capacity in kW."""
    path = download_export(
        client,
        REGISTER,
        select=",".join(REGISTER_FIELDS),
        where="filiere in ('Solaire','Eolien')",
        cache_dir=cache_dir,
    )
    return pd.read_parquet(path).rename(columns=REGISTER_FIELDS)


def transmission_share(register: pd.DataFrame, sector: str) -> float:
    """Share of the capacity of `sector` in mainland France without Corsica that is connected to
    the grid of RTE, from rows of `fetch_register`; rows without a region count as mainland."""
    rows = register[
        (register["sector"] == sector) & ~register["region_code"].isin(OUTSIDE_NATIONAL)
    ]
    on_rte = rows["grid_operator"] == RTE_EIC
    return float(rows.loc[on_rte, "capacity_kw"].sum() / rows["capacity_kw"].sum())


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
    """Cached export, downloaded again when it stops before the last hour of `end` and was
    retrieved more than 12 hours ago: consolidated months are added in the middle of the next
    month (ODRÉ dataset description) and then leave the real-time dataset."""
    select, where = ",".join(fields), utc_days(start, end)
    path = download_export(client, dataset, select=select, where=where, cache_dir=cache_dir)
    frame = schema.validate(_tidy(path, fields))
    short = frame.empty or frame.index[-1] < pd.Timestamp(end, tz="UTC") + pd.Timedelta(hours=23)
    age = datetime.now(UTC) - datetime.fromtimestamp(path.stat().st_mtime, UTC)
    if short and age > SHORT_EXPORT_MAX_AGE:
        path = download_export(
            client, dataset, select=select, where=where, cache_dir=cache_dir, refresh=True
        )
        frame = schema.validate(_tidy(path, fields))
    return frame


def _days(start: pd.Timestamp, end: pd.Timestamp) -> tuple[date, date]:
    """First and last UTC days of the hours [start, end), refusing a wrong period before any
    download."""
    grid.check_period(start, end)
    return start.date(), (end - grid.HOUR).date()


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
