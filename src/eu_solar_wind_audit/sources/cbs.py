"""Wind on land in two tables of Statistics Netherlands (CBS) on its open data portal: the capacity
at the end of each year by province (70960ned, « Windenergie op land; productie en capaciteit per
provincie ») and by hub height (71227ned, « Windenergie op land; productie en capaciteit naar
ashoogte »). Wind on land includes the turbines in inland waters. CBS licenses its tables under
CC BY 4.0, cited « Bron: CBS ».

Each table is read whole through the OData version 3 API and kept on disk as received. CBS revises
the tables once a year (both last on 2026-11-13). A year at its first, provisional release
(Voorlopig) is left out; a revised provisional year (NaderVoorlopig) is taken as final (evaluation
plan, Deviations).
"""

import json
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import pandera.pandas as pa
from pandera.typing import DataFrame, Series

URL = "https://opendata.cbs.nl/ODataApi/OData"
RAW_DIR = Path("data/raw/cbs")
TIMEOUT_SECONDS = 60.0
PROVINCES = "70960ned"
HUB_HEIGHTS = "71227ned"
# The dimension that each table crosses with the years.
DIMENSIONS = {PROVINCES: "RegioS", HUB_HEIGHTS: "AshoogtenWindmolens"}
# « Opgesteld vermogen einde jaar », in MW, in both tables.
CAPACITY = "OpgesteldVermogenEindeJaar_9"
USED_STATUSES = ("Definitief", "NaderVoorlopig")
PROVINCE_PREFIX = "PV"
PROVINCE_SUFFIX = " (PV)"
# The five hub height classes of 71227ned, merged into the three generations of the Dutch fleet.
# The total of all classes, A044968, is left out.
HUB_HEIGHT_CLASSES = {
    "A044969": "up to 70 m",  # t/m 30 meter
    "A044970": "up to 70 m",  # 31 t/m 50 meter
    "A044971": "up to 70 m",  # 51 t/m 70 meter
    "A044972": "71 to 95 m",
    "A044973": "96 m or more",
}


class YearEndCapacity(pa.DataFrameModel):
    """Onshore capacity in MW at the end of each year of `group`: a province, named as CBS names
    it, or a hub height class of HUB_HEIGHT_CLASSES."""

    year: Series[int]
    group: Series[str]
    capacity_mw: Series[float] = pa.Field(ge=0)

    class Config:
        strict = True


def download(table: str, client: httpx.Client | None = None, raw_dir: Path = RAW_DIR) -> Path:
    """Directory of `table`, holding its data set, its years and dimension and its table
    information as JSON, downloaded unless cached."""
    folder = raw_dir / table
    endpoints = ("TypedDataSet", "Perioden", DIMENSIONS[table], "TableInfos")
    if all((folder / f"{endpoint}.json").exists() for endpoint in endpoints):
        return folder
    if client is None:
        with httpx.Client(timeout=TIMEOUT_SECONDS) as own_client:
            return download(table, own_client, raw_dir)

    folder.mkdir(parents=True, exist_ok=True)
    for endpoint in endpoints:
        response = client.get(f"{URL}/{table}/{endpoint}")
        response.raise_for_status()
        partial = folder / f"{endpoint}.json.part"
        partial.write_bytes(response.content)
        partial.replace(folder / f"{endpoint}.json")
    return folder


def modified(folder: Path) -> str:
    """Date of the last revision of the table in `folder`, as CBS gives it."""
    info: dict[str, Any] = _values(folder, "TableInfos")[0]
    return str(info["Modified"])[:10]


def province_capacity(folder: Path | None = None) -> DataFrame[YearEndCapacity]:
    """Capacity of each province at each used year end, from 70960ned."""
    folder = folder or download(PROVINCES)
    names = {
        row["Key"].strip(): row["Title"].removesuffix(PROVINCE_SUFFIX)
        for row in _values(folder, DIMENSIONS[PROVINCES])
        if row["Key"].startswith(PROVINCE_PREFIX)
    }
    capacity = _capacity(folder, DIMENSIONS[PROVINCES])
    capacity = capacity[capacity["group"].isin(names)]
    return YearEndCapacity.validate(capacity.assign(group=capacity["group"].map(names)))


def hub_height_capacity(folder: Path | None = None) -> DataFrame[YearEndCapacity]:
    """National capacity of each hub height class at each used year end, from 71227ned."""
    folder = folder or download(HUB_HEIGHTS)
    capacity = _capacity(folder, DIMENSIONS[HUB_HEIGHTS])
    capacity = capacity[capacity["group"].isin(HUB_HEIGHT_CLASSES)]
    merged = (
        capacity.assign(group=capacity["group"].map(HUB_HEIGHT_CLASSES))
        .groupby(["year", "group"])["capacity_mw"]
        .sum()
        .reset_index()
    )
    return YearEndCapacity.validate(merged)


def _capacity(folder: Path, dimension: str) -> pd.DataFrame:
    """Year-end capacity of each key of `dimension` in the used years, sorted."""
    statuses = {row["Key"]: row["Status"] for row in _values(folder, "Perioden")}
    rows = pd.DataFrame(_values(folder, "TypedDataSet"))
    rows = rows[rows["Perioden"].map(statuses).isin(USED_STATUSES)]
    capacity = pd.DataFrame(
        {
            # Years are keyed as "2025JJ00".
            "year": rows["Perioden"].str[:4].astype(int),
            "group": rows[dimension].str.strip(),
            # Cells are empty only before 2005, for provinces and classes without turbines yet.
            "capacity_mw": rows[CAPACITY].astype(float).fillna(0.0),
        }
    )
    return capacity.sort_values(["year", "group"], ignore_index=True)


def _values(folder: Path, endpoint: str) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = json.loads((folder / f"{endpoint}.json").read_text())["value"]
    return values
