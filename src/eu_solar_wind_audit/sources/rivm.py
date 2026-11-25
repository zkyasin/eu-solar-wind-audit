"""Map of the wind turbines of the Netherlands of the RIVM Atlas Leefomgeving (« Windturbines -
vermogen »), snapshot of 1 January 2026: position, capacity, rotor diameter, hub height, province
and ground (land, inland water or sea) of each turbine, and of the Belgian and German turbines
near the border.

The metadata record (read 2026-11-25) marks the map with the Public Domain Mark 1.0 and names
RIVM, the BGT, Rijkswaterstaat, OpenStreetMap and the Register Externe Veiligheid in its lineage.
The map enters the model only, as municipality weights and hub height classes, and none of its
rows is published.

The map is served only through WFS, which names each feature anew on every request, so the
checksum pins the rows read rather than the file.
"""

import hashlib
from pathlib import Path

import geopandas as gpd
import httpx
import numpy as np
import pandas as pd
import pandera.pandas as pa
from pandera.typing import DataFrame, Series

URL = "https://data.rivm.nl/geo/alo/wfs"
LAYER = "alo:rivm_20260101_windturbines_vermogen"
QUERY = {
    "service": "WFS",
    "version": "2.0.0",
    "request": "GetFeature",
    "typeNames": LAYER,
    "outputFormat": "csv",
    "sortBy": "id",
}
# Computed on two downloads of 2026-11-25, over the text of COLUMNS with the rows in order of id.
ROWS_SHA256 = "94fc5479a4d7342faaa00a10138545d94a256df50d722ce5ecdc0f7d21145981"
RAW_DIR = Path("data/raw/rivm")
TIMEOUT_SECONDS = 300.0
COLUMNS = {
    "id": "turbine",
    "kw": "capacity_kw",
    "diam": "rotor_diameter_m",
    "ash": "hub_height_m",
    "x": "x",
    "y": "y",
    "land": "country",
    "prov_naam": "province",
    "ondergrond": "ground",
}
# Amersfoort / RD New.
MAP_CRS = "EPSG:28992"
NETHERLANDS = "Nederland"
ONSHORE_GROUNDS = ("land", "binnenwater")
# The three classes of cbs.HUB_HEIGHT_CLASSES, by hub height rounded to the metre.
HUB_HEIGHT_CLASSES = {"up to 70 m": 70, "71 to 95 m": 95, "96 m or more": None}


class Turbines(pa.DataFrameModel):
    """One row per turbine of the map, with its position in degrees (EPSG:4326). `province` is
    null outside the Dutch provinces, and `ground` for two turbines that the map places nowhere."""

    turbine: Series[int] = pa.Field(unique=True)
    capacity_kw: Series[float] = pa.Field(ge=0)
    rotor_diameter_m: Series[float] = pa.Field(gt=0)
    hub_height_m: Series[float] = pa.Field(gt=0)
    longitude: Series[float]
    latitude: Series[float]
    country: Series[str]
    province: Series[str] = pa.Field(nullable=True)
    ground: Series[str] = pa.Field(nullable=True)

    class Config:
        strict = True


class HubHeightClasses(pa.DataFrameModel):
    """Turbines of each hub height class: their capacity in MW, their hub height in m weighted by
    capacity, and their specific power, the capacity over the area swept by their rotors."""

    hub_height_class: Series[str] = pa.Field(isin=list(HUB_HEIGHT_CLASSES))
    capacity_mw: Series[float] = pa.Field(ge=0)
    hub_height_m: Series[float]
    specific_power_w_m2: Series[float]

    class Config:
        strict = True


def download(client: httpx.Client | None = None, raw_dir: Path = RAW_DIR) -> Path:
    """Path of the map as CSV, downloaded unless cached; read_turbines checks its rows."""
    dest = raw_dir / f"{LAYER.split(':')[1]}.csv"
    if dest.exists():
        return dest
    if client is None:
        with httpx.Client(timeout=TIMEOUT_SECONDS) as own_client:
            return download(own_client, raw_dir)
    response = client.get(URL, params=QUERY)
    response.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".part")
    partial.write_bytes(response.content)
    partial.replace(dest)
    return dest


def rows_sha256(text: pd.DataFrame) -> str:
    """SHA-256 of the text of COLUMNS in `text`, the map read as strings, rows in order of id."""
    ordered = text.iloc[text["id"].astype(int).argsort()]
    lines = ordered[list(COLUMNS)].apply(",".join, axis="columns")
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def read_turbines(
    path: Path | None = None, expected_sha256: str = ROWS_SHA256
) -> DataFrame[Turbines]:
    """`Turbines` of the map, downloaded when `path` is None. Raises ValueError when its rows
    differ from those pinned by `expected_sha256`."""
    text = pd.read_csv(path or download(), dtype=str, keep_default_na=False)
    digest = rows_sha256(text)
    if digest != expected_sha256:
        raise ValueError(f"{LAYER}: rows with SHA-256 {digest}, expected {expected_sha256}")
    frame = text[list(COLUMNS)].rename(columns=COLUMNS).replace("", None)
    numbers = ["capacity_kw", "rotor_diameter_m", "hub_height_m", "x", "y"]
    frame[numbers] = frame[numbers].astype(float)
    points = gpd.points_from_xy(frame["x"], frame["y"], crs=MAP_CRS).to_crs("EPSG:4326")
    turbines = frame.drop(columns=["x", "y"]).assign(
        turbine=frame["turbine"].astype(int), longitude=points.x, latitude=points.y
    )
    return Turbines.validate(turbines[list(Turbines.to_schema().columns)])


def onshore(turbines: pd.DataFrame) -> DataFrame[Turbines]:
    """The Dutch turbines on land or in inland waters, which CBS counts as wind on land."""
    kept = Turbines.validate(turbines)
    kept = kept[(kept["country"] == NETHERLANDS) & kept["ground"].isin(ONSHORE_GROUNDS)]
    return Turbines.validate(kept.reset_index(drop=True))


def hub_height_classes(turbines: pd.DataFrame) -> DataFrame[HubHeightClasses]:
    """`HubHeightClasses` of `turbines`, one row per class that has turbines, in order of hub
    height."""
    counted = Turbines.validate(turbines)
    rounded = counted["hub_height_m"].round()
    rows = []
    low = -np.inf
    for name, high in HUB_HEIGHT_CLASSES.items():
        top = np.inf if high is None else high
        members = counted[(rounded > low) & (rounded <= top)]
        low = top
        if members.empty:
            continue
        swept_m2 = np.pi * (members["rotor_diameter_m"] / 2) ** 2
        rows.append(
            {
                "hub_height_class": name,
                "capacity_mw": members["capacity_kw"].sum() / 1000,
                "hub_height_m": float(
                    np.average(members["hub_height_m"], weights=members["capacity_kw"])
                ),
                "specific_power_w_m2": float(members["capacity_kw"].sum() * 1000 / swept_m2.sum()),
            }
        )
    return HubHeightClasses.validate(pd.DataFrame(rows))


def municipality_capacity(turbines: pd.DataFrame, municipalities: gpd.GeoDataFrame) -> pd.DataFrame:
    """Capacity in MW of `turbines` in each province and municipality: `province`, `region` (the
    `region` of `municipalities`, polygons in EPSG:4326) and `capacity_mw`. A turbine outside
    every polygon goes to the nearest one."""
    counted = Turbines.validate(turbines)
    points = gpd.GeoDataFrame(
        counted,
        geometry=gpd.points_from_xy(counted["longitude"], counted["latitude"]),
        crs="EPSG:4326",
    )
    polygons = municipalities[["region", "geometry"]]
    within = gpd.sjoin(points, polygons, how="left", predicate="within")
    outside = within["region"].isna().to_numpy()
    # Distances in metres, on the grid of the map.
    nearest = gpd.sjoin_nearest(
        points[outside].to_crs(MAP_CRS), polygons.to_crs(MAP_CRS), how="left"
    )
    kept = ["province", "region", "capacity_kw"]
    placed = pd.concat([within.loc[~outside, kept], nearest[kept]])
    # A point on the border of two polygons joins both: it counts once, in the first.
    placed = placed[~placed.index.duplicated()]
    capacity = (placed.groupby(["province", "region"])["capacity_kw"].sum() / 1000).rename(
        "capacity_mw"
    )
    return capacity.reset_index()
