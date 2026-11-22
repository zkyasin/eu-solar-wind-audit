"""Offshore wind farms from the EMODnet Human Activities wind farm dataset, compiled by CETMAR
(licence CC BY 4.0, attribution "This data was downloaded from the EMODnet Portal
(https://emodnet.ec.europa.eu/en/). The data originator is CETMAR.").

Farms are read from the polygon layer, located at the centroid of their area. The point layer is
not used: for these five countries its operating entries are mostly duplicates of polygons with
coarse or wrong coordinates, or farms in the IJsselmeer that national statistics count as onshore.
No field marks onshore farms, and the polygon layer holds none for these countries.

A farm built in phases is either one record per phase, each with its own capacity and start year
(Trianel Windpark Borkum Phase 1 and 2), or one record for the whole farm with no start year
(C-Power, Hollandse Kust (Zuid)). When a whole-farm record replaces the phase records, the older
ones can stay in the layer (C-Power next to Thorntonbank Part 1 and 2); `drop_superseded` removes
them. The start year is missing for some single-phase farms too, and is sometimes wrong.
"""

import json
import time
from datetime import date
from pathlib import Path
from typing import Any

import geopandas as gpd
import httpx
import pandas as pd
import pandera.pandas as pa
import shapely
from pandera.typing import DataFrame, Series

URL = "https://ows.emodnet-humanactivities.eu/wfs"
LAYER = "emodnet:windfarmspoly"
CACHE_DIR = Path("data/raw/emodnet")
TIMEOUT_SECONDS = 60.0
RETRY_DELAYS_SECONDS = (2.0, 10.0)
# Lambert azimuthal equal-area for Europe, for areas and centroids in metres.
EQUAL_AREA = "EPSG:3035"

COUNTRIES = {
    "BE": "Belgium",
    "DE": "Germany",
    "DK": "Denmark",
    "FR": "France",
    "NL": "Netherlands",
}
STATUSES = {
    "Planned": "planned",
    "Approved": "approved",
    "Construction": "construction",
    "Production": "production",
    "Test site": "test_site",
    "Dismantled": "dismantled",
}
# A newer record supersedes an older one only at the same or a later stage.
STAGES = {"planned": 0, "approved": 1, "construction": 2, "production": 3}
SUPERSEDED_COVERAGE = 0.9


class OffshoreFarms(pa.DataFrameModel):
    # Some planned zones have no name or capacity.
    name: Series[str] = pa.Field(nullable=True)
    country: Series[str] = pa.Field(isin=list(COUNTRIES))
    status: Series[str] = pa.Field(isin=list(STATUSES.values()))
    capacity_mw: Series[float] = pa.Field(gt=0, nullable=True)
    year: Series[pd.Int64Dtype] = pa.Field(nullable=True)
    latitude: Series[float] = pa.Field(ge=-90, le=90)
    longitude: Series[float] = pa.Field(ge=-180, le=180)

    class Config:
        strict = True


def load_offshore_farms(
    countries: list[str],
    status: tuple[str, ...] = ("production",),
    *,
    client: httpx.Client | None = None,
    cache_dir: Path = CACHE_DIR,
    refresh: bool = False,
) -> DataFrame[OffshoreFarms]:
    """Farms of `countries` (ISO codes) with one of the given statuses, superseded records
    removed. Capacity is in MW, `year` is the start year as published, and latitude and longitude
    (WGS84 degrees) locate the centroid of the farm area."""
    path = download_layer(countries, client=client, cache_dir=cache_dir, refresh=refresh)
    return parse_farms(json.loads(path.read_bytes()), countries, status)


def download_layer(
    countries: list[str],
    *,
    client: httpx.Client | None = None,
    cache_dir: Path = CACHE_DIR,
    refresh: bool = False,
) -> Path:
    """Path of the latest cached GeoJSON for `countries`, downloaded if there is none or if
    `refresh` is set. The file name holds the retrieval date."""
    codes = "-".join(sorted(countries))
    cached = sorted(cache_dir.glob(f"windfarmspoly_{codes}_*.geojson"))
    if cached and not refresh:
        return cached[-1]

    names = ", ".join(f"'{COUNTRIES[code]}'" for code in sorted(countries))
    params = {
        "service": "WFS",
        "version": "2.0.0",
        "request": "GetFeature",
        "typeNames": LAYER,
        "outputFormat": "application/json",
        "srsName": "EPSG:4326",
        "CQL_FILTER": f"country IN ({names})",
    }
    if client is None:
        with httpx.Client(timeout=TIMEOUT_SECONDS) as own_client:
            content = _get(own_client, params)
    else:
        content = _get(client, params)
    # Checked before caching, so that an error report is never kept as data.
    payload: dict[str, Any] = json.loads(content) if content.startswith(b"{") else {}
    if payload.get("type") != "FeatureCollection":
        raise ValueError(f"not a feature collection: {content[:200]!r}")
    if payload["numberReturned"] != payload["numberMatched"]:
        raise ValueError(f"{payload['numberReturned']} of {payload['numberMatched']} features")

    dest = cache_dir / f"windfarmspoly_{codes}_{date.today().isoformat()}.geojson"
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Write under a temporary name so that an interrupted write never looks complete.
    partial = dest.with_name(dest.name + ".part")
    partial.write_bytes(content)
    partial.replace(dest)
    return dest


def parse_farms(
    payload: dict[str, Any], countries: list[str], status: tuple[str, ...]
) -> DataFrame[OffshoreFarms]:
    unknown = set(status) - set(STATUSES.values())
    if unknown:
        raise ValueError(f"unknown statuses: {sorted(unknown)}")
    farms = gpd.GeoDataFrame.from_features(payload["features"], crs="EPSG:4326")
    farms = farms[farms["country"].isin([COUNTRIES[code] for code in countries])]
    farms = farms.assign(status=farms["status"].map(STATUSES))
    if farms["status"].isna().any():
        raise ValueError("a farm has an unknown status")
    farms = drop_superseded(farms)
    farms = farms[farms["status"].isin(status)]

    centroids = farms.geometry.to_crs(EQUAL_AREA).centroid.to_crs("EPSG:4326")
    table = pd.DataFrame(
        {
            "name": farms["name"],
            "country": farms["country"].map({name: code for code, name in COUNTRIES.items()}),
            "status": farms["status"],
            "capacity_mw": farms["power_mw"].astype("float64"),
            "year": pd.to_numeric(farms["year"]).astype("Int64"),
            "latitude": centroids.y,
            "longitude": centroids.x,
        }
    )
    return OffshoreFarms.validate(table.reset_index(drop=True))


def drop_superseded(farms: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Drop each record that lies mostly inside a record with a later update year and the same
    or a later stage: the layer keeps older records of a farm next to the current one."""
    shapes = farms.geometry.to_crs(EQUAL_AREA).to_numpy()
    older, newer = shapely.STRtree(shapes).query(shapes, predicate="intersects")
    overlap = shapely.area(shapely.intersection(shapes[older], shapes[newer]))
    covered = overlap >= SUPERSEDED_COVERAGE * shapely.area(shapes[older])
    # A missing update year counts as the oldest.
    updated = pd.to_numeric(farms["updateyear"]).fillna(0).to_numpy()
    stage = farms["status"].map(STAGES).to_numpy(dtype="float64")
    superseded = covered & (updated[newer] > updated[older]) & (stage[newer] >= stage[older])
    kept: gpd.GeoDataFrame = farms.drop(index=farms.index[older[superseded]])
    return kept


def _get(client: httpx.Client, params: dict[str, str]) -> bytes:
    """Retries timeouts, connection errors and server errors; other HTTP errors raise at once."""
    for delay in RETRY_DELAYS_SECONDS:
        try:
            response = client.get(URL, params=params)
            if response.status_code < 500:
                break
        except httpx.TransportError:
            pass
        time.sleep(delay)
    else:
        response = client.get(URL, params=params)
    response.raise_for_status()
    return response.content
