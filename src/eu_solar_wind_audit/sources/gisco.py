"""Municipalities from the local administrative units (LAU) of GISCO, the geographic information
system of the European Commission at Eurostat: the LAU 2024 edition at 1:1,000,000 in EPSG:4326,
one file for every country.

The download rules of GISCO (statistical units page, read 2026-11-25) allow non-commercial use
only, with the notice NOTICE on any publication that uses the data.

A unit's GISCO_ID is its country code, an underscore and its national code: "DK_101" for
Copenhagen, whose municipality number is 101, and "NL_GM0363" for Amsterdam, whose CBS code is
GM0363.
"""

import re
from pathlib import Path

import geopandas as gpd
import httpx

from eu_solar_wind_audit.sources import pinned

EDITION = "LAU 2024"
URL = "https://gisco-services.ec.europa.eu/distribution/v2/lau/shp/LAU_RG_01M_2024_4326.shp.zip"
# Computed on the download of 2026-11-25, the release of 2026-02-18, which updated the codes of
# Norway (lau-2024-release-notes.txt).
SHA256 = "8eb297d914133b04415a32e768d88e046c4dc0665502223a20444f2305a0eb42"
RAW_DIR = Path("data/raw/gisco")
NOTICE = "© EuroGeographics for the administrative boundaries"


def download_lau(client: httpx.Client | None = None, raw_dir: Path = RAW_DIR) -> Path:
    """Path of the pinned LAU zip, downloaded and checked unless already cached."""
    return pinned.download(URL, SHA256, raw_dir / "LAU_RG_01M_2024_4326.shp.zip", client)


def load_municipalities(country: str, path: Path | None = None) -> gpd.GeoDataFrame:
    """Municipalities of `country`, the two-letter code GISCO gives it ("DK", "NL"), in
    EPSG:4326: `region` is the national code ("101", "GM0363") and `name` the name."""
    # The code goes into an SQL filter.
    if not re.fullmatch(r"[A-Z]{2}", country):
        raise ValueError(f"not a two-letter country code: {country!r}")
    if path is None:
        path = download_lau()
    # The filter reads only the columns it is given.
    units = gpd.read_file(
        f"zip://{path}",
        where=f"CNTR_CODE = '{country}'",
        columns=["GISCO_ID", "CNTR_CODE", "LAU_NAME"],
    )
    if units.empty:
        raise ValueError(f"no municipality of {country} in {EDITION}")
    region = units["GISCO_ID"].str.removeprefix(f"{country}_")
    if not region.is_unique:
        raise ValueError(f"municipality codes of {country} are not unique")
    municipalities = gpd.GeoDataFrame(
        {"region": region, "name": units["LAU_NAME"]}, geometry=units.geometry, crs=units.crs
    )
    return municipalities.sort_values("region").reset_index(drop=True).to_crs("EPSG:4326")
