"""Admin-1 regions (states, provinces, departments) from the Natural Earth 1:10m vector data,
which are in the public domain.

The file is pinned to a tagged release of github.com/nvkelso/natural-earth-vector and checked
against its SHA-256 before it is cached.
"""

from pathlib import Path

import geopandas as gpd
import httpx

from eu_solar_wind_audit.sources import pinned

VERSION = "v5.1.2"
URL = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
    f"{VERSION}/geojson/ne_10m_admin_1_states_provinces.geojson"
)
# Computed on the download of 2026-11-22; its git blob hash matches the tag's tree entry.
SHA256 = "22d0e3ad85eb3e27f17cabf8ba2d50e554fbc27a87796ff891d958185da62fb5"
RAW_DIR = Path("data/raw/natural_earth")


def download_admin1(client: httpx.Client | None = None, raw_dir: Path = RAW_DIR) -> Path:
    """Path of the pinned admin-1 GeoJSON, downloaded and checked unless already cached."""
    dest = raw_dir / f"ne_10m_admin_1_states_provinces_{VERSION}.geojson"
    return pinned.download(URL, SHA256, dest, client)


def load_admin1(country: str, path: Path | None = None) -> gpd.GeoDataFrame:
    """Admin-1 regions of `country` (ISO 3166-1 alpha-2, "FR") in EPSG:4326.

    `region` is the ISO 3166-2 code without its country prefix: "01".."95", "2A", "2B" for the
    French metropolitan departments, as in the SDES tables; the overseas departments keep their
    letter codes ("GF", "GP", "MQ", "RE", "YT").
    """
    if path is None:
        path = download_admin1()
    admin1 = gpd.read_file(path, columns=["iso_a2", "iso_3166_2", "name"])
    regions = admin1[admin1["iso_a2"] == country]
    if regions.empty:
        raise ValueError(f"no admin-1 regions for {country!r}")
    regions = regions.assign(region=regions["iso_3166_2"].str.removeprefix(f"{country}-"))
    return regions[["region", "name", "geometry"]].reset_index(drop=True).to_crs("EPSG:4326")
