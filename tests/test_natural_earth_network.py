import hashlib
import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import xarray as xr

from eu_solar_wind_audit.sources import natural_earth
from eu_solar_wind_audit.spatial import capacity_weights, region_fractions

pytestmark = pytest.mark.skipif(os.environ.get("RUN_NETWORK_TESTS") != "1", reason="network")

SDES_METROPOLITAN = {f"{number:02d}" for number in range(1, 96) if number != 20} | {"2A", "2B"}
OVERSEAS = {"GF", "GP", "MQ", "RE", "YT"}
LAENDER = {"BB", "BE", "BW", "BY", "HB", "HE", "HH", "MV", "NI", "NW", "RP", "SH", "SL", "SN"}
LAENDER |= {"ST", "TH"}
# IGN area of metropolitan France including inland water, as cited in
# https://en.wikipedia.org/wiki/Metropolitan_France (read 2026-11-22); the land register gives
# 543,940 km2 without lakes, ponds, glaciers and estuaries.
METROPOLITAN_FRANCE_KM2 = 551_695.0
# Authalic radius of GRS80 (Moritz 2000, Geodetic Reference System 1980).
EARTH_RADIUS_KM = 6371.0071810
STEP = 0.25


@pytest.fixture(scope="module")
def admin1_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return natural_earth.download_admin1(raw_dir=tmp_path_factory.mktemp("natural_earth"))


@pytest.fixture(scope="module")
def metropolitan_france(admin1_path: Path) -> gpd.GeoDataFrame:
    france = natural_earth.load_admin1("FR", admin1_path)
    return france[~france["region"].isin(OVERSEAS)]


@pytest.fixture(scope="module")
def france_fractions(metropolitan_france: gpd.GeoDataFrame) -> xr.DataArray:
    latitude = np.arange(41.0, 51.5 + STEP, STEP)
    longitude = np.arange(-5.5, 10.0 + STEP, STEP)
    return region_fractions(metropolitan_france, latitude, longitude)


def test_download_matches_the_pinned_checksum(admin1_path: Path) -> None:
    assert hashlib.sha256(admin1_path.read_bytes()).hexdigest() == natural_earth.SHA256


def test_french_departments_match_the_sdes_codes(admin1_path: Path) -> None:
    france = natural_earth.load_admin1("FR", admin1_path)

    assert france["region"].is_unique
    assert set(france["region"]) == SDES_METROPOLITAN | OVERSEAS


def test_germany_has_its_16_laender(admin1_path: Path) -> None:
    germany = natural_earth.load_admin1("DE", admin1_path)

    assert len(germany) == 16
    assert set(germany["region"]) == LAENDER


def test_grid_area_of_metropolitan_france(france_fractions: xr.DataArray) -> None:
    latitude = np.deg2rad(france_fractions["latitude"])
    half = np.deg2rad(STEP) / 2
    cell_km2 = EARTH_RADIUS_KM**2 * 2 * half * (np.sin(latitude + half) - np.sin(latitude - half))

    area = float((france_fractions * cell_km2).sum())

    assert area == pytest.approx(METROPOLITAN_FRANCE_KM2, rel=0.02)


def test_every_department_gets_its_capacity(france_fractions: xr.DataArray) -> None:
    capacity = xr.ones_like(france_fractions["region"], dtype=float)

    weights = capacity_weights(france_fractions, capacity)

    assert float(weights.sum()) == pytest.approx(len(SDES_METROPOLITAN))
