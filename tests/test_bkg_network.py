import hashlib
import os
from pathlib import Path

import geopandas as gpd
import pytest

from eu_solar_wind_audit.sources import bkg

pytestmark = pytest.mark.skipif(os.environ.get("RUN_NETWORK_TESTS") != "1", reason="network")

# Area of Germany in 2023, https://www.statistikportal.de/de/bevoelkerung/flaeche-und-bevoelkerung
# (read 2026-11-22).
GERMANY_KM2 = 357_684.19


@pytest.fixture(scope="module")
def vg5000(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return bkg.download_vg5000(raw_dir=tmp_path_factory.mktemp("bkg"))


@pytest.fixture(scope="module")
def districts(vg5000: Path) -> gpd.GeoDataFrame:
    return bkg.load_districts(vg5000)


def test_download_matches_the_pinned_checksum(vg5000: Path) -> None:
    with vg5000.open("rb") as file:
        assert hashlib.file_digest(file, "sha256").hexdigest() == bkg.SHA256


def test_edition_has_401_districts(districts: gpd.GeoDataFrame) -> None:
    # 400 from 2021 to 2025; Hanau (06415) left the Main-Kinzig-Kreis (06435) on 2026-01-01.
    assert len(districts) == 401
    assert {"06415", "06435"} <= set(districts["region"])


def test_known_district_codes(districts: gpd.GeoDataFrame) -> None:
    names = districts.set_index("region")["name"]

    assert names["11000"] == "Berlin"
    assert names["02000"] == "Hamburg"
    assert names["09162"] == "München"


def test_laender_match_the_district_codes(vg5000: Path, districts: gpd.GeoDataFrame) -> None:
    laender = bkg.load_laender(vg5000)

    assert laender["region"].to_list() == [f"{number:02d}" for number in range(1, 17)]
    assert set(districts["region"].str[:2]) == set(laender["region"])


def test_total_land_area_is_close_to_the_official_area(
    vg5000: Path, districts: gpd.GeoDataFrame
) -> None:
    laender = bkg.load_laender(vg5000)
    district_km2 = districts.to_crs("EPSG:3035").area.sum() / 1e6
    land_km2 = laender.to_crs("EPSG:3035").area.sum() / 1e6

    # 357,218 km2 on the 2026 edition: the generalised coastline loses about 0.13%.
    assert district_km2 == pytest.approx(GERMANY_KM2, rel=0.005)
    assert land_km2 == pytest.approx(district_km2, rel=1e-6)
