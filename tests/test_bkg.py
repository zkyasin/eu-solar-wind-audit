import hashlib
import zipfile
from collections.abc import Callable
from pathlib import Path

import geopandas as gpd
import httpx
import pytest
from shapely.geometry import Point

from eu_solar_wind_audit.sources import bkg

# Fixture: bkg.URL (VG5000, state 2026-01-01), retrieved 2026-11-22; layers vg5000_krs (AGS 02000,
# 09162, 11000) and vg5000_lan (AGS 02, 09, 11, Bodensee rows dropped) in EPSG:25832, geometry
# unchanged, columns cut to AGS, GEN, BEZ, GF, rewritten as a GeoPackage at bkg.MEMBER in a zip.
FIXTURE = Path(__file__).parent / "fixtures" / "bkg" / "vg5000_hamburg_muenchen_berlin.gpkg.zip"
# Official areas in km2 for 2023, https://www.statistikportal.de/de/bevoelkerung/flaeche-und-bevoelkerung
# (read 2026-11-22).
BERLIN_KM2 = 891.12
HAMBURG_KM2 = 755.09
LAEA_EUROPE = "EPSG:3035"

Handler = Callable[[httpx.Request], httpx.Response]


def serve(handler: Handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def respond(status: int, content: bytes, requests: list[str]) -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(str(request.url))
        return httpx.Response(status, content=content)

    return handler


def area_km2(frame: gpd.GeoDataFrame, region: str) -> float:
    return float(frame[frame["region"] == region].to_crs(LAEA_EUROPE).area.iloc[0]) / 1e6


def test_district_codes_are_five_digit_strings() -> None:
    districts = bkg.load_districts(FIXTURE)

    assert list(districts.columns) == ["region", "name", "kind", "geometry"]
    assert districts["region"].to_list() == ["02000", "09162", "11000"]
    assert districts["name"].to_list() == ["Hamburg", "München", "Berlin"]
    assert set(districts["kind"]) == {"Kreisfreie Stadt"}


def test_districts_are_reprojected_to_wgs84() -> None:
    districts = bkg.load_districts(FIXTURE).set_index("region")

    assert districts.crs == "EPSG:4326"
    # Rathaus Hamburg, Marienplatz München, Brandenburger Tor Berlin (longitude, latitude).
    assert districts.geometry["02000"].contains(Point(9.9925, 53.5503))
    assert districts.geometry["09162"].contains(Point(11.5755, 48.1374))
    assert districts.geometry["11000"].contains(Point(13.3777, 52.5163))
    assert not districts.geometry["11000"].contains(Point(11.5755, 48.1374))


def test_district_area_matches_the_official_area() -> None:
    districts = bkg.load_districts(FIXTURE)

    assert area_km2(districts, "11000") == pytest.approx(BERLIN_KM2, rel=0.01)


def test_land_codes_are_the_first_two_digits_of_district_codes() -> None:
    districts = bkg.load_districts(FIXTURE)
    laender = bkg.load_laender(FIXTURE)

    assert list(laender.columns) == ["region", "name", "geometry"]
    assert laender["region"].to_list() == ["02", "09", "11"]
    assert laender["name"].to_list() == ["Hamburg", "Bayern", "Berlin"]
    assert laender.crs == "EPSG:4326"
    assert set(districts["region"].str[:2]) == set(laender["region"])


def test_laender_leave_out_the_coastal_sea() -> None:
    raw = gpd.read_file(f"zip://{FIXTURE}!{bkg.MEMBER}", layer="vg5000_lan")
    assert (raw["AGS"] == "02").sum() == 2

    laender = bkg.load_laender(FIXTURE)

    assert (laender["region"] == "02").sum() == 1
    # Generalisation at 1:5,000,000 loses about 2% of the city-state.
    assert area_km2(laender, "02") == pytest.approx(HAMBURG_KM2, rel=0.03)


@pytest.mark.parametrize(
    ("index", "code"), [(0, "2000"), (1, "11000")], ids=["leading zero lost", "duplicate"]
)
def test_malformed_or_duplicate_codes_raise(tmp_path: Path, index: int, code: str) -> None:
    districts = gpd.read_file(f"zip://{FIXTURE}!{bkg.MEMBER}", layer="vg5000_krs")
    districts.loc[index, "AGS"] = code
    gpkg = tmp_path / "DE_VG5000.gpkg"
    districts.to_file(gpkg, layer="vg5000_krs")
    path = tmp_path / "vg5000.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.write(gpkg, bkg.MEMBER)

    with pytest.raises(ValueError, match="5-digit"):
        bkg.load_districts(path)


def test_download_is_checked_then_cached(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    body = FIXTURE.read_bytes()
    monkeypatch.setattr(bkg, "SHA256", hashlib.sha256(body).hexdigest())
    requests: list[str] = []

    with serve(respond(200, body, requests)) as client:
        first = bkg.download_vg5000(client, tmp_path)
        second = bkg.download_vg5000(client, tmp_path)

    assert requests == [bkg.URL]
    assert first == second
    assert first.read_bytes() == body
    assert bkg.EDITION in first.name
    assert list(tmp_path.iterdir()) == [first]
