import hashlib
from pathlib import Path

import httpx
import pytest
from shapely.geometry import Point

from eu_solar_wind_audit.sources import gisco

# Fixture: gisco.URL (LAU 2024, release of 2026-02-18), retrieved 2026-11-25; units DK_101, DK_400,
# DK_411 and NL_GM0363, geometry unchanged, columns cut to GISCO_ID, CNTR_CODE and LAU_NAME,
# rewritten as a shapefile with the archive's member names.
FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "gisco"
    / "lau_2024_copenhagen_bornholm_christianso_amsterdam.shp.zip"
)


def test_danish_municipalities_keep_their_numbers() -> None:
    denmark = gisco.load_municipalities("DK", FIXTURE)

    assert list(denmark.columns) == ["region", "name", "geometry"]
    assert denmark["region"].to_list() == ["101", "400", "411"]
    assert denmark["name"].to_list() == ["København", "Bornholm", "Christiansø"]
    assert denmark.crs == "EPSG:4326"


def test_municipalities_hold_their_town_halls() -> None:
    denmark = gisco.load_municipalities("DK", FIXTURE).set_index("region")

    # Københavns Rådhus and Rønne (longitude, latitude).
    assert denmark.geometry["101"].contains(Point(12.5700, 55.6761))
    assert denmark.geometry["400"].contains(Point(14.7065, 55.1007))
    assert not denmark.geometry["101"].contains(Point(14.7065, 55.1007))


def test_dutch_municipalities_keep_their_cbs_codes() -> None:
    netherlands = gisco.load_municipalities("NL", FIXTURE)

    assert netherlands["region"].to_list() == ["GM0363"]
    assert netherlands["name"].to_list() == ["Amsterdam"]


def test_unknown_country_raises() -> None:
    with pytest.raises(ValueError, match="no municipality of XX"):
        gisco.load_municipalities("XX", FIXTURE)


@pytest.mark.parametrize("country", ["dk", "DK' OR '1'='1"])
def test_malformed_country_code_raises(country: str) -> None:
    with pytest.raises(ValueError, match="not a two-letter country code"):
        gisco.load_municipalities(country, FIXTURE)


def test_download_is_checked_then_cached(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    body = FIXTURE.read_bytes()
    monkeypatch.setattr(gisco, "SHA256", hashlib.sha256(body).hexdigest())
    requests: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(str(request.url))
        return httpx.Response(200, content=body)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        first = gisco.download_lau(client, tmp_path)
        second = gisco.download_lau(client, tmp_path)

    assert requests == [gisco.URL]
    assert first == second
    assert first.name == "LAU_RG_01M_2024_4326.shp.zip"
    assert gisco.load_municipalities("NL", first)["region"].to_list() == ["GM0363"]
