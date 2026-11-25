import hashlib
from pathlib import Path

import httpx
import pytest

from eu_solar_wind_audit.sources import natural_earth

# Fixture: features FR-75, FR-2A and DE-BE of natural_earth.URL (v5.1.2), retrieved 2026-11-22,
# geometry unchanged, properties cut to iso_a2, iso_3166_2 and name.
FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "natural_earth"
    / "ne_10m_admin_1_paris_corse-du-sud_berlin.geojson"
)


def serve(status: int, content: bytes, requests: list[str]) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(str(request.url))
        return httpx.Response(status, content=content)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_french_departments_keep_the_sdes_codes() -> None:
    france = natural_earth.load_admin1("FR", FIXTURE)

    assert list(france.columns) == ["region", "name", "geometry"]
    assert sorted(france["region"]) == ["2A", "75"]
    assert france.crs == "EPSG:4326"
    paris = france[france["region"] == "75"].geometry.iloc[0]
    assert paris.contains(paris.representative_point())
    assert 2.2 < paris.centroid.x < 2.5
    assert 48.8 < paris.centroid.y < 48.95


def test_regions_are_filtered_by_country() -> None:
    germany = natural_earth.load_admin1("DE", FIXTURE)

    assert germany["region"].to_list() == ["BE"]
    assert germany["name"].to_list() == ["Berlin"]


def test_unknown_country_raises() -> None:
    with pytest.raises(ValueError, match="XX"):
        natural_earth.load_admin1("XX", FIXTURE)


def test_download_is_checked_then_cached(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    body = FIXTURE.read_bytes()
    monkeypatch.setattr(natural_earth, "SHA256", hashlib.sha256(body).hexdigest())
    requests: list[str] = []

    with serve(200, body, requests) as client:
        first = natural_earth.download_admin1(client, tmp_path)
        second = natural_earth.download_admin1(client, tmp_path)

    assert requests == [natural_earth.URL]
    assert first == second
    assert first.read_bytes() == body
    assert natural_earth.VERSION in first.name
