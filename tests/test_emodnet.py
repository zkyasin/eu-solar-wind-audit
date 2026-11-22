import json
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

import geopandas as gpd
import httpx
import pandas as pd
import pytest
from shapely.geometry import Polygon, mapping

from eu_solar_wind_audit.sources import emodnet

# Fixture: windfarmspoly features 122, 154, 161, 177, 188, 189, 192, 540 and 580, cut from the
# response of emodnet.download_layer(["BE", "DE", "DK", "FR", "NL"]) retrieved 2026-11-22; features
# kept verbatim, the three feature counts set to 9.
FIXTURE = Path(__file__).parent / "fixtures" / "emodnet" / "windfarmspoly_BE-DE_2026-11-22.geojson"
ALL_STATUSES = tuple(emodnet.STATUSES.values())

Handler = Callable[[httpx.Request], httpx.Response]


@pytest.fixture
def payload() -> dict[str, Any]:
    content: dict[str, Any] = json.loads(FIXTURE.read_bytes())
    return content


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("time.sleep", lambda seconds: None)


def serve(responses: list[httpx.Response], requests: list[httpx.Request]) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return responses.pop(0)

    return httpx.Client(transport=httpx.MockTransport(handler))


def square(lon: float, lat: float, side: float) -> Polygon:
    return Polygon([(lon, lat), (lon + side, lat), (lon + side, lat + side), (lon, lat + side)])


def feature(shape: Polygon, status: str, update_year: str | None) -> dict[str, Any]:
    properties = {
        "country": "Germany",
        "name": f"{status} {update_year}",
        "power_mw": 100.0,
        "status": status,
        "updateyear": update_year,
        "year": None,
    }
    return {"type": "Feature", "geometry": mapping(shape), "properties": properties}


def collection(features: list[dict[str, Any]]) -> dict[str, Any]:
    return {"type": "FeatureCollection", "features": features}


def test_operating_farms_are_parsed_with_codes_units_and_years(payload: dict[str, Any]) -> None:
    farms = emodnet.parse_farms(payload, ["BE", "DE"], ("production",))

    assert farms.set_index("name")["capacity_mw"].to_dict() == {
        "Norther": 369.6,
        "Riffgat": 113.4,
        "C-Power": 325.2,
    }
    assert farms["country"].to_list() == ["BE", "DE", "BE"]
    assert set(farms["status"]) == {"production"}
    years = farms.set_index("name")["year"]
    assert years["Norther"] == 2018
    assert years["Riffgat"] == 2014
    assert pd.isna(years["C-Power"])


def test_phase_records_replaced_by_a_whole_farm_record_are_dropped(
    payload: dict[str, Any],
) -> None:
    farms = emodnet.parse_farms(payload, ["BE"], ALL_STATUSES)

    # Thorntonbank Part 1 (30.45 MW) and Part 2 (184.5 MW), updated 2025, lie inside C-Power,
    # updated 2026.
    assert "Thorntonbank Part 1" not in farms["name"].to_list()
    assert "Thorntonbank Part 2" not in farms["name"].to_list()
    assert farms.loc[farms["status"] == "production", "capacity_mw"].sum() == pytest.approx(694.8)


def test_a_later_stage_supersedes_an_older_approval(payload: dict[str, Any]) -> None:
    farms = emodnet.parse_farms(payload, ["DE"], ALL_STATUSES)

    # Borkum Riffgrund West (approved, no update year) lies inside Borkum Riffgrund 3.
    assert farms.set_index("name")["status"].to_dict() == {
        "Riffgat": "production",
        "N-3.7": "planned",
        "Borkum Riffgrund 3": "construction",
    }


def test_status_filter(payload: dict[str, Any]) -> None:
    planned = emodnet.parse_farms(payload, ["BE", "DE"], ("planned",))
    building = emodnet.parse_farms(payload, ["BE", "DE"], ("construction", "approved"))

    assert planned["name"].to_list() == ["Princess Elisabeth Zone Lot 1", "N-3.7"]
    assert building["name"].to_list() == ["Borkum Riffgrund 3"]


def test_unknown_status_is_rejected(payload: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="Production"):
        emodnet.parse_farms(payload, ["BE"], ("Production",))


def test_country_filter(payload: dict[str, Any]) -> None:
    belgium = emodnet.parse_farms(payload, ["BE"], ALL_STATUSES)
    germany = emodnet.parse_farms(payload, ["DE"], ALL_STATUSES)

    assert set(belgium["country"]) == {"BE"}
    assert len(belgium) == 3
    assert set(germany["country"]) == {"DE"}
    assert len(germany) == 3


def test_locations_are_area_centroids_in_an_equal_area_projection(
    payload: dict[str, Any],
) -> None:
    farms = emodnet.parse_farms(payload, ["BE", "DE"], ("production",)).set_index("name")

    shapes = gpd.GeoDataFrame.from_features(payload["features"], crs="EPSG:4326")
    norther = shapes[shapes["name"] == "Norther"].geometry
    expected = norther.to_crs("EPSG:3035").centroid.to_crs("EPSG:4326")
    assert farms.loc[["Norther"], "longitude"].to_list() == pytest.approx(expected.x.to_list())
    assert farms.loc[["Norther"], "latitude"].to_list() == pytest.approx(expected.y.to_list())
    assert norther.contains(expected, align=False).all()


def test_centroid_is_not_taken_in_degrees() -> None:
    # A triangle spanning 10 degrees of latitude: the centroid of its outline in degrees lies
    # several kilometres from the centroid of its area on the ground.
    triangle = Polygon([(0.0, 45.0), (10.0, 45.0), (5.0, 55.0)])
    farms = emodnet.parse_farms(
        collection([feature(triangle, "Production", "2026")]), ["DE"], ("production",)
    )

    projected = gpd.GeoSeries([triangle], crs="EPSG:4326").to_crs("EPSG:3035").centroid
    expected = projected.to_crs("EPSG:4326")
    assert farms["latitude"].to_list() == pytest.approx(expected.y.to_list())
    assert ((farms["latitude"] - triangle.centroid.y).abs() > 0.05).all()


def test_a_newer_planned_zone_does_not_remove_operating_farms() -> None:
    zone = feature(square(3.0, 51.6, 0.4), "Planned", "2026")
    farm = feature(square(3.1, 51.7, 0.05), "Production", "2025")

    farms = emodnet.parse_farms(collection([zone, farm]), ["DE"], ALL_STATUSES)

    assert sorted(farms["status"]) == ["planned", "production"]


def test_records_with_the_same_update_year_are_both_kept() -> None:
    # A large farm polygon can contain a smaller, separate farm.
    large = feature(square(4.0, 52.5, 0.3), "Production", "2026")
    small = feature(square(4.1, 52.6, 0.05), "Production", "2026")

    farms = emodnet.parse_farms(collection([large, small]), ["DE"], ("production",))

    assert len(farms) == 2


def test_a_partly_covered_record_is_kept() -> None:
    newer = feature(square(6.0, 54.0, 0.1), "Production", "2026")
    older = feature(square(6.05, 54.0, 0.1), "Production", "2024")

    farms = emodnet.parse_farms(collection([newer, older]), ["DE"], ("production",))

    assert len(farms) == 2


def test_download_is_cached_under_the_retrieval_date(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []
    responses = [httpx.Response(200, content=FIXTURE.read_bytes())]
    with serve(responses, requests) as client:
        first = emodnet.download_layer(["DE", "BE"], client=client, cache_dir=tmp_path)
        second = emodnet.download_layer(["BE", "DE"], client=client, cache_dir=tmp_path)

    assert first == second == tmp_path / f"windfarmspoly_BE-DE_{date.today().isoformat()}.geojson"
    assert first.read_bytes() == FIXTURE.read_bytes()
    assert len(requests) == 1
    params = requests[0].url.params
    assert params["typeNames"] == "emodnet:windfarmspoly"
    assert params["outputFormat"] == "application/json"
    assert params["CQL_FILTER"] == "country IN ('Belgium', 'Germany')"


def test_the_latest_cached_file_is_used_unless_refreshed(tmp_path: Path) -> None:
    old = tmp_path / "windfarmspoly_BE_2026-01-05.geojson"
    new = tmp_path / "windfarmspoly_BE_2026-07-20.geojson"
    for path in (old, new):
        path.write_bytes(b"{}")

    assert emodnet.download_layer(["BE"], cache_dir=tmp_path) == new

    requests: list[httpx.Request] = []
    responses = [httpx.Response(200, content=FIXTURE.read_bytes())]
    with serve(responses, requests) as client:
        fresh = emodnet.download_layer(["BE"], client=client, cache_dir=tmp_path, refresh=True)
    assert fresh.name == f"windfarmspoly_BE_{date.today().isoformat()}.geojson"
    assert len(requests) == 1


def test_an_exception_report_is_not_cached(tmp_path: Path) -> None:
    report = b'<?xml version="1.0"?><ows:ExceptionReport version="2.0.0"/>'
    with (
        serve([httpx.Response(200, content=report)], []) as client,
        pytest.raises(ValueError, match="not a feature collection"),
    ):
        emodnet.download_layer(["BE"], client=client, cache_dir=tmp_path)

    assert list(tmp_path.iterdir()) == []


def test_a_truncated_response_is_rejected(tmp_path: Path, payload: dict[str, Any]) -> None:
    payload["numberMatched"] = 10
    responses = [httpx.Response(200, content=json.dumps(payload).encode())]
    with serve(responses, []) as client, pytest.raises(ValueError, match="9 of 10"):
        emodnet.download_layer(["BE"], client=client, cache_dir=tmp_path)

    assert list(tmp_path.iterdir()) == []


def test_server_errors_are_retried(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []
    responses = [httpx.Response(503), httpx.Response(200, content=FIXTURE.read_bytes())]
    with serve(responses, requests) as client:
        farms = emodnet.load_offshore_farms(["BE"], client=client, cache_dir=tmp_path)

    assert len(requests) == 2
    assert farms["name"].to_list() == ["Norther", "C-Power"]


def test_client_errors_are_not_retried(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []
    with (
        serve([httpx.Response(400)], requests) as client,
        pytest.raises(httpx.HTTPStatusError),
    ):
        emodnet.download_layer(["BE"], client=client, cache_dir=tmp_path)

    assert len(requests) == 1
