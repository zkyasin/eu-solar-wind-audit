import json
from pathlib import Path

import httpx
import pytest

from eu_solar_wind_audit.sources import cbs

PERIODS = [
    {"Key": "2024JJ00", "Title": "2024", "Status": "Definitief"},
    {"Key": "2025JJ00", "Title": "2025", "Status": "NaderVoorlopig"},
    {"Key": "2026JJ00", "Title": "2026", "Status": "Voorlopig"},
]
INFO = [{"Identifier": "70960ned", "Modified": "2026-11-13T02:00:00"}]


def serving(
    table: str, rows: list[dict[str, object]], dimension: list[dict[str, str]]
) -> tuple[httpx.Client, list[str]]:
    """A client answering the endpoints of `table`, and the paths it was asked for."""
    payloads = {
        "TypedDataSet": rows,
        "Perioden": PERIODS,
        cbs.DIMENSIONS[table]: dimension,
        "TableInfos": INFO,
    }
    paths: list[str] = []

    def answer(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        endpoint = request.url.path.rsplit("/", 1)[1]
        return httpx.Response(200, content=json.dumps({"value": payloads[endpoint]}).encode())

    return httpx.Client(transport=httpx.MockTransport(answer)), paths


def row(key: str, dimension: str, year: int, capacity: float | None) -> dict[str, object]:
    return {dimension: key, "Perioden": f"{year}JJ00", cbs.CAPACITY: capacity}


def test_a_table_is_downloaded_once(tmp_path: Path) -> None:
    client, paths = serving(cbs.PROVINCES, [], [])

    first = cbs.download(cbs.PROVINCES, client, tmp_path)
    second = cbs.download(cbs.PROVINCES, client, tmp_path)

    assert first == second == tmp_path / cbs.PROVINCES
    assert sorted(path.rsplit("/", 1)[1] for path in paths) == [
        "Perioden",
        "RegioS",
        "TableInfos",
        "TypedDataSet",
    ]
    assert cbs.modified(first) == "2026-11-13"


def test_provinces_without_the_nation_and_first_provisional_years(tmp_path: Path) -> None:
    # CBS pads the keys of regions with spaces.
    rows = [
        row(key, "RegioS", year, capacity)
        for key, capacity in [("NL01  ", 7017.0), ("PV20  ", 920.0)]
        for year in (2024, 2025, 2026)
    ]
    dimension = [
        {"Key": "NL01  ", "Title": "Nederland"},
        {"Key": "PV20  ", "Title": "Groningen (PV)"},
    ]
    client, _ = serving(cbs.PROVINCES, rows, dimension)

    capacity = cbs.province_capacity(cbs.download(cbs.PROVINCES, client, tmp_path))

    assert capacity.to_dict("records") == [
        {"year": 2024, "group": "Groningen", "capacity_mw": 920.0},
        {"year": 2025, "group": "Groningen", "capacity_mw": 920.0},
    ]


def test_hub_heights_up_to_70_m_form_one_class(tmp_path: Path) -> None:
    capacities = {
        "A044968": 7017.0,
        "A044969": 8.0,
        "A044970": 259.0,
        "A044971": None,
        "A044972": 1348.0,
        "A044973": 4930.0,
    }
    rows = [row(key, "AshoogtenWindmolens", 2025, value) for key, value in capacities.items()]
    client, _ = serving(cbs.HUB_HEIGHTS, rows, [])

    capacity = cbs.hub_height_capacity(cbs.download(cbs.HUB_HEIGHTS, client, tmp_path))

    by_class = dict(zip(capacity["group"], capacity["capacity_mw"], strict=True))
    assert by_class == pytest.approx(
        {"71 to 95 m": 1348.0, "96 m or more": 4930.0, "up to 70 m": 267.0}
    )
