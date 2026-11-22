import json
from collections.abc import Callable
from pathlib import Path

import httpx
import pandas as pd
import pytest

from eu_solar_wind_audit.sources import smard

# Fixtures: https://www.smard.de/app/chart_data/254/DE/index_quarterhour.json and
# .../254/DE/254_DE_quarterhour_1747000800000.json (French day-ahead price, week of 2025-05-12),
# retrieved 2026-11-22.
FIXTURES = Path(__file__).parent / "fixtures" / "smard"
WEEK_START = pd.Timestamp("2025-05-11 22:00", tz="UTC")
WEEK_FILE = "254_DE_quarterhour_1747000800000.json"

Handler = Callable[[httpx.Request], httpx.Response]


def file_server(files: dict[str, bytes], requests: list[str]) -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request.url.path)
        name = request.url.path.rsplit("/", 1)[-1]
        if name in files:
            return httpx.Response(200, content=files[name])
        return httpx.Response(404)

    return handler


def serve(files: dict[str, bytes], requests: list[str]) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(file_server(files, requests)))


@pytest.fixture
def real_files() -> dict[str, bytes]:
    return {
        "index_quarterhour.json": (FIXTURES / "254_DE_index_quarterhour.json").read_bytes(),
        WEEK_FILE: (FIXTURES / WEEK_FILE).read_bytes(),
    }


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("time.sleep", lambda seconds: None)


def test_fetch_downloads_only_the_overlapping_week(
    real_files: dict[str, bytes], tmp_path: Path
) -> None:
    requests: list[str] = []
    start = pd.Timestamp("2025-05-12 09:00", tz="UTC")
    with serve(real_files, requests) as client:
        prices = smard.fetch_series(
            254,
            "DE",
            "quarterhour",
            start,
            start + pd.Timedelta(hours=3),
            client=client,
            cache_dir=tmp_path,
        )

    assert requests == [
        "/app/chart_data/254/DE/index_quarterhour.json",
        f"/app/chart_data/254/DE/{WEEK_FILE}",
    ]
    expected_index = pd.date_range(start, periods=12, freq="15min")
    assert prices.index.equals(expected_index)
    assert prices.dtype == float
    assert prices.to_list() == [-0.01] * 4 + [-0.1] * 4 + [-2.54] * 4


def test_interval_end_is_exclusive_at_week_boundaries(
    real_files: dict[str, bytes], tmp_path: Path
) -> None:
    requests: list[str] = []
    end = WEEK_START + pd.Timedelta(days=7)
    with serve(real_files, requests) as client:
        prices = smard.fetch_series(
            254, "DE", "quarterhour", WEEK_START, end, client=client, cache_dir=tmp_path
        )

    # The weeks ending at WEEK_START and starting at `end` are not requested.
    assert requests[1:] == [f"/app/chart_data/254/DE/{WEEK_FILE}"]
    assert len(prices) == 7 * 96
    assert prices.index[0] == WEEK_START
    assert prices.index[-1] == end - pd.Timedelta(minutes=15)


def test_autumn_week_keeps_its_extra_hour_and_nulls_become_nan(tmp_path: Path) -> None:
    # The week of the 2025 autumn clock change lasts 169 hours in UTC.
    first = pd.Timestamp("2025-10-19 22:00", tz="UTC")
    stamps = pd.date_range(first, periods=169, freq="h")
    ms = [int(stamp.timestamp() * 1000) for stamp in stamps]
    next_week = ms[-1] + 3_600_000
    values: list[float | None] = [1.0] * 169
    values[100] = None
    files = {
        "index_hour.json": json.dumps({"timestamps": [ms[0], next_week]}).encode(),
        f"4068_DE_hour_{ms[0]}.json": json.dumps(
            {"series": list(zip(ms, values, strict=True))}
        ).encode(),
        f"4068_DE_hour_{next_week}.json": b'{"series": []}',
    }
    with serve(files, []) as client:
        solar = smard.fetch_series(
            smard.SOLAR,
            "DE",
            "hour",
            first,
            first + pd.Timedelta(days=8),
            client=client,
            cache_dir=tmp_path,
        )

    assert solar.index.equals(stamps)
    assert solar.isna().sum() == 1
    assert pd.isna(solar.iloc[100])


def test_settled_weeks_are_read_from_the_cache(
    real_files: dict[str, bytes], tmp_path: Path
) -> None:
    start, end = WEEK_START, WEEK_START + pd.Timedelta(days=1)
    with serve(real_files, []) as client:
        first = smard.fetch_series(
            254, "DE", "quarterhour", start, end, client=client, cache_dir=tmp_path
        )
    requests: list[str] = []
    with serve(real_files, requests) as client:
        second = smard.fetch_series(
            254, "DE", "quarterhour", start, end, client=client, cache_dir=tmp_path
        )

    assert requests == ["/app/chart_data/254/DE/index_quarterhour.json"]
    assert (tmp_path / WEEK_FILE).read_bytes() == real_files[WEEK_FILE]
    pd.testing.assert_series_equal(first, second)


def test_the_open_week_is_not_cached(tmp_path: Path) -> None:
    files = {
        "index_hour.json": b'{"timestamps": [1794783600000]}',
        "410_DE_hour_1794783600000.json": b'{"series": [[1794783600000, 51234.5]]}',
    }
    start = pd.Timestamp(1794783600000, unit="ms", tz="UTC")
    with serve(files, []) as client:
        load = smard.fetch_series(
            smard.LOAD,
            "DE",
            "hour",
            start,
            start + pd.Timedelta(hours=1),
            client=client,
            cache_dir=tmp_path,
        )

    assert load.to_list() == [51234.5]
    assert list(tmp_path.iterdir()) == []


def test_server_errors_and_connection_errors_are_retried(
    real_files: dict[str, bytes], tmp_path: Path
) -> None:
    requests: list[str] = []
    serve_file = file_server(real_files, requests)
    failures: list[httpx.Response | httpx.ConnectError] = [
        httpx.Response(503),
        httpx.ConnectError("refused"),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        if not failures:
            return serve_file(request)
        failure = failures.pop(0)
        if isinstance(failure, httpx.ConnectError):
            raise failure
        return failure

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        prices = smard.fetch_series(
            254,
            "DE",
            "quarterhour",
            WEEK_START,
            WEEK_START + pd.Timedelta(hours=1),
            client=client,
            cache_dir=tmp_path,
        )

    assert len(prices) == 4
    assert len(requests) == 2


def test_client_errors_are_not_retried(tmp_path: Path) -> None:
    requests: list[str] = []
    with serve({}, requests) as client, pytest.raises(httpx.HTTPStatusError):
        smard.fetch_series(
            254,
            "FR",
            "quarterhour",
            WEEK_START,
            WEEK_START + pd.Timedelta(hours=1),
            client=client,
            cache_dir=tmp_path,
        )

    assert len(requests) == 1


def test_persistent_server_errors_raise(tmp_path: Path) -> None:
    requests: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request.url.path)
        return httpx.Response(503)

    with (
        httpx.Client(transport=httpx.MockTransport(handler)) as client,
        pytest.raises(httpx.HTTPStatusError),
    ):
        smard.fetch_series(
            254,
            "DE",
            "quarterhour",
            WEEK_START,
            WEEK_START + pd.Timedelta(hours=1),
            client=client,
            cache_dir=tmp_path,
        )

    assert len(requests) == len(smard.RETRY_DELAYS_SECONDS) + 1
