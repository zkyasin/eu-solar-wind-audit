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


# The week of the 2025 autumn clock change starts on Monday 00:00 Berlin time and lasts 169 hours.
AUTUMN_WEEK = pd.date_range("2025-10-19 22:00", periods=169, freq="h", tz="UTC")


def autumn_files(values: list[float | None]) -> dict[str, bytes]:
    ms = [int(stamp.timestamp() * 1000) for stamp in AUTUMN_WEEK]
    next_week = ms[-1] + 3_600_000
    return {
        "index_hour.json": json.dumps({"timestamps": [ms[0], next_week]}).encode(),
        f"4068_DE_hour_{ms[0]}.json": json.dumps(
            {"series": list(zip(ms, values, strict=True))}
        ).encode(),
        f"4068_DE_hour_{next_week}.json": b'{"series": []}',
    }


def test_autumn_week_keeps_its_extra_hour_and_nulls_become_nan(tmp_path: Path) -> None:
    values: list[float | None] = [1.0] * 169
    values[100] = None
    with serve(autumn_files(values), []) as client:
        solar = smard.fetch_series(
            smard.SOLAR,
            "DE",
            "hour",
            AUTUMN_WEEK[0],
            AUTUMN_WEEK[0] + pd.Timedelta(days=8),
            client=client,
            cache_dir=tmp_path,
        )

    assert solar.index.equals(AUTUMN_WEEK)
    assert solar.isna().sum() == 1
    assert pd.isna(solar.iloc[100])


def test_the_extra_autumn_hour_is_read_from_its_own_week(tmp_path: Path) -> None:
    # A week assumed to last 7 x 24 hours would skip this file and return nothing.
    extra_hour = AUTUMN_WEEK[-1]
    requests: list[str] = []
    with serve(autumn_files([1.0] * 169), requests) as client:
        solar = smard.fetch_series(
            smard.SOLAR,
            "DE",
            "hour",
            extra_hour,
            extra_hour + pd.Timedelta(hours=1),
            client=client,
            cache_dir=tmp_path,
        )

    assert requests[1:] == ["/app/chart_data/4068/DE/4068_DE_hour_1760911200000.json"]
    assert solar.index.to_list() == [extra_hour]


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


def test_a_recently_closed_week_is_not_cached(tmp_path: Path) -> None:
    week_end = pd.Timestamp.now(tz="UTC").floor("h") - smard.SETTLED_AFTER + pd.Timedelta(hours=1)
    week_start = week_end - pd.Timedelta(days=7)
    start_ms, end_ms = (int(stamp.timestamp() * 1000) for stamp in (week_start, week_end))
    files = {
        "index_hour.json": json.dumps({"timestamps": [start_ms, end_ms]}).encode(),
        f"410_DE_hour_{start_ms}.json": json.dumps({"series": [[start_ms, 51234.5]]}).encode(),
    }
    with serve(files, []) as client:
        smard.fetch_series(
            smard.LOAD,
            "DE",
            "hour",
            week_start,
            week_start + pd.Timedelta(hours=1),
            client=client,
            cache_dir=tmp_path,
        )

    assert list(tmp_path.iterdir()) == []


def test_a_settled_week_with_nulls_is_downloaded_again(tmp_path: Path) -> None:
    values: list[float | None] = [1.0] * 169
    values[100] = None
    requests: list[str] = []
    for _ in range(2):
        with serve(autumn_files(values), requests) as client:
            smard.fetch_series(
                smard.SOLAR,
                "DE",
                "hour",
                AUTUMN_WEEK[0],
                AUTUMN_WEEK[1],
                client=client,
                cache_dir=tmp_path,
            )

    assert requests.count("/app/chart_data/4068/DE/4068_DE_hour_1760911200000.json") == 2
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


# Fixtures: https://www.smard.de/app/chart_data/4068/DE/index_quarterhour.json and
# .../4068/DE/4068_DE_quarterhour_<ms>.json for the weeks of the 2025 clock changes (German solar
# feed-in), retrieved 2026-11-22.
SPRING_WEEK_MS = 1742770800000
AUTUMN_WEEK_MS = 1760911200000


def solar_files(region: str = "DE", filter_id: int = smard.SOLAR) -> dict[str, bytes]:
    files = {"index_quarterhour.json": (FIXTURES / "4068_DE_index_quarterhour.json").read_bytes()}
    for week_ms in (SPRING_WEEK_MS, AUTUMN_WEEK_MS):
        content = (FIXTURES / f"4068_DE_quarterhour_{week_ms}.json").read_bytes()
        files[f"{filter_id}_{region}_quarterhour_{week_ms}.json"] = content
    return files


def published_quarter_hours(week_ms: int) -> pd.Series:
    rows = json.loads((FIXTURES / f"4068_DE_quarterhour_{week_ms}.json").read_bytes())["series"]
    index = pd.to_datetime([ms for ms, _ in rows], unit="ms", utc=True)
    return pd.Series([value for _, value in rows], index=index, dtype=float)


@pytest.mark.parametrize(
    ("week_ms", "local_day", "hours"),
    [(SPRING_WEEK_MS, "2025-03-30", 23), (AUTUMN_WEEK_MS, "2025-10-26", 25)],
)
def test_clock_change_days_have_all_their_utc_hours(
    week_ms: int, local_day: str, hours: int, tmp_path: Path
) -> None:
    day = pd.Timestamp(local_day, tz="Europe/Berlin")
    start, end = day.tz_convert("UTC"), (day + pd.DateOffset(days=1)).tz_convert("UTC")
    with serve(solar_files(), []) as client:
        solar = smard.fetch_hourly_feed_in(
            "DE", "solar", start, end, client=client, cache_dir=tmp_path
        )

    assert len(solar) == hours
    assert solar["complete"].all()
    published = published_quarter_hours(week_ms)
    in_day = published[(published.index >= start) & (published.index < end)]
    assert len(in_day) == 4 * hours
    assert solar["energy_mwh"].sum() == pytest.approx(in_day.sum())


def test_an_hour_sums_the_four_quarter_hours_that_start_in_it(tmp_path: Path) -> None:
    hour = pd.Timestamp("2025-03-30 11:00", tz="UTC")
    with serve(solar_files(), []) as client:
        solar = smard.fetch_hourly_feed_in(
            "DE", "solar", hour, hour + pd.Timedelta(hours=1), client=client, cache_dir=tmp_path
        )

    quarters = published_quarter_hours(SPRING_WEEK_MS)
    expected = quarters[hour : hour + pd.Timedelta(minutes=45)]
    assert len(expected) == 4
    assert solar["energy_mwh"].item() == pytest.approx(expected.sum())


def test_feed_in_is_read_from_the_region_and_technology_files(tmp_path: Path) -> None:
    requests: list[str] = []
    files = solar_files("TenneT", smard.WIND_ONSHORE)
    start = pd.Timestamp("2025-03-25", tz="UTC")
    with serve(files, requests) as client:
        wind = smard.fetch_hourly_feed_in(
            "TenneT",
            "wind_onshore",
            start,
            start + pd.Timedelta(hours=2),
            client=client,
            cache_dir=tmp_path,
        )

    assert requests == [
        "/app/chart_data/4067/TenneT/index_quarterhour.json",
        f"/app/chart_data/4067/TenneT/4067_TenneT_quarterhour_{SPRING_WEEK_MS}.json",
    ]
    assert wind["complete"].tolist() == [True, True]


def test_a_missing_quarter_hour_leaves_its_hour_incomplete(tmp_path: Path) -> None:
    files = solar_files()
    rows = json.loads(files[f"4068_DE_quarterhour_{SPRING_WEEK_MS}.json"])
    rows["series"][1][1] = None
    files[f"4068_DE_quarterhour_{SPRING_WEEK_MS}.json"] = json.dumps(rows).encode()
    start = pd.Timestamp(SPRING_WEEK_MS, unit="ms", tz="UTC")

    with serve(files, []) as client:
        solar = smard.fetch_hourly_feed_in(
            "DE", "solar", start, start + pd.Timedelta(hours=2), client=client, cache_dir=tmp_path
        )

    assert solar["complete"].tolist() == [False, True]
    assert solar["energy_mwh"].isna().tolist() == [True, False]


def test_hours_before_the_first_published_week_are_incomplete(tmp_path: Path) -> None:
    start = pd.Timestamp("2014-12-01", tz="UTC")
    with serve(solar_files(), []) as client:
        solar = smard.fetch_hourly_feed_in(
            "DE", "solar", start, start + pd.Timedelta(hours=2), client=client, cache_dir=tmp_path
        )

    assert solar["complete"].tolist() == [False, False]


@pytest.mark.parametrize("region", ["Amprion", "TransnetBW"])
def test_offshore_wind_is_refused_where_it_is_not_published(region: str, tmp_path: Path) -> None:
    requests: list[str] = []
    with serve({}, requests) as client, pytest.raises(ValueError, match=region):
        smard.fetch_hourly_feed_in(
            region,
            "wind_offshore",
            WEEK_START,
            WEEK_START + pd.Timedelta(hours=1),
            client=client,
            cache_dir=tmp_path,
        )

    assert requests == []
