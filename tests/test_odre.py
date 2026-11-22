from datetime import date
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import pandera.errors
import pytest

from eu_solar_wind_audit.sources import odre

# Exports from https://odre.opendatasoft.com/api/explore/v2.1/catalog/datasets/<dataset>/exports/parquet
# with the select of each fetch function and where=odre.utc_days(first day, last day), retrieved
# 2026-11-22.
FIXTURES = Path(__file__).parent / "fixtures" / "odre"
NATIONAL_CONSOLIDATED = FIXTURES / "eco2mix-national-cons-def_2025-03-30.parquet"
NATIONAL_REALTIME = FIXTURES / "eco2mix-national-tr_2026-10-24_2026-10-25.parquet"
REGIONAL_CONSOLIDATED = FIXTURES / "eco2mix-regional-cons-def_2025-12-31.parquet"


def serving(content: bytes, requests: list[httpx.Request]) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, content=content)

    return httpx.Client(transport=httpx.MockTransport(handler))


def utc(stamp: str) -> pd.Timestamp:
    return pd.Timestamp(stamp, tz="UTC")


def values_at(frame: pd.DataFrame, stamp: str, *columns: str) -> list[float]:
    return list(frame.loc[frame.index == utc(stamp), list(columns)].to_numpy().ravel())


def test_utc_days_includes_midnight_after_the_last_day() -> None:
    assert odre.utc_days(date(2025, 3, 30), date(2025, 3, 31)) == (
        "date_heure >= date'2025-03-30T00:00:00+00:00'"
        " AND date_heure <= date'2025-04-01T00:00:00+00:00'"
    )


def test_utc_days_rejects_an_end_before_the_start() -> None:
    with pytest.raises(ValueError, match="before start"):
        odre.utc_days(date(2025, 3, 30), date(2025, 3, 29))


def test_national_consolidated_requests_the_days_and_fields(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []
    with serving(NATIONAL_CONSOLIDATED.read_bytes(), requests) as client:
        odre.fetch_national_consolidated(
            client, date(2025, 3, 30), date(2025, 3, 30), cache_dir=tmp_path
        )

    (request,) = requests
    assert request.url.path == (
        "/api/explore/v2.1/catalog/datasets/eco2mix-national-cons-def/exports/parquet"
    )
    assert request.url.params["select"] == "date_heure,nature,consommation,solaire,eolien"
    assert request.url.params["where"] == odre.utc_days(date(2025, 3, 30), date(2025, 3, 30))


def test_national_consolidated_keeps_published_values_on_a_spring_clock_change(
    tmp_path: Path,
) -> None:
    with serving(NATIONAL_CONSOLIDATED.read_bytes(), []) as client:
        frame = odre.fetch_national_consolidated(
            client, date(2025, 3, 30), date(2025, 3, 30), cache_dir=tmp_path
        )

    assert str(frame.index.dtype) == "datetime64[ns, UTC]"
    assert frame.index[0] == utc("2025-03-30 00:00")
    assert frame.index[-1] == utc("2025-03-31 00:00")
    # 96 quarter-hours plus the closing midnight, and the four local 02:xx rows that the
    # source maps onto the same UTC stamps as 03:xx.
    assert len(frame) == 101
    duplicated = frame.index[frame.index.duplicated()]
    assert list(duplicated) == [utc(f"2025-03-30 01:{m}") for m in ("00", "15", "30", "45")]
    # Published order is kept: the local 02:30 row comes before the local 03:30 row.
    assert values_at(frame, "2025-03-30 01:30", "wind_mw") == [7192.0, 7435.0]
    production = ("load_mw", "solar_mw", "wind_mw")
    assert values_at(frame, "2025-03-30 00:00", *production) == [49203.0, 0.0, 7036.0]
    # Half-hour averages on a quarter-hour grid: one row in two is empty.
    assert np.isnan(values_at(frame, "2025-03-30 00:15", *production)).all()
    assert set(frame["status"]) == {"definitive"}


def test_national_realtime_has_the_wind_split_and_no_repeated_autumn_hour(tmp_path: Path) -> None:
    with serving(NATIONAL_REALTIME.read_bytes(), []) as client:
        frame = odre.fetch_national_realtime(
            client, date(2026, 10, 24), date(2026, 10, 25), cache_dir=tmp_path
        )

    # Two days of quarter-hours plus the closing midnight, without the four quarter-hours
    # of 25 October from 00:00 to 00:45 UTC.
    assert len(frame) == 2 * 96 + 1 - 4
    assert frame.index.is_unique
    around_change = frame.loc[utc("2026-10-24 23:45") : utc("2026-10-25 01:00")].index
    assert list(around_change) == [utc("2026-10-24 23:45"), utc("2026-10-25 01:00")]
    wind = ("wind_mw", "wind_onshore_mw", "wind_offshore_mw")
    assert values_at(frame, "2026-10-24 23:30", *wind) == [19053.0, 16975.0, 2078.0]
    assert set(frame["status"]) == {"realtime"}


def test_regional_consolidated_reads_the_dash_placeholder_as_missing(tmp_path: Path) -> None:
    with serving(REGIONAL_CONSOLIDATED.read_bytes(), []) as client:
        frame = odre.fetch_regional_consolidated(
            client, date(2025, 12, 31), date(2025, 12, 31), cache_dir=tmp_path
        )

    assert frame["region_code"].nunique() == 12
    assert len(frame) == 12 * 49
    loire = frame[frame["region_code"] == "52"]
    assert np.isnan(values_at(loire, "2025-12-31 23:00", "wind_mw")).all()
    assert values_at(loire, "2025-12-31 23:30", "wind_mw") == [934.0]
    # The status follows the local date: 23:00 UTC is midnight on 1 January in Paris.
    assert set(loire.loc[: utc("2025-12-31 22:30"), "status"]) == {"definitive"}
    assert set(loire.loc[utc("2025-12-31 23:00") :, "status"]) == {"consolidated"}


def parquet_bytes(frame: pd.DataFrame, tmp_path: Path) -> bytes:
    path = tmp_path / "served.parquet"
    frame.to_parquet(path, index=False)
    return path.read_bytes()


def published_rows(nature: str, stamps: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date_heure": stamps,
            "nature": nature,
            "consommation": 50000,
            "solaire": 0,
            "eolien": 7000,
        }
    )


def test_unknown_status_fails_validation(tmp_path: Path) -> None:
    stamps = pd.date_range("2025-03-30", periods=2, freq="30min", tz="UTC")
    content = parquet_bytes(published_rows("Données provisoires", stamps), tmp_path)

    with serving(content, []) as client, pytest.raises(pandera.errors.SchemaError):
        odre.fetch_national_consolidated(
            client, date(2025, 3, 30), date(2025, 3, 30), cache_dir=tmp_path / "cache"
        )


def test_timestamps_without_time_zone_fail_validation(tmp_path: Path) -> None:
    stamps = pd.date_range("2025-03-30", periods=2, freq="30min")
    content = parquet_bytes(published_rows("Données définitives", stamps), tmp_path)

    with serving(content, []) as client, pytest.raises(pandera.errors.SchemaError):
        odre.fetch_national_consolidated(
            client, date(2025, 3, 30), date(2025, 3, 30), cache_dir=tmp_path / "cache"
        )


def test_negative_load_fails_validation(tmp_path: Path) -> None:
    stamps = pd.date_range("2025-03-30", periods=2, freq="30min", tz="UTC")
    rows = published_rows("Données définitives", stamps).assign(consommation=[50000, -1])
    content = parquet_bytes(rows, tmp_path)

    with serving(content, []) as client, pytest.raises(pandera.errors.SchemaError):
        odre.fetch_national_consolidated(
            client, date(2025, 3, 30), date(2025, 3, 30), cache_dir=tmp_path / "cache"
        )


def test_download_export_uses_the_cache(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []
    with serving(b"export", requests) as client:
        first = odre.download_export(client, "eco2mix-national-cons-def", cache_dir=tmp_path)
        second = odre.download_export(client, "eco2mix-national-cons-def", cache_dir=tmp_path)

    assert first == second
    assert first.parent == tmp_path / "eco2mix-national-cons-def"
    assert first.read_bytes() == b"export"
    assert len(requests) == 1


def test_download_export_downloads_a_different_query_again(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []
    with serving(b"export", requests) as client:
        all_rows = odre.download_export(client, "eco2mix-national-cons-def", cache_dir=tmp_path)
        one_day = odre.download_export(
            client,
            "eco2mix-national-cons-def",
            where=odre.utc_days(date(2025, 3, 30), date(2025, 3, 30)),
            cache_dir=tmp_path,
        )
        as_csv = odre.download_export(
            client, "eco2mix-national-cons-def", export_format="csv", cache_dir=tmp_path
        )

    assert len({all_rows, one_day, as_csv}) == 3
    assert len(requests) == 3
    assert requests[2].url.path.endswith("/eco2mix-national-cons-def/exports/csv")


def test_download_export_refresh_downloads_again(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []
    with serving(b"export", requests) as client:
        odre.download_export(client, "eco2mix-national-tr", cache_dir=tmp_path)
        odre.download_export(client, "eco2mix-national-tr", cache_dir=tmp_path, refresh=True)

    assert len(requests) == 2


def test_consolidated_fetch_is_cached_and_realtime_fetch_is_not(tmp_path: Path) -> None:
    consolidated: list[httpx.Request] = []
    with serving(NATIONAL_CONSOLIDATED.read_bytes(), consolidated) as client:
        for _ in range(2):
            odre.fetch_national_consolidated(
                client, date(2025, 3, 30), date(2025, 3, 30), cache_dir=tmp_path
            )

    realtime: list[httpx.Request] = []
    with serving(NATIONAL_REALTIME.read_bytes(), realtime) as client:
        for _ in range(2):
            odre.fetch_national_realtime(
                client, date(2026, 10, 24), date(2026, 10, 25), cache_dir=tmp_path
            )

    assert len(consolidated) == 1
    assert len(realtime) == 2


def test_failed_download_leaves_no_file(tmp_path: Path) -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(503))

    with httpx.Client(transport=transport) as client, pytest.raises(httpx.HTTPStatusError):
        odre.download_export(client, "eco2mix-national-tr", cache_dir=tmp_path)

    assert list((tmp_path / "eco2mix-national-tr").iterdir()) == []
