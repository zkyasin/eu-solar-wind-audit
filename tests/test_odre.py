from datetime import date
from pathlib import Path

import httpx
import pytest

from eu_solar_wind_audit.sources import odre


def serving(content: bytes, requests: list[httpx.Request]) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, content=content)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_utc_days_includes_midnight_after_the_last_day() -> None:
    assert odre.utc_days(date(2025, 3, 30), date(2025, 3, 31)) == (
        "date_heure >= date'2025-03-30T00:00:00+00:00'"
        " AND date_heure <= date'2025-04-01T00:00:00+00:00'"
    )


def test_utc_days_rejects_an_end_before_the_start() -> None:
    with pytest.raises(ValueError, match="before start"):
        odre.utc_days(date(2025, 3, 30), date(2025, 3, 29))


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


def test_failed_download_leaves_no_file(tmp_path: Path) -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(503))

    with httpx.Client(transport=transport) as client, pytest.raises(httpx.HTTPStatusError):
        odre.download_export(client, "eco2mix-national-tr", cache_dir=tmp_path)

    assert list((tmp_path / "eco2mix-national-tr").iterdir()) == []
