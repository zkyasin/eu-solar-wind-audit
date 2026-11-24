import hashlib
import json
import os
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
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
# Same query for 2012-12-31, retrieved 2026-11-22: the first stamp of the regional dataset.
REGIONAL_FIRST_DAY = FIXTURES / "eco2mix-regional-cons-def_2012-12-31.parquet"
# Eleven rows of the export of odre.fetch_register, retrieved 2026-11-24: Corsica, La Réunion,
# mainland wind and solar on the grids of RTE and Enedis, and a row without a region.
REGISTER = FIXTURES / f"{odre.REGISTER}_cut.parquet"


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


@pytest.mark.parametrize(
    ("fetch", "fixture", "dataset", "fields"),
    [
        (
            odre.fetch_national_consolidated,
            NATIONAL_CONSOLIDATED,
            "eco2mix-national-cons-def",
            "date_heure,nature,consommation,solaire,eolien",
        ),
        (
            odre.fetch_national_realtime,
            NATIONAL_REALTIME,
            "eco2mix-national-tr",
            "date_heure,nature,consommation,solaire,eolien,eolien_terrestre,eolien_offshore",
        ),
        (
            odre.fetch_regional_consolidated,
            REGIONAL_CONSOLIDATED,
            "eco2mix-regional-cons-def",
            "date_heure,code_insee_region,libelle_region,nature,consommation,solaire,eolien,"
            "eolien_terrestre,eolien_offshore",
        ),
    ],
)
def test_fetch_requests_the_dataset_days_and_fields(
    fetch: Callable[..., pd.DataFrame], fixture: Path, dataset: str, fields: str, tmp_path: Path
) -> None:
    requests: list[httpx.Request] = []
    with serving(fixture.read_bytes(), requests) as client:
        fetch(client, date(2025, 3, 30), date(2025, 3, 30), cache_dir=tmp_path)

    (request,) = requests
    assert request.url.path == f"/api/explore/v2.1/catalog/datasets/{dataset}/exports/parquet"
    assert request.url.params["select"] == fields
    assert request.url.params["where"] == (
        "date_heure >= date'2025-03-30T00:00:00+00:00'"
        " AND date_heure <= date'2025-03-31T00:00:00+00:00'"
    )


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
    # The source gives no local label for these rows; they keep the export order.
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


def test_regional_consolidated_reads_the_not_available_placeholder_as_missing(
    tmp_path: Path,
) -> None:
    with serving(REGIONAL_FIRST_DAY.read_bytes(), []) as client:
        frame = odre.fetch_regional_consolidated(
            client, date(2012, 12, 31), date(2012, 12, 31), cache_dir=tmp_path
        )

    assert len(frame) == 12 * 3
    assert np.isnan(values_at(frame, "2012-12-31 23:00", "wind_mw")).all()
    loire = frame[frame["region_code"] == "52"]
    assert values_at(loire, "2012-12-31 23:30", "wind_mw") == [182.0]


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


def test_unknown_region_fails_validation(tmp_path: Path) -> None:
    published = pd.read_parquet(REGIONAL_CONSOLIDATED)
    published.loc[published["code_insee_region"] == "53", "code_insee_region"] = "94"
    content = parquet_bytes(published, tmp_path)

    with serving(content, []) as client, pytest.raises(pandera.errors.SchemaError):
        odre.fetch_regional_consolidated(
            client, date(2025, 12, 31), date(2025, 12, 31), cache_dir=tmp_path / "cache"
        )


def test_negative_load_fails_validation(tmp_path: Path) -> None:
    stamps = pd.date_range("2025-03-30", periods=2, freq="30min", tz="UTC")
    rows = published_rows("Données définitives", stamps).assign(consommation=[50000, -1])
    content = parquet_bytes(rows, tmp_path)

    with serving(content, []) as client, pytest.raises(pandera.errors.SchemaError):
        odre.fetch_national_consolidated(
            client, date(2025, 3, 30), date(2025, 3, 30), cache_dir=tmp_path / "cache"
        )


def test_negative_production_is_kept(tmp_path: Path) -> None:
    stamps = pd.date_range("2025-03-30", periods=2, freq="30min", tz="UTC")
    rows = published_rows("Données définitives", stamps).assign(solaire=[0, -23], eolien=[-2, 0])
    content = parquet_bytes(rows, tmp_path)

    with serving(content, []) as client:
        frame = odre.fetch_national_consolidated(
            client, date(2025, 3, 30), date(2025, 3, 30), cache_dir=tmp_path / "cache"
        )

    assert list(frame["solar_mw"]) == [0.0, -23.0]
    assert list(frame["wind_mw"]) == [-2.0, 0.0]


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


# An export cut at 23:00 stops at 22:45 UTC, as the last consolidated month does when it ends at
# midnight in Paris in winter: only the last hour of the day misses its rows.
@pytest.mark.parametrize("cut", ["2025-03-30 12:00", "2025-03-30 23:00"])
def test_a_short_consolidated_export_is_downloaded_again_once_12_hours_old(
    cut: str, tmp_path: Path
) -> None:
    full = pd.read_parquet(NATIONAL_CONSOLIDATED)
    short_path = tmp_path / "short.parquet"
    full[full["date_heure"] < utc(cut)].to_parquet(short_path, index=False)
    exports = [short_path.read_bytes(), NATIONAL_CONSOLIDATED.read_bytes()]
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, content=exports[len(requests) - 1])

    cache_dir = tmp_path / "cache"
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:

        def fetch() -> pd.DataFrame:
            return odre.fetch_national_consolidated(
                client, date(2025, 3, 30), date(2025, 3, 30), cache_dir=cache_dir
            )

        downloaded, cached = fetch(), fetch()
        (path,) = cache_dir.rglob("*.parquet")
        retrieved = (datetime.now(UTC) - timedelta(hours=13)).timestamp()
        os.utime(path, (retrieved, retrieved))
        refreshed = fetch()

    assert downloaded.index.max() < utc("2025-03-30 23:00")
    assert cached.index.equals(downloaded.index)
    assert refreshed.index.max() == utc("2025-03-31 00:00")
    assert len(requests) == 2


def test_a_full_consolidated_export_is_kept_however_old(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []
    cache_dir = tmp_path / "cache"
    with serving(NATIONAL_CONSOLIDATED.read_bytes(), requests) as client:
        for _ in range(2):
            odre.fetch_national_consolidated(
                client, date(2025, 3, 30), date(2025, 3, 30), cache_dir=cache_dir
            )
            (path,) = cache_dir.rglob("*.parquet")
            os.utime(path, (0, 0))

    assert len(requests) == 1


def test_failed_download_leaves_no_file(tmp_path: Path) -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(503))

    with httpx.Client(transport=transport) as client, pytest.raises(httpx.HTTPStatusError):
        odre.download_export(client, "eco2mix-national-tr", cache_dir=tmp_path)

    assert list((tmp_path / "eco2mix-national-tr").iterdir()) == []


def test_snapshot_realtime_saves_the_full_export_and_its_checksum(tmp_path: Path) -> None:
    content = NATIONAL_REALTIME.read_bytes()
    requests: list[httpx.Request] = []
    before = datetime.now(UTC).replace(microsecond=0)
    with serving(content, requests) as client:
        snapshot = odre.snapshot_realtime(client, tmp_path / "vintages")
    after = datetime.now(UTC)

    (request,) = requests
    assert request.url.path.endswith("/eco2mix-national-tr/exports/parquet")
    assert not request.url.params
    assert snapshot.path.read_bytes() == content
    assert snapshot.sha256 == hashlib.sha256(content).hexdigest()
    assert before <= snapshot.retrieved_at <= after
    assert snapshot.path.name == (
        f"eco2mix-national-tr_{snapshot.retrieved_at:%Y%m%dT%H%M%SZ}.parquet"
    )
    metadata = json.loads(snapshot.path.with_suffix(".json").read_text())
    assert metadata["sha256"] == snapshot.sha256
    assert datetime.fromisoformat(metadata["retrieved_at"]) == snapshot.retrieved_at
    assert metadata["dataset"] == "eco2mix-national-tr"


def test_fetch_register_asks_for_the_wind_and_solar_rows(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []
    with serving(REGISTER.read_bytes(), requests) as client:
        register = odre.fetch_register(client, cache_dir=tmp_path)

    (request,) = requests
    assert request.url.path.endswith(f"/{odre.REGISTER}/exports/parquet")
    assert request.url.params["select"] == "coderegion,filiere,codegestionnaire,puismaxinstallee"
    assert request.url.params["where"] == "filiere in ('Solaire','Eolien')"
    assert register.columns.tolist() == ["region_code", "sector", "grid_operator", "capacity_kw"]


def test_transmission_share_is_that_of_the_rte_grid_in_mainland_france(tmp_path: Path) -> None:
    with serving(REGISTER.read_bytes(), []) as client:
        register = odre.fetch_register(client, cache_dir=tmp_path)

    # Wind: 68,000 and 10,800 kW on the grid of RTE, 11,750 and 10,000 kW on that of Enedis; the
    # Corsican farm does not count.
    assert odre.transmission_share(register, "Eolien") == pytest.approx(78_800 / 100_550)
    # Solar: 9,749 kW on the grid of RTE, 4,455 kW on that of Enedis and 309.11 kW without a region.
    assert odre.transmission_share(register, "Solaire") == pytest.approx(9_749 / 14_513.11)
