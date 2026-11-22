from datetime import date
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import pytest

from eu_solar_wind_audit.sources import odre

# Exports from https://odre.opendatasoft.com/api/explore/v2.1/catalog/datasets/<dataset>/exports/parquet
# with the select of each fetch function and where=odre.utc_days(first day, last day), retrieved
# 2026-11-22. The national consolidated export for 2026-10-24 to 2026-10-25 has no rows.
FIXTURES = Path(__file__).parent / "fixtures" / "odre"
SPRING_DEFINITIVE = FIXTURES / "eco2mix-national-cons-def_2025-03-30.parquet"
AUTUMN_REALTIME = FIXTURES / "eco2mix-national-tr_2026-10-24_2026-10-25.parquet"
AUTUMN_CONSOLIDATED = FIXTURES / "eco2mix-national-cons-def_2026-10-24_2026-10-25.parquet"
SWITCH_CONSOLIDATED = FIXTURES / "eco2mix-national-cons-def_2026-08-31_2026-09-01.parquet"
SWITCH_REALTIME = FIXTURES / "eco2mix-national-tr_2026-08-31_2026-09-01.parquet"
SWITCH_REGIONAL = FIXTURES / "eco2mix-regional-cons-def_2026-08-31_2026-09-01.parquet"
SPLIT_START_REGIONAL = FIXTURES / "eco2mix-regional-cons-def_2023-12-31.parquet"
YEAR_END_REGIONAL = FIXTURES / "eco2mix-regional-cons-def_2025-12-31.parquet"


def serving(exports: dict[str, bytes], requests: list[httpx.Request]) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        dataset = request.url.path.split("/")[-3]
        return httpx.Response(200, content=exports[dataset])

    return httpx.Client(transport=httpx.MockTransport(handler))


def national(
    series: odre.NationalSeries,
    start: date,
    end: date,
    exports: dict[str, Path],
    tmp_path: Path,
    requests: list[httpx.Request] | None = None,
) -> pd.DataFrame:
    content = {dataset: path.read_bytes() for dataset, path in exports.items()}
    with serving(content, [] if requests is None else requests) as client:
        return odre.national_hourly(client, series, start, end, cache_dir=tmp_path)


def regional(
    series: odre.RegionalSeries, day: date, content: bytes, tmp_path: Path
) -> pd.DataFrame:
    with serving({odre.REGIONAL_CONSOLIDATED: content}, []) as client:
        return odre.regional_hourly(client, series, day, day, cache_dir=tmp_path)


def utc(*stamps: str) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(stamps, tz="UTC")


def hours(start: str, end: str) -> pd.DatetimeIndex:
    return pd.date_range(start, end, freq="h", tz="UTC", name="time")


SWITCH = {
    odre.NATIONAL_CONSOLIDATED: SWITCH_CONSOLIDATED,
    odre.NATIONAL_REALTIME: SWITCH_REALTIME,
    odre.REGIONAL_CONSOLIDATED: SWITCH_REGIONAL,
}


def test_definitive_hours_average_the_half_hours_ending_in_them(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []
    wind = national(
        "wind",
        date(2025, 3, 30),
        date(2025, 3, 30),
        {odre.NATIONAL_CONSOLIDATED: SPRING_DEFINITIVE},
        tmp_path,
        requests,
    )

    assert wind.index.equals(hours("2025-03-30 00:00", "2025-03-30 23:00"))
    # Half-hours ending 02:30 (7438 MW) and 03:00 (7250 MW).
    assert wind.loc[utc("2025-03-30 02:00"), "energy_mwh"].item() == 7344.0
    # The half-hours ending 01:00 and 01:30 are published twice on the clock-change day.
    assert wind.index[~wind["complete"]].equals(utc("2025-03-30 00:00", "2025-03-30 01:00"))
    assert wind["energy_mwh"].isna().equals(~wind["complete"])
    assert set(wind["status"]) == {"definitive"}
    # The consolidated rows cover the last hour, so real time is not requested.
    assert {request.url.path.split("/")[-3] for request in requests} == {odre.NATIONAL_CONSOLIDATED}


def test_realtime_hours_are_trapezoids_and_the_autumn_hour_is_incomplete(
    tmp_path: Path,
) -> None:
    exports = {
        odre.NATIONAL_CONSOLIDATED: AUTUMN_CONSOLIDATED,
        odre.NATIONAL_REALTIME: AUTUMN_REALTIME,
    }

    wind = national("wind", date(2026, 10, 24), date(2026, 10, 25), exports, tmp_path)

    assert wind.index.equals(hours("2026-10-24 00:00", "2026-10-25 23:00"))
    # Points at 01:00, 01:15, 01:30, 01:45 and 02:00: 18931, 19029, 19056, 18922, 19388 MW.
    expected = (18931 / 2 + 19029 + 19056 + 18922 + 19388 / 2) / 4
    assert wind.loc[utc("2026-10-25 01:00"), "energy_mwh"].item() == expected
    # The source has no points from 00:00 to 00:45 UTC on the day of the change.
    assert wind.index[~wind["complete"]].equals(utc("2026-10-24 23:00", "2026-10-25 00:00"))
    assert set(wind["status"]) == {"realtime"}


@pytest.mark.parametrize("series", ["load", "solar", "wind"])
def test_switch_to_realtime_keeps_every_hour_once(
    series: odre.NationalSeries, tmp_path: Path
) -> None:
    requests: list[httpx.Request] = []
    frame = national(series, date(2026, 8, 31), date(2026, 9, 1), SWITCH, tmp_path, requests)

    assert frame.index.equals(hours("2026-08-31 00:00", "2026-09-01 23:00"))
    switch = pd.Timestamp("2026-08-31 22:00", tz="UTC")
    assert set(frame.loc[frame.index < switch, "status"]) == {"consolidated"}
    assert set(frame.loc[frame.index >= switch, "status"]) == {"realtime"}
    # The half-hour ending at 22:00 is published neither as a consolidated mean nor with the
    # real-time points it would need.
    assert frame.index[~frame["complete"]].equals(utc("2026-08-31 21:00"))
    realtime = [r for r in requests if r.url.path.split("/")[-3] == odre.NATIONAL_REALTIME]
    assert realtime[0].url.params["where"].startswith("date_heure >= date'2026-08-31T00:00")


def test_switch_takes_each_hour_from_its_own_source(tmp_path: Path) -> None:
    load = national("load", date(2026, 8, 31), date(2026, 9, 1), SWITCH, tmp_path)

    # Consolidated half-hours ending 20:30 (44606 MW) and 21:00 (45070 MW).
    assert load.loc[utc("2026-08-31 20:00"), "energy_mwh"].item() == 44838.0
    # Real-time points from 22:00 to 23:00: 42233, 42256, 41091, 39517, 38939 MW.
    expected = (42233 / 2 + 42256 + 41091 + 39517 + 38939 / 2) / 4
    assert load.loc[utc("2026-08-31 22:00"), "energy_mwh"].item() == expected


def test_national_onshore_and_offshore_before_realtime_sum_the_regions(tmp_path: Path) -> None:
    exports = {odre.REGIONAL_CONSOLIDATED: SPLIT_START_REGIONAL}

    onshore = national("wind_onshore", date(2023, 12, 31), date(2023, 12, 31), exports, tmp_path)
    offshore = national("wind_offshore", date(2023, 12, 31), date(2023, 12, 31), exports, tmp_path)

    # Sums of the 12 regions for the half-hours ending 23:30 and 00:00.
    assert onshore.loc[utc("2023-12-31 23:00"), "energy_mwh"].item() == (14888 + 14827) / 2
    assert offshore.loc[utc("2023-12-31 23:00"), "energy_mwh"].item() == (579 + 581) / 2
    for frame in (onshore, offshore):
        assert frame.index[frame["complete"]].equals(utc("2023-12-31 23:00"))
        assert set(frame["status"]) == {"definitive"}


def test_national_onshore_and_offshore_switch_to_the_realtime_split(tmp_path: Path) -> None:
    onshore = national("wind_onshore", date(2026, 8, 31), date(2026, 9, 1), SWITCH, tmp_path)
    offshore = national("wind_offshore", date(2026, 8, 31), date(2026, 9, 1), SWITCH, tmp_path)
    wind = national("wind", date(2026, 8, 31), date(2026, 9, 1), SWITCH, tmp_path)

    # Real-time onshore points from 22:00 to 23:00: 2913, 2766, 2686, 2561, 2412 MW.
    expected = (2913 / 2 + 2766 + 2686 + 2561 + 2412 / 2) / 4
    assert onshore.loc[utc("2026-08-31 22:00"), "energy_mwh"].item() == expected
    assert set(onshore.loc[pd.Timestamp("2026-08-31 22:00", tz="UTC") :, "status"]) == {"realtime"}
    assert onshore.index[~onshore["complete"]].equals(utc("2026-08-31 21:00"))
    split = onshore["energy_mwh"] + offshore["energy_mwh"]
    # Values are published in whole MW: 12 regions and the national total differ by rounding
    # alone by at most 13 x 0.5 MW.
    assert (split - wind["energy_mwh"]).abs().max() <= 13 * 0.5


def test_national_sum_takes_the_least_final_status_of_the_regions(tmp_path: Path) -> None:
    published = pd.read_parquet(SPLIT_START_REGIONAL)
    brittany = published["code_insee_region"] == "53"
    published.loc[brittany, "nature"] = "Données consolidées"
    path = tmp_path / "export.parquet"
    published.to_parquet(path, index=False)

    onshore = national(
        "wind_onshore",
        date(2023, 12, 31),
        date(2023, 12, 31),
        {odre.REGIONAL_CONSOLIDATED: path},
        tmp_path / "cache",
    )

    assert set(onshore["status"]) == {"consolidated"}


def test_a_missing_region_makes_the_national_sum_incomplete(tmp_path: Path) -> None:
    published = pd.read_parquet(SPLIT_START_REGIONAL)
    without_brittany = published[published["code_insee_region"] != "53"]
    path = tmp_path / "export.parquet"
    without_brittany.to_parquet(path, index=False)

    offshore = national(
        "wind_offshore",
        date(2023, 12, 31),
        date(2023, 12, 31),
        {odre.REGIONAL_CONSOLIDATED: path},
        tmp_path / "cache",
    )

    assert not offshore["complete"].any()
    assert offshore["energy_mwh"].isna().all()


def test_regional_split_is_missing_before_2024(tmp_path: Path) -> None:
    content = SPLIT_START_REGIONAL.read_bytes()

    wind = regional("wind", date(2023, 12, 31), content, tmp_path)
    offshore = regional("wind_offshore", date(2023, 12, 31), content, tmp_path)

    loire = offshore[offshore["region_code"] == "52"]
    # The source fills the split with zeros until 31 December 2023 local time.
    assert loire.index[loire["complete"]].equals(utc("2023-12-31 23:00"))
    assert loire.loc[utc("2023-12-31 23:00"), "energy_mwh"].item() == 242.0
    assert wind["complete"].all()


def test_regional_frame_has_every_region_and_hour(tmp_path: Path) -> None:
    solar = regional("solar", date(2025, 12, 31), YEAR_END_REGIONAL.read_bytes(), tmp_path)

    assert len(solar) == 12 * 24
    assert solar.groupby("region_code").size().to_dict() == dict.fromkeys(odre.REGIONS, 24)
    assert dict(zip(solar["region_code"], solar["region"], strict=True)) == odre.REGIONS
    for _, frame in solar.groupby("region_code"):
        assert frame.index.equals(hours("2025-12-31 00:00", "2025-12-31 23:00"))


def test_hour_status_is_the_least_final_of_its_half_hours(tmp_path: Path) -> None:
    wind = regional("wind", date(2025, 12, 31), YEAR_END_REGIONAL.read_bytes(), tmp_path)

    loire = wind[wind["region_code"] == "52"]
    # The half-hour ending 23:00 UTC is midnight in Paris: the first consolidated row.
    assert loire.loc[utc("2025-12-31 21:00"), "status"].item() == "definitive"
    assert loire.loc[utc("2025-12-31 22:00"), "status"].item() == "consolidated"
    assert loire.loc[utc("2025-12-31 23:00"), "status"].item() == "consolidated"
    # That row is empty, so the hour it closes is incomplete.
    assert loire.index[~loire["complete"]].equals(utc("2025-12-31 22:00"))
    assert np.isnan(loire.loc[utc("2025-12-31 22:00"), "energy_mwh"].item())


def test_hours_without_published_values_have_no_status(tmp_path: Path) -> None:
    published = pd.read_parquet(SWITCH_REALTIME)
    later = published[published["date_heure"] >= pd.Timestamp("2026-09-01", tz="UTC")]
    path = tmp_path / "export.parquet"
    later.to_parquet(path, index=False)
    exports = {odre.NATIONAL_CONSOLIDATED: SWITCH_CONSOLIDATED, odre.NATIONAL_REALTIME: path}

    solar = national("solar", date(2026, 8, 31), date(2026, 9, 1), exports, tmp_path / "cache")

    gap = solar.loc[utc("2026-08-31 22:00", "2026-08-31 23:00")]
    assert gap["status"].isna().all()
    assert not gap["complete"].any()
    assert solar.loc[utc("2026-09-01 00:00"), "status"].item() == "realtime"


def test_end_before_start_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="before start"):
        national("solar", date(2025, 3, 30), date(2025, 3, 29), {}, tmp_path)
