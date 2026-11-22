import os
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from eu_solar_wind_audit.sources import odre

pytestmark = pytest.mark.skipif(os.environ.get("RUN_NETWORK_TESTS") != "1", reason="network")


def test_one_day_of_each_dataset_matches_its_schema(tmp_path: Path) -> None:
    yesterday = datetime.now(UTC).date() - timedelta(days=1)
    with odre.make_client() as client:
        consolidated = odre.fetch_national_consolidated(
            client, date(2025, 6, 1), date(2025, 6, 1), cache_dir=tmp_path
        )
        realtime = odre.fetch_national_realtime(client, yesterday, yesterday, cache_dir=tmp_path)
        regional = odre.fetch_regional_consolidated(
            client, date(2025, 6, 1), date(2025, 6, 1), cache_dir=tmp_path
        )

    # A UTC day without clock change: 96 quarter-hours and the closing midnight.
    assert len(consolidated) == 97
    assert consolidated["solar_mw"].count() == 49
    # Bounds rather than a row count, so that the test also holds on clock-change days.
    midnight = pd.Timestamp(yesterday, tz="UTC")
    assert realtime.index[0] == midnight
    assert realtime.index[-1] == midnight + pd.Timedelta(days=1)
    assert realtime["wind_offshore_mw"].count() > 0
    assert len(regional) == 12 * 49
    assert regional["wind_offshore_mw"].count() > 0


def test_one_month_of_hourly_series_adds_up(tmp_path: Path) -> None:
    start, end = pd.Timestamp("2025-06-01", tz="UTC"), pd.Timestamp("2025-07-01", tz="UTC")
    with odre.make_client() as client:
        solar = odre.national_hourly(client, "solar", start, end, cache_dir=tmp_path)
        onshore = odre.national_hourly(client, "wind_onshore", start, end, cache_dir=tmp_path)
        regional = odre.regional_hourly(client, "solar", start, end, cache_dir=tmp_path)

    assert len(solar) == len(onshore) == 30 * 24
    assert list(regional) == list(odre.REGIONS)
    assert all(frame.index.equals(solar.index) for frame in regional.values())
    assert solar["complete"].all()
    assert onshore["complete"].all()
    assert set(solar["status"]) == {"definitive"}
    # National minus the sum of the regions stayed within 3 MWh in every hour of 2019 to
    # August 2026 (ODRÉ exports read on 2026-11-22): rounding of the published MW values.
    regions = pd.concat(regional.values()).groupby(level="time")["energy_mwh"].sum()
    assert (solar["energy_mwh"] - regions).abs().max() <= 3


def test_onshore_reference_before_offshore_output_is_the_sum_of_regional_wind(
    tmp_path: Path,
) -> None:
    # May 2022 in Paris, the last month before offshore output.
    start = pd.Timestamp("2022-05-01", tz="Europe/Paris").tz_convert("UTC")
    end = odre.FIRST_OFFSHORE_POWER
    with odre.make_client() as client:
        reference = odre.onshore_hourly(client, start, end, cache_dir=tmp_path)
        regional = odre.regional_hourly(client, "wind", start, end, cache_dir=tmp_path)

    regions = pd.concat(regional.values()).groupby(level="time")["energy_mwh"].sum(min_count=12)
    gap = reference["energy_mwh"] - regions
    assert gap.notna().all()
    # Values are published in whole MW: the national total and 12 regions differ by rounding
    # alone by at most 13 x 0.5 MW.
    assert gap.abs().max() <= 13 * 0.5


def test_onshore_reference_from_the_split_is_national_wind_without_offshore(
    tmp_path: Path,
) -> None:
    # January 2024 in Paris, the first month of the split.
    start = odre.WIND_SPLIT_START
    end = pd.Timestamp("2024-02-01", tz="Europe/Paris").tz_convert("UTC")
    with odre.make_client() as client:
        reference = odre.onshore_hourly(client, start, end, cache_dir=tmp_path)
        wind = odre.national_hourly(client, "wind", start, end, cache_dir=tmp_path)
        offshore = odre.national_hourly(client, "wind_offshore", start, end, cache_dir=tmp_path)

    gap = wind["energy_mwh"] - reference["energy_mwh"] - offshore["energy_mwh"]
    assert gap.notna().all()
    # The national total and the 12 regions of each sum: at most 25 x 0.5 MW of rounding.
    assert gap.abs().max() <= 25 * 0.5
