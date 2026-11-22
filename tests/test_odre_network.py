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
    assert len(regional) == 12 * 30 * 24
    assert solar["complete"].all()
    assert onshore["complete"].all()
    assert set(solar["status"]) == {"definitive"}
    # National minus the sum of the regions stayed within 3 MWh in every hour of 2019 to
    # August 2026 (ODRÉ exports read on 2026-11-22): rounding of the published MW values.
    regions = regional.groupby(level="time")["energy_mwh"].sum()
    assert (solar["energy_mwh"] - regions).abs().max() <= 3
