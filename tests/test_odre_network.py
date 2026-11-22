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
