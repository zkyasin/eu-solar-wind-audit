import os
from pathlib import Path

import pandas as pd
import pytest

from eu_solar_wind_audit.regimes import classify_hours
from eu_solar_wind_audit.sources import smard

pytestmark = pytest.mark.skipif(os.environ.get("RUN_NETWORK_TESTS") != "1", reason="network")


# Counts computed on 2026-11-22 from the SMARD quarter-hour day-ahead prices of 2025 (UTC).
@pytest.mark.parametrize(
    ("zone", "negative_hours", "zero_hours"),
    [("FR", 513, 219), ("DE-LU", 576, 78), ("BE", 519, 56), ("NL", 584, 66)],
)
def test_negative_hours_of_2025(
    zone: str, negative_hours: int, zero_hours: int, tmp_path: Path
) -> None:
    filter_id, region = smard.PRICES[zone]
    prices = smard.fetch_series(
        filter_id,
        region,
        "quarterhour",
        pd.Timestamp("2025-01-01", tz="UTC"),
        pd.Timestamp("2026-01-01", tz="UTC"),
        cache_dir=tmp_path,
    )

    hours = classify_hours(prices)

    assert len(hours) == 8760
    assert (hours["regime"] == "missing").sum() == 0
    assert (hours["regime"] == "negative").sum() == negative_hours
    assert (hours["mean_price"] == 0).sum() == zero_hours
