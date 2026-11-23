import os
from pathlib import Path

import pandas as pd
import pytest

from eu_solar_wind_audit import eligibility

pytestmark = pytest.mark.skipif(os.environ.get("RUN_NETWORK_TESTS") != "1", reason="network")


def test_reference_hours_of_2022(tmp_path: Path) -> None:
    start, end = pd.Timestamp("2022-01-01", tz="UTC"), pd.Timestamp("2023-01-01", tz="UTC")

    classes = eligibility.price_classes(start, end, cache_dir=tmp_path)

    assert classes.index.equals(pd.date_range(start, end, freq="h", inclusive="left"))
    # Counted on 2026-11-23 from the SMARD quarter-hour prices of 2022 (UTC).
    regimes = classes["regime"].value_counts()
    assert regimes["reference"] == 8715
    assert regimes["missing"] == 0
