import os
from pathlib import Path

import pandas as pd
import pytest

from eu_solar_wind_audit.sources import energinet

pytestmark = pytest.mark.skipif(os.environ.get("RUN_NETWORK_TESTS") != "1", reason="network")

WEEK = (pd.Timestamp("2024-06-03", tz="UTC"), pd.Timestamp("2024-06-10", tz="UTC"))


def test_a_week_of_danish_solar_in_both_price_areas(tmp_path: Path) -> None:
    solar = {
        area: energinet.fetch_hourly_settlement(area, "solar", *WEEK, cache_dir=tmp_path)
        for area in ("DK1", "DK2", "DK")
    }

    assert solar["DK"]["complete"].all()
    both = solar["DK1"]["energy_mwh"] + solar["DK2"]["energy_mwh"]
    assert solar["DK"]["energy_mwh"].to_numpy() == pytest.approx(both.to_numpy())
    assert solar["DK"]["energy_mwh"].max() > 100


def test_the_capacity_of_every_municipality(tmp_path: Path) -> None:
    capacity = energinet.fetch_capacity(cache_dir=tmp_path)

    latest = capacity[capacity["month_start"] == capacity["month_start"].max()]
    assert set(latest["price_area"].dropna()) == {"DK1", "DK2"}
    assert latest["solar_mw"].sum() > 1000
