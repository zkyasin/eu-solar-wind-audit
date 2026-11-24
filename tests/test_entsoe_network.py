import os
from pathlib import Path

import pandas as pd
import pytest

from eu_solar_wind_audit.sources import entsoe

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_NETWORK_TESTS") != "1" or not os.environ.get(entsoe.TOKEN_VARIABLE),
    reason="network",
)

DAY = (pd.Timestamp("2024-06-01", tz="UTC"), pd.Timestamp("2024-06-02", tz="UTC"))


def test_one_day_of_dutch_solar(tmp_path: Path) -> None:
    hourly = entsoe.fetch_hourly_generation("NL", "solar", *DAY, cache_dir=tmp_path)

    assert hourly["complete"].all()
    assert 0 < hourly["energy_mwh"].max() < 1000


def test_one_day_of_dutch_load(tmp_path: Path) -> None:
    hourly = entsoe.fetch_hourly_load("NL", *DAY, cache_dir=tmp_path)

    assert hourly["complete"].all()
    assert hourly["energy_mwh"].between(5_000, 30_000).all()


def test_one_day_of_the_dutch_solar_forecast(tmp_path: Path) -> None:
    hourly = entsoe.fetch_hourly_forecast("NL", "solar", *DAY, cache_dir=tmp_path)

    assert hourly["complete"].all()
    assert hourly["energy_mwh"].max() > 0


def test_austria_has_no_offshore_wind(tmp_path: Path) -> None:
    hourly = entsoe.fetch_hourly_generation("AT", "wind_offshore", *DAY, cache_dir=tmp_path)

    assert not hourly["complete"].any()


def test_the_dutch_capacity_of_2025_and_its_wind_farms_of_100_mw_or_more(tmp_path: Path) -> None:
    capacity = entsoe.fetch_capacity("NL", 2025, cache_dir=tmp_path)
    units = entsoe.fetch_unit_capacity("NL", 2025, cache_dir=tmp_path)

    wind = units[units["technology"].isin(["wind_onshore", "wind_offshore"])]
    assert set(capacity.index) == set(entsoe.PRODUCTION_TYPES)
    assert (wind["capacity_mw"] >= 100).all()
    assert (wind["voltage_kv"] >= 66).all()
    assert wind["capacity_mw"].sum() < capacity[["wind_onshore", "wind_offshore"]].sum()
