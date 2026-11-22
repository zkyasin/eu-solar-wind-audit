import os
from pathlib import Path

import pandas as pd
import pytest
from pandera.typing import DataFrame

from eu_solar_wind_audit.grid import Hourly
from eu_solar_wind_audit.sources import energy_charts

pytestmark = pytest.mark.skipif(os.environ.get("RUN_NETWORK_TESTS") != "1", reason="network")


@pytest.fixture(scope="module")
def cache_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("energy_charts")


@pytest.fixture(scope="module")
def german_capacity(cache_dir: Path) -> pd.DataFrame:
    # One request for the whole module: the API allows about two per minute.
    return energy_charts.fetch_installed_capacity("de", cache_dir=cache_dir)


def test_solar_capacity_grows_month_on_month(german_capacity: pd.DataFrame) -> None:
    for technology in ["solar_ac", "solar_dc"]:
        solar = german_capacity[german_capacity["technology"] == technology]
        assert (solar["capacity_mw"].diff().dropna() > 0).all(), technology


def test_offshore_wind_in_2025_is_between_5_and_15_gw(german_capacity: pd.DataFrame) -> None:
    offshore = german_capacity[
        (german_capacity["technology"] == "wind_offshore")
        & (german_capacity["month_end"].dt.year == 2025)
    ]

    assert len(offshore) == 12
    assert offshore["capacity_mw"].between(5_000, 15_000).all()


def test_dc_solar_capacity_exceeds_ac(german_capacity: pd.DataFrame) -> None:
    solar = german_capacity.pivot(index="month_end", columns="technology", values="capacity_mw")

    assert (solar["solar_dc"] > solar["solar_ac"]).all()


# Installed capacity at the end of 2025 from Energy-Charts (tests/fixtures/energy_charts), in MW:
# no hour of a 2025 week can exceed it.
CAPACITY_2025_MW = {"solar": 107_856, "wind_onshore": 68_178, "wind_offshore": 9_733}


@pytest.fixture(scope="module")
def june_week(cache_dir: Path) -> dict[str, DataFrame[Hourly]]:
    start = pd.Timestamp("2025-06-16", tz="UTC")
    # Two requests, one per endpoint, both for June.
    return energy_charts.fetch_hourly_power(
        start, start + pd.Timedelta(days=7), cache_dir=cache_dir
    )


def test_a_june_week_is_complete_and_below_capacity(
    june_week: dict[str, DataFrame[Hourly]],
) -> None:
    for hourly in june_week.values():
        assert len(hourly) == 168
        assert hourly["complete"].all()
    for technology, capacity_mw in CAPACITY_2025_MW.items():
        for perimeter in ["public", "total"]:
            energy = june_week[f"{perimeter}.{technology}"]["energy_mwh"]
            assert energy.between(0, capacity_mw).all(), (perimeter, technology)


def test_total_adds_solar_self_consumption_but_not_wind(
    june_week: dict[str, DataFrame[Hourly]],
) -> None:
    self_consumption = june_week["self_consumption.solar"]["energy_mwh"]

    assert (self_consumption >= 0).all()
    assert self_consumption.sum() > 0
    for technology in ["wind_onshore", "wind_offshore"]:
        pd.testing.assert_frame_equal(
            june_week[f"total.{technology}"], june_week[f"public.{technology}"]
        )
