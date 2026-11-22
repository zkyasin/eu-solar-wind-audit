import os
from pathlib import Path

import pandas as pd
import pytest

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
        & (german_capacity["month"].dt.year == 2025)
    ]

    assert len(offshore) == 12
    assert offshore["capacity_mw"].between(5_000, 15_000).all()


def test_dc_solar_capacity_exceeds_ac(german_capacity: pd.DataFrame) -> None:
    solar = german_capacity.pivot(index="month", columns="technology", values="capacity_mw")

    assert (solar["solar_dc"] > solar["solar_ac"]).all()
