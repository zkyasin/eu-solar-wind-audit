import os

import pandas as pd
import pytest

from eu_solar_wind_audit.sources import rivm

pytestmark = pytest.mark.skipif(os.environ.get("RUN_NETWORK_TESTS") != "1", reason="network")


@pytest.fixture(scope="module")
def turbines(tmp_path_factory: pytest.TempPathFactory) -> pd.DataFrame:
    return rivm.read_turbines(rivm.download(raw_dir=tmp_path_factory.mktemp("rivm")))


def test_the_onshore_turbines_of_1_january_2026(turbines: pd.DataFrame) -> None:
    dutch = rivm.onshore(turbines)

    # CBS gives 7,017 MW of wind on land at the end of 2025.
    assert len(dutch) == 3590
    assert dutch["capacity_kw"].sum() / 1000 == pytest.approx(7005.3, abs=0.05)


def test_the_hub_height_classes_of_the_dutch_fleet(turbines: pd.DataFrame) -> None:
    classes = rivm.hub_height_classes(rivm.onshore(turbines)).set_index("hub_height_class")

    assert classes["hub_height_m"].round(1).tolist() == [54.7, 85.8, 126.9]
    assert classes["specific_power_w_m2"].round(0).tolist() == [415.0, 356.0, 330.0]
