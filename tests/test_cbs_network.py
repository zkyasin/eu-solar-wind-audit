import os
from pathlib import Path

import pytest

from eu_solar_wind_audit.sources import cbs

pytestmark = pytest.mark.skipif(os.environ.get("RUN_NETWORK_TESTS") != "1", reason="network")


@pytest.fixture(scope="module")
def raw_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("cbs")


def test_the_provinces_add_up_to_the_capacity_of_2025(raw_dir: Path) -> None:
    capacity = cbs.province_capacity(cbs.download(cbs.PROVINCES, raw_dir=raw_dir))
    end_2025 = capacity[capacity["year"] == 2025]

    # 7,017 MW for the Netherlands in the table, revised provisional; the provinces are rounded.
    assert len(end_2025) == 12
    assert end_2025["capacity_mw"].sum() == pytest.approx(7017.0, abs=2.0)
    assert end_2025.set_index("group").loc["Flevoland", "capacity_mw"] == 2152.0


def test_the_hub_height_classes_of_2025(raw_dir: Path) -> None:
    capacity = cbs.hub_height_capacity(cbs.download(cbs.HUB_HEIGHTS, raw_dir=raw_dir))
    end_2025 = capacity[capacity["year"] == 2025].set_index("group")["capacity_mw"]

    assert end_2025.to_dict() == {"71 to 95 m": 1348.0, "96 m or more": 4930.0, "up to 70 m": 739.0}
