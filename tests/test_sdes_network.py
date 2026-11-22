import os
from pathlib import Path

import httpx
import pandas as pd
import pytest

from eu_solar_wind_audit.sources import sdes

pytestmark = pytest.mark.skipif(os.environ.get("RUN_NETWORK_TESTS") != "1", reason="network")


@pytest.fixture(scope="module")
def workbooks(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    raw_dir = tmp_path_factory.mktemp("sdes")
    with httpx.Client(timeout=60) as client:
        return {
            "solar": sdes.download_dashboard(client, sdes.SOLAR_2026_Q2, raw_dir),
            "wind": sdes.download_dashboard(client, sdes.WIND_2026_Q2, raw_dir),
        }


def test_solar_metropolitan_total_at_the_end_of_2024(workbooks: dict[str, Path]) -> None:
    totals = sdes.read_totals(workbooks["solar"])

    metropole = totals[
        (totals["total"] == "Total métropole")
        & (totals["quarter_end"] == pd.Timestamp("2024-12-31"))
    ]
    assert round(metropole["capacity_mw"].item()) == 24365


@pytest.mark.parametrize("workbook", ["solar", "wind"])
def test_mainland_is_the_metropolitan_total_without_corsica(
    workbooks: dict[str, Path], workbook: str
) -> None:
    panel = sdes.read_capacity(workbooks[workbook])
    totals = sdes.read_totals(workbooks[workbook])

    keys = ["technology", "quarter_end"]
    mainland = sdes.metropolitan_without_corsica(panel).groupby(keys)["capacity_mw"].sum()
    metropole = totals[totals["total"] == "Total métropole"].set_index(keys)["capacity_mw"]
    corsica = panel[panel["department_code"] == "20"].set_index(keys)["capacity_mw"]
    expected = (metropole - corsica).sort_index()
    assert mainland.sort_index().index.equals(expected.index)
    assert mainland.sort_index().to_numpy() == pytest.approx(expected.to_numpy(), abs=1e-6)
