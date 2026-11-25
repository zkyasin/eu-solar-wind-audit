import hashlib
import os

import pandas as pd
import pytest

from eu_solar_wind_audit.sources import ens

pytestmark = pytest.mark.skipif(os.environ.get("RUN_NETWORK_TESTS") != "1", reason="network")


@pytest.fixture(scope="module")
def turbines(tmp_path_factory: pytest.TempPathFactory) -> pd.DataFrame:
    paths = ens.download_registers(raw_dir=tmp_path_factory.mktemp("ens"))
    for path, expected in zip(paths, [ens.CURRENT_SHA256, ens.HISTORIC_SHA256], strict=True):
        with path.open("rb") as file:
            assert hashlib.file_digest(file, "sha256").hexdigest() == expected
    return ens.read_turbines(paths)


def test_onshore_capacity_at_the_end_of_2025(turbines: pd.DataFrame) -> None:
    end = pd.Timestamp("2025-12-31")
    panel = ens.monthly_capacity(turbines, end, end)
    west = panel.loc[panel["municipality"] >= 410, "capacity_mw"].sum()
    east = panel.loc[panel["municipality"].between(101, 400), "capacity_mw"].sum()

    # Energi Data Service, CapacityPerMunicipality, 1 January 2026: 4,138.6 MW in DK1 and 745.2 MW
    # in DK2; the register leaves out the capacity of test turbines.
    assert west == pytest.approx(4035.0, abs=0.5)
    assert east == pytest.approx(749.1, abs=0.5)


def test_every_onshore_turbine_with_a_capacity_has_a_municipality(turbines: pd.DataFrame) -> None:
    onshore = turbines[~turbines["offshore"] & turbines["capacity_kw"].notna()]
    unknown = onshore[onshore["municipality"].isna()]

    # Four household turbines, 40 kW in all.
    assert len(unknown) == 4
    assert unknown["capacity_kw"].sum() == pytest.approx(40.0)
