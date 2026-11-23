import dataclasses
from pathlib import Path

import pandas as pd
from synthetic import PV, solar_year

from eu_solar_wind_audit import calibrate, evaluation
from eu_solar_wind_audit.config import CONFIG_DIR, load_country

YEARS = list(evaluation.TRAINING_YEARS)


def test_solar_calibration_reports_the_dc_ac_profile(tmp_path: Path) -> None:
    parameters = dataclasses.replace(load_country(CONFIG_DIR / "fr.toml"), solar=PV)

    # Whole years: the clear-sky days of the DC/AC sample need every hour of their local day.
    years = [solar_year(year, step=1) for year in (2019, 2020, 2021)]

    calibrated = calibrate.report_of("solar", years, parameters, replicates=3)
    written = calibrate.write(calibrated, tmp_path / "out", tmp_path / "fits", "v0-test")

    assert calibrated.dc_ac is not None
    assert {"sample", "bootstrap_5", "bootstrap_50", "bootstrap_95"} <= set(
        calibrated.dc_ac.minima.index
    )
    names = {path.name for path in written}
    assert {
        "solar-dc-ac-profiles.parquet",
        "solar-dc-ac-minima.parquet",
        "solar-dc-ac-sample.parquet",
        "solar-dc-ac-bootstrap.parquet",
        "solar.json",
    } <= names
    bootstrap = pd.read_parquet(tmp_path / "out" / "solar-dc-ac-bootstrap.parquet")
    assert len(bootstrap) == 3
