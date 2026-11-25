import dataclasses
import json
from pathlib import Path

import httpx
import pandas as pd
import pytest
from synthetic import PV, solar_year, wind_year

from eu_solar_wind_audit import calibrate, cli, eligibility, evaluation, fits, physics
from eu_solar_wind_audit.config import CONFIG_DIR, Country, load_country

YEARS = list(evaluation.TRAINING_YEARS)


def test_calibrate_france_writes_the_tables_the_description_and_the_frozen_fit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    requested: list[tuple[str, int]] = []

    def year_inputs(
        client: httpx.Client, technology: physics.Technology, year: int, parameters: Country
    ) -> eligibility.YearInputs:
        requested.append((technology, year))
        return wind_year(year)

    monkeypatch.setattr(eligibility, "year_inputs", year_inputs)
    output, fits_dir = tmp_path / "out", tmp_path / "fits"

    cli.main(["calibrate", "FR", "wind_onshore", str(output), "--fits-dir", str(fits_dir)])

    assert requested == [("wind_onshore", year) for year in YEARS]
    tables = {
        path.name.removeprefix("wind_onshore-").removesuffix(".parquet")
        for path in output.glob("*.parquet")
    }
    assert tables == {
        "folds",
        "strata",
        "predictions",
        "metrics",
        "summary",
        "negative-prices",
        "composition",
    }
    fold_table = pd.read_parquet(output / "wind_onshore-folds.parquet")
    # Onshore wind has no fold in 2023, the gap of its reference.
    assert fold_table["fit"].tolist() == ["2019", "2020", "2021", "2022", "2024", "2025", "final"]
    description = json.loads((output / "wind_onshore-calibration.json").read_text())
    assert [year["year"] for year in description["inputs"]["years"]] == YEARS
    assert description["fold_years"] == [2019, 2020, 2021, 2022, 2024, 2025]
    frozen = fits.read(fits_dir / "wind_onshore.json")
    assert frozen.technology == "wind_onshore"
    assert sorted(frozen.fold_calibrations) == [2019, 2020, 2021, 2022, 2024, 2025]
    assert frozen.final.calibration.fit.training_years == (2019, 2020, 2021, 2022, 2024, 2025)
    assert frozen.git_describe == description["git_describe"]


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


def test_calibrate_germany_takes_its_config_and_writes_its_fit_in_fits_de(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    countries: list[str] = []

    def year_inputs(
        client: httpx.Client, technology: physics.Technology, year: int, parameters: Country
    ) -> eligibility.YearInputs:
        countries.append(parameters.country)
        return wind_year(year, "DE")

    monkeypatch.setattr(eligibility, "year_inputs", year_inputs)
    # The default config and fits folders are relative to the repository root.
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config").symlink_to(Path(__file__).parents[1] / "config")

    cli.main(["calibrate", "DE", "wind_onshore", "out"])

    assert set(countries) == {"DE"}
    frozen = fits.read(tmp_path / "fits" / "DE" / "wind_onshore.json")
    # German onshore wind has a fold in every year.
    assert sorted(frozen.fold_calibrations) == YEARS
    description = json.loads((tmp_path / "out" / "wind_onshore-calibration.json").read_text())
    assert description["inputs"]["prices"] == "SMARD.de, day-ahead prices of DE-LU, filter 4169"
    assert description["inputs"]["published"] == "SMARD.de, feed-in of Germany, filter 4067"


def test_calibrate_a_danish_price_area_takes_its_config_and_writes_its_fit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        eligibility,
        "year_inputs",
        lambda client, technology, year, parameters: wind_year(year, "DK2"),
    )
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config").symlink_to(Path(__file__).parents[1] / "config")

    cli.main(["calibrate", "DK2", "wind_onshore", "out"])

    frozen = fits.read(tmp_path / "fits" / "DK2" / "wind_onshore.json")
    assert sorted(frozen.fold_calibrations) == YEARS
    description = json.loads((tmp_path / "out" / "wind_onshore-calibration.json").read_text())
    assert description["inputs"]["prices"] == "SMARD.de, day-ahead prices of DK2, filter 253"
    assert description["inputs"]["published"] == (
        "Energi Data Service, ProductionConsumptionSettlement of DK2: "
        "OnshoreWindLt50kW_MWh + OnshoreWindGe50kW_MWh"
    )
    assert description["inputs"]["regions"] == "GISCO LAU 2024 municipalities"


def test_calibrate_refuses_a_config_of_another_country(tmp_path: Path) -> None:
    config = CONFIG_DIR / "fr.toml"

    with pytest.raises(SystemExit):
        cli.main(["calibrate", "DE", "solar", str(tmp_path), "--config", str(config)])
