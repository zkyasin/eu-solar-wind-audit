import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from synthetic import PV, solar_year, wind_year

from eu_solar_wind_audit import fits, folds


@pytest.fixture(scope="module")
def solar_report() -> folds.Report:
    return folds.solar([solar_year(year, step=11) for year in (2019, 2020, 2021)], PV)


def frozen(report: folds.Report, technology: str = "solar") -> fits.Fit:
    return fits.Fit(
        technology="solar" if technology == "solar" else "wind_onshore",
        final=report.final,
        fold_calibrations={year: fold.calibration for year, fold in report.folds.items()},
        git_describe="v0.0.0-1-gabcdef0",
    )


def test_a_written_fit_reads_back_unchanged(solar_report: folds.Report, tmp_path: Path) -> None:
    fit = frozen(solar_report)

    fits.write(tmp_path / "solar.json", fit)
    read = fits.read(tmp_path / "solar.json")

    assert read.technology == "solar"
    assert read.git_describe == fit.git_describe
    assert read.final.calibration == fit.final.calibration
    assert read.fold_calibrations == fit.fold_calibrations
    assert read.final.ratios.decile_edges == fit.final.ratios.decile_edges
    assert read.final.ratios.pooled_ratio == fit.final.ratios.pooled_ratio
    pd.testing.assert_frame_equal(read.final.ratios.table, fit.final.ratios.table)


def test_strata_without_training_hours_are_written_as_null(
    solar_report: folds.Report, tmp_path: Path
) -> None:
    fits.write(tmp_path / "solar.json", frozen(solar_report))

    content = json.loads((tmp_path / "solar.json").read_text(encoding="utf-8"))

    table = content["final"]["strata"]["table"]
    empty = [row for row in table if row["training_hours"] == 0]
    # Solar has no eligible night hour, so the night strata are empty.
    assert empty
    assert all(row["raw_ratio"] is None for row in empty)
    assert np.isnan(fits.read(tmp_path / "solar.json").final.ratios.table["raw_ratio"]).sum() == (
        len(empty)
    )


def test_a_wind_fit_reads_back_as_a_wind_calibration(tmp_path: Path) -> None:
    report = folds.wind_onshore([wind_year(year) for year in (2019, 2020)])
    fit = frozen(report, "wind_onshore")

    fits.write(tmp_path / "wind_onshore.json", fit)

    assert fits.read(tmp_path / "wind_onshore.json").final.calibration == fit.final.calibration
