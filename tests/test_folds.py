import dataclasses
from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest
from pandera.typing import DataFrame

from eu_solar_wind_audit import (
    calibration,
    days,
    eligibility,
    evaluation,
    folds,
    physics,
    precompute,
    regimes,
    strata,
)
from eu_solar_wind_audit.config import WindGeneration
from eu_solar_wind_audit.pv import PVParameters, System
from eu_solar_wind_audit.sources import odre

CAPACITY_MW = 100.0
PV = PVParameters(fleet=(System(35.0, 180.0, 0.25), System(15.0, 110.0, 0.75)), dc_ac_ratio=1.2)
# Level, horizontal share and seasonal amplitude of the published solar output.
SOLAR_TRUTH = (0.85, 0.55, 0.04)
GENERATIONS = (
    WindGeneration("VestasV82_1.65MW_82", 80.0, 2012),
    WindGeneration("IEA_Reference_3.4MW_130", 100.0),
)
WIND_TRUTH = 0.88
MODELS = ["raw", "climatology", "calibrated", "calibrated_strata", "calibrated_level"]
YEARS = list(range(2019, 2026))
WIND_FOLDS = [2019, 2020, 2021, 2022, 2024, 2025]


def utc_hours(year: int, step: int = 5) -> pd.DatetimeIndex:
    """Every `step`-th hour of the UTC year `year`, counted back from its last hour, so that the
    hours go through every hour of the day in every month."""
    hours = pd.date_range(
        f"{year}-01-01", f"{year + 1}-01-01", freq="h", inclusive="left", tz="UTC", unit="ns"
    )
    return hours[::-step][::-1].rename("time")


def covariates(
    hours: pd.DatetimeIndex, rng: np.random.Generator
) -> DataFrame[precompute.Covariates]:
    """100 MW on `hours`, with the sun above the horizon from 07:00 to 17:00 UTC and a fleet
    clear-sky index drawn for each Europe/Paris day."""
    _, day = np.unique(days.local_days(hours), return_inverse=True)
    frame = pd.DataFrame(
        {
            "complete": True,
            "capacity_mw": CAPACITY_MW,
            "capacity_provisional": False,
            "era5_source": "ERA5",
            "sun_elevation_deg": 60.0 - 10.0 * np.abs(hours.hour.to_numpy() - 12),
            "ghi_w_m2": rng.uniform(40.0, 110.0, day.max() + 1)[day],
            "clear_sky_ghi_w_m2": 100.0,
            "air_temperature_c": 20.0,
        },
        index=hours,
    )
    return precompute.Covariates.validate(frame)


def year_inputs(
    arrays: precompute.SolarArrays | precompute.WindArrays,
    truth_mwh: pd.Series,
    rng: np.random.Generator,
) -> eligibility.YearInputs:
    """Inputs on the hours of `arrays`: the published output is `truth_mwh` with a noise of 0.01
    in capacity factor, definitive, except where odre.onshore_hourly has no onshore wind, and one
    hour in 50 has a negative price."""
    hours = pd.DatetimeIndex(truth_mwh.index)
    published = pd.DataFrame(
        {
            "energy_mwh": truth_mwh + rng.normal(0.0, 0.01 * CAPACITY_MW, hours.size),
            "complete": True,
            "status": "definitive",
        },
        index=hours,
    )
    if isinstance(arrays, precompute.WindArrays):
        offshore = (hours >= odre.FIRST_OFFSHORE_POWER) & (hours < odre.WIND_SPLIT_START)
        published.loc[offshore, "energy_mwh"] = np.nan
        published.loc[offshore, "complete"] = False
        published.loc[offshore, "status"] = None
    prices = pd.Series(50.0, index=pd.date_range(hours[0], hours[-1], freq="h"))
    prices[hours[::50]] = -5.0
    classes = regimes.classify_hours(prices).reindex(hours)
    technology: physics.Technology = (
        "solar" if isinstance(arrays, precompute.SolarArrays) else "wind_onshore"
    )
    rules = eligibility.eligible_hours(technology, published, arrays.covariates, classes)
    return eligibility.YearInputs(odre.Hourly.validate(published), classes, arrays, rules)


def solar_year(year: int, step: int = 5) -> eligibility.YearInputs:
    rng = np.random.default_rng(year)
    hours = utc_hours(year, step)
    arrays = precompute.SolarArrays(
        fleet_mwh=rng.uniform(0.0, 80.0, (hours.size, 2, precompute.DC_AC_GRID.size)),
        horizontal_mwh=rng.uniform(0.0, 80.0, (hours.size, precompute.DC_AC_GRID.size)),
        shares=np.array([system.share for system in PV.fleet]),
        covariates=covariates(hours, rng),
        centroid=(46.0, 2.0),
    )
    return year_inputs(arrays, precompute.solar_energy(arrays, PV.dc_ac_ratio, *SOLAR_TRUTH), rng)


def wind_year(year: int) -> eligibility.YearInputs:
    """Each generation holds 50 MW, spread over 20 cells whose hub-height speeds scatter around
    a mean speed of the hour."""
    rng = np.random.default_rng(year)
    hours = utc_hours(year)
    cells = 20
    mean_speed = rng.uniform(2.0, 14.0, hours.size)
    capacity_mw = np.zeros((len(GENERATIONS), hours.size, precompute.SPEED_CENTRES.size))
    for generation in capacity_mw:
        speeds = mean_speed[:, np.newaxis] * rng.lognormal(0.0, 0.25, (hours.size, cells))
        bins = np.digitize(np.minimum(speeds, 39.0), precompute.SPEED_EDGES) - 1
        rows = np.repeat(np.arange(hours.size), cells)
        np.add.at(generation, (rows, bins.ravel()), CAPACITY_MW / len(GENERATIONS) / cells)
    arrays = precompute.WindArrays(
        capacity_mw=capacity_mw, generations=GENERATIONS, covariates=covariates(hours, rng)
    )
    return year_inputs(arrays, precompute.wind_energy(arrays, WIND_TRUTH), rng)


def solar_potential(
    inputs: eligibility.YearInputs, fitted: calibration.SolarCalibration | None
) -> "pd.Series[float]":
    """Solar output on the hours of `inputs` with the parameters of `fitted`, or nominal."""
    assert isinstance(inputs.arrays, precompute.SolarArrays)
    if fitted is None:
        return precompute.solar_energy(inputs.arrays, PV.dc_ac_ratio)
    return precompute.solar_energy(
        inputs.arrays,
        fitted.dc_ac_ratio,
        level=fitted.level,
        horizontal_share=fitted.horizontal_share,
        seasonal_amplitude=fitted.seasonal_amplitude,
    )


def solar_arrays(inputs: list[eligibility.YearInputs]) -> list[precompute.SolarArrays]:
    arrays = [part.arrays for part in inputs if isinstance(part.arrays, precompute.SolarArrays)]
    assert len(arrays) == len(inputs)
    return arrays


def joined(inputs: list[eligibility.YearInputs]) -> pd.DataFrame:
    """Published energy, capacity and eligibility of the hours of `inputs`."""
    return pd.DataFrame(
        {
            "published_mwh": pd.concat([part.published["energy_mwh"] for part in inputs]),
            "capacity_mw": pd.concat([part.arrays.covariates["capacity_mw"] for part in inputs]),
            "eligible": pd.concat([part.eligibility["eligible"] for part in inputs]),
        }
    )


@pytest.fixture(scope="module")
def solar_inputs() -> list[eligibility.YearInputs]:
    return [solar_year(year) for year in YEARS]


@pytest.fixture(scope="module")
def solar_report(solar_inputs: list[eligibility.YearInputs]) -> folds.Report:
    return folds.solar(solar_inputs, PV)


@pytest.fixture(scope="module")
def wind_inputs() -> list[eligibility.YearInputs]:
    return [wind_year(year) for year in YEARS]


@pytest.fixture(scope="module")
def wind_report(wind_inputs: list[eligibility.YearInputs]) -> folds.Report:
    return folds.wind_onshore(wind_inputs)


def test_each_fold_learns_on_the_other_fold_years(solar_report: folds.Report) -> None:
    assert list(solar_report.folds) == YEARS
    for held_out, potential in solar_report.folds.items():
        fit = potential.calibration.fit
        assert fit.training_years == tuple(year for year in YEARS if year != held_out)
        # The strata learn on the eligible hours that the parameters learn on.
        assert potential.ratios.table["training_hours"].sum() == fit.eligible_hours


def with_published_output(inputs: eligibility.YearInputs, factor: float) -> eligibility.YearInputs:
    published = inputs.published.assign(energy_mwh=factor * inputs.published["energy_mwh"])
    return dataclasses.replace(inputs, published=published)


def test_a_fold_learns_nothing_from_its_held_out_year() -> None:
    inputs = [solar_year(year) for year in (2019, 2020, 2021)]
    tripled = [inputs[0], with_published_output(inputs[1], 3.0), inputs[2]]

    report, changed = folds.solar(inputs, PV), folds.solar(tripled, PV)

    assert changed.folds[2020].calibration == report.folds[2020].calibration
    assert changed.folds[2020].ratios.decile_edges == report.folds[2020].ratios.decile_edges
    pd.testing.assert_frame_equal(changed.folds[2020].ratios.table, report.folds[2020].ratios.table)
    in_2020 = report.predictions["held_out_year"] == 2020
    learned = ["raw_mwh", "climatology_mwh", "calibrated_mwh", "calibrated_strata_mwh"]
    pd.testing.assert_frame_equal(
        changed.predictions.loc[in_2020, learned], report.predictions.loc[in_2020, learned]
    )
    # The level is cross-fitted within the held-out year, so it follows its published output.
    np.testing.assert_allclose(
        changed.predictions.loc[in_2020, "calibrated_level_mwh"],
        3.0 * report.predictions.loc[in_2020, "calibrated_level_mwh"],
    )
    # The other folds learn from 2020.
    assert changed.folds[2019].calibration != report.folds[2019].calibration


def test_wind_folds_follow_the_eligible_hours(
    wind_inputs: list[eligibility.YearInputs], wind_report: folds.Report
) -> None:
    eligible = {
        year: inputs.eligibility.index[inputs.eligibility["eligible"]]
        for year, inputs in zip(YEARS, wind_inputs, strict=True)
    }
    # Onshore wind of 2024 starts at midnight in Paris, the last hour of 2023 in UTC.
    assert eligible[2023].equals(pd.DatetimeIndex([odre.WIND_SPLIT_START]))
    assert eligible[2022][-1] < odre.FIRST_OFFSHORE_POWER

    assert list(wind_report.folds) == WIND_FOLDS
    for year in WIND_FOLDS:
        others = [other for other in WIND_FOLDS if other != year]
        fit = wind_report.folds[year].calibration.fit
        assert fit.training_years == tuple(others)
        assert fit.eligible_hours == sum(eligible[other].size for other in others)
    predictions = wind_report.predictions
    assert predictions["held_out_year"].unique().tolist() == WIND_FOLDS
    for year in WIND_FOLDS:
        held_out = predictions[predictions["held_out_year"] == year]
        assert held_out.index[held_out["eligible"]].equals(eligible[year])


def test_the_raw_model_is_the_nominal_potential(
    solar_inputs: list[eligibility.YearInputs],
    solar_report: folds.Report,
    wind_inputs: list[eligibility.YearInputs],
    wind_report: folds.Report,
) -> None:
    for inputs in solar_inputs:
        raw = solar_report.predictions.loc[inputs.published.index, "raw_mwh"]
        pd.testing.assert_series_equal(
            raw, solar_potential(inputs, None), check_names=False, check_freq=False
        )
    for year, inputs in zip(YEARS, wind_inputs, strict=True):
        if year in WIND_FOLDS:
            assert isinstance(inputs.arrays, precompute.WindArrays)
            raw = wind_report.predictions.loc[inputs.published.index, "raw_mwh"]
            expected = precompute.wind_energy(inputs.arrays)
            pd.testing.assert_series_equal(raw, expected, check_names=False, check_freq=False)


def test_the_predictions_hold_every_model_of_each_fold(
    solar_inputs: list[eligibility.YearInputs], solar_report: folds.Report
) -> None:
    predictions = solar_report.predictions

    evaluation.Predictions.validate(predictions)
    models = predictions.filter(regex=evaluation.MODEL_COLUMNS).columns
    assert models.tolist() == [f"{model}_mwh" for model in MODELS]
    observed = joined(solar_inputs)
    for year, inputs in zip(YEARS, solar_inputs, strict=True):
        potential = solar_report.folds[year]
        held_out = predictions[predictions["held_out_year"] == year]
        assert held_out.index.equals(inputs.published.index)
        pd.testing.assert_frame_equal(
            held_out[["published_mwh", "capacity_mw", "eligible"]], observed.loc[held_out.index]
        )
        assert isinstance(potential.calibration, calibration.SolarCalibration)
        calibrated = solar_potential(inputs, potential.calibration)
        with_strata = strata.apply_ratios(potential.ratios, calibrated, inputs.arrays.covariates)
        level = strata.cross_fitted_levels(
            inputs.published["energy_mwh"], with_strata, inputs.eligibility["eligible"]
        )
        expected = {
            "climatology_mwh": evaluation.climatology(observed, year),
            "calibrated_mwh": calibrated,
            "calibrated_strata_mwh": with_strata,
            "calibrated_level_mwh": with_strata * level,
        }
        for column, values in expected.items():
            pd.testing.assert_series_equal(
                held_out[column], values, check_names=False, check_freq=False
            )


def test_metrics_and_their_summary_come_from_evaluation(solar_report: folds.Report) -> None:
    pd.testing.assert_frame_equal(
        solar_report.metrics, evaluation.metrics(solar_report.predictions, "solar")
    )
    pd.testing.assert_frame_equal(
        solar_report.period_summary, evaluation.period_summary(solar_report.metrics)
    )
    assert solar_report.metrics["model"].unique().tolist() == MODELS


def test_season_and_sky_class_come_from_the_joined_years() -> None:
    # Every hour, so that the Europe/Paris days are whole.
    inputs = [solar_year(year, step=1) for year in (2019, 2020)]

    predictions = folds.solar(inputs, PV).predictions

    covariates = pd.concat([part.arrays.covariates for part in inputs])
    pd.testing.assert_series_equal(predictions["sky_class"], days.sky_classes(covariates))
    months = pd.DatetimeIndex(predictions.index).month
    assert predictions["season"].tolist() == months.map(evaluation.SEASON_OF_MONTH).tolist()
    # New Year's Day 2020 in Paris starts with the last hour of 2019, so both years cover it. The
    # days that start before 2019 and end after 2020 in Paris are cut.
    without_class = predictions.index[predictions["sky_class"].isna()]
    expected = pd.date_range("2019-01-01 00:00", "2019-01-01 22:00", freq="h", tz="UTC")
    assert without_class.equals(expected.append(pd.DatetimeIndex(["2020-12-31 23:00"], tz="UTC")))
    assert predictions.loc[without_class, "eligible"].any()


def realtime_from(inputs: eligibility.YearInputs, start: str) -> eligibility.YearInputs:
    """`inputs` of onshore wind whose published hours from `start` on are real-time values."""
    published = inputs.published.copy()
    published.loc[published.index >= pd.Timestamp(start, tz="UTC"), "status"] = "realtime"
    rules = eligibility.eligible_hours(
        "wind_onshore", published, inputs.arrays.covariates, inputs.price_classes
    )
    return dataclasses.replace(inputs, published=published, eligibility=rules)


def test_an_eligible_hour_without_a_cross_fitted_level_is_refused() -> None:
    # The eligible hours of 2020 all lie in its first week.
    inputs = [wind_year(2019), realtime_from(wind_year(2020), "2020-01-08")]

    with pytest.raises(ValueError, match=r"2020-01-01 .* has no cross-fitted level"):
        folds.wind_onshore(inputs)


def test_inputs_of_2026_are_refused() -> None:
    inputs = [wind_year(2024), wind_year(2025), wind_year(2026)]

    with pytest.raises(ValueError, match="2026 lies outside the training years 2019-2025"):
        folds.wind_onshore(inputs)


def across_new_year() -> list[eligibility.YearInputs]:
    rng = np.random.default_rng(0)
    hours = pd.date_range("2019-12-31 20:00", periods=8, freq="h", tz="UTC", unit="ns")
    arrays = precompute.SolarArrays(
        fleet_mwh=np.ones((hours.size, 1, precompute.DC_AC_GRID.size)),
        horizontal_mwh=np.ones((hours.size, precompute.DC_AC_GRID.size)),
        shares=np.array([1.0]),
        covariates=covariates(hours.rename("time"), rng),
        centroid=(46.0, 2.0),
    )
    return [
        year_inputs(arrays, precompute.solar_energy(arrays, PV.dc_ac_ratio), rng),
        solar_year(2021),
    ]


def classes_off_the_hours() -> list[eligibility.YearInputs]:
    inputs = solar_year(2019)
    shifted = inputs.price_classes.set_axis(inputs.price_classes.index + pd.Timedelta(hours=1))
    return [dataclasses.replace(inputs, price_classes=shifted), solar_year(2020)]


@pytest.mark.parametrize(
    ("given", "error"),
    [
        (lambda: [wind_year(2019), wind_year(2020)], "expected the arrays of precompute.solar"),
        (lambda: [solar_year(2020), solar_year(2019)], r"in increasing order, not \[2020, 2019\]"),
        (lambda: [solar_year(2019), solar_year(2019)], "in increasing order"),
        (across_new_year, r"holds the hours of \[2019, 2020\], not of one UTC year"),
        (classes_off_the_hours, "not on the same hours"),
        (lambda: [solar_year(2019)], "needs two of the fold years"),
    ],
    ids=["wind", "decreasing", "twice", "across-new-year", "off-the-hours", "one-fold"],
)
def test_inputs_are_checked(given: Callable[[], list[eligibility.YearInputs]], error: str) -> None:
    with pytest.raises(ValueError, match=error):
        folds.solar(given(), PV)


def test_onshore_wind_needs_its_arrays() -> None:
    with pytest.raises(ValueError, match=r"expected the arrays of precompute\.wind_onshore"):
        folds.wind_onshore([solar_year(2019), solar_year(2020)])
