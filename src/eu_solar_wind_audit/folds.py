"""Leave-one-year-out evaluation of the calibrated potential of mainland France (evaluation plan,
"Two model objects", item 2, "Axis 1: model quality" and Deviations).

The inputs are the eligibility.YearInputs of one technology, one per UTC year. Each year of
FOLD_YEARS among them is held out in turn, and everything is learned on the eligible hours of the
other fold years: the parameters of calibration, the decile edges and stratum ratios of strata for
the potential with these parameters, and the climatology. Five models predict every hour of the
held-out year, as the model columns of evaluation.Predictions:

- raw: the nominal model of physics.model, from precompute at level 1, without horizontal modules
  or seasonal term and at the configured DC/AC ratio for solar, at speed scale 1 for onshore wind;
- climatology: evaluation.climatology;
- calibrated: the potential with the parameters of the fold;
- calibrated_strata: the potential with strata, strata.apply_ratios;
- calibrated_level: the potential with strata times the annual level of the held-out year,
  cross-fitted within it by weeks, strata.cross_fitted_levels.

The season of an hour is that of its UTC month, and its sky class that of days.sky_classes over the
joined inputs, so that the days across New Year are whole when the input years follow each other.
The eligible hours of input years that are no fold years are never learned from or evaluated.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import pandas as pd
from pandera.typing import DataFrame

from eu_solar_wind_audit import (
    calibration,
    days,
    eligibility,
    evaluation,
    physics,
    precompute,
    strata,
)
from eu_solar_wind_audit.pv import PVParameters

# Onshore wind has no reference from June 2022 to December 2023 in Paris time, so its 2022 fold
# ends with May and 2023 is no fold (evaluation plan, Deviations).
FOLD_YEARS: dict[physics.Technology, tuple[int, ...]] = {
    "solar": (2019, 2020, 2021, 2022, 2023, 2024, 2025),
    "wind_onshore": (2019, 2020, 2021, 2022, 2024, 2025),
}

Calibration = calibration.SolarCalibration | calibration.WindCalibration


@dataclass(frozen=True)
class Potential:
    """Calibrated potential learned on the eligible hours of the UTC years
    `calibration.fit.training_years`: the parameters of calibration and the strata of the potential
    with these parameters."""

    calibration: Calibration
    ratios: strata.StratumRatios


@dataclass(frozen=True)
class Report:
    """Leave-one-year-out evaluation of one technology: the potential of each fold, keyed by its
    held-out year; the predictions of the held-out years, their metrics and the summary of these
    metrics by period of evaluation.PERIODS."""

    folds: dict[int, Potential]
    predictions: DataFrame[evaluation.Predictions]
    metrics: DataFrame[evaluation.Metrics]
    period_summary: pd.DataFrame


def solar(years: Sequence[eligibility.YearInputs], parameters: PVParameters) -> Report:
    """Report of the module docstring on the solar inputs `years`, whose arrays come from
    precompute.solar with the PV `parameters`.

    Raises ValueError as `wind_onshore` does.
    """
    arrays = [
        inputs.arrays for inputs in years if isinstance(inputs.arrays, precompute.SolarArrays)
    ]
    if len(arrays) != len(years):
        raise ValueError("expected the arrays of precompute.solar")

    def nominal(part: precompute.SolarArrays) -> pd.Series:
        return precompute.solar_energy(part, parameters.dc_ac_ratio)

    def fit(
        parts: Sequence[precompute.SolarArrays], published_mwh: pd.Series, eligible: pd.Series
    ) -> calibration.SolarCalibration:
        return calibration.fit_solar(parts, published_mwh, eligible, parameters)

    def energy(part: precompute.SolarArrays, fitted: calibration.SolarCalibration) -> pd.Series:
        return precompute.solar_energy(
            part,
            fitted.dc_ac_ratio,
            level=fitted.level,
            horizontal_share=fitted.horizontal_share,
            seasonal_amplitude=fitted.seasonal_amplitude,
        )

    return _report("solar", years, arrays, nominal, fit, energy)


def wind_onshore(years: Sequence[eligibility.YearInputs]) -> Report:
    """Report of the module docstring on the onshore wind inputs `years`, whose arrays come from
    precompute.wind_onshore.

    Raises ValueError unless the frames of each input are on the hours of one UTC year of
    evaluation.TRAINING_YEARS, the years increase, two of them are fold years, and each eligible
    hour of a held-out year has a cross-fitted level; and as calibration and strata do on their
    inputs.
    """
    arrays = [inputs.arrays for inputs in years if isinstance(inputs.arrays, precompute.WindArrays)]
    if len(arrays) != len(years):
        raise ValueError("expected the arrays of precompute.wind_onshore")

    def energy(part: precompute.WindArrays, fitted: calibration.WindCalibration) -> pd.Series:
        return precompute.wind_energy(part, fitted.speed_scale)

    return _report(
        "wind_onshore", years, arrays, precompute.wind_energy, calibration.fit_wind, energy
    )


def _report[Arrays: precompute.SolarArrays | precompute.WindArrays, Fitted: Calibration](
    technology: physics.Technology,
    years: Sequence[eligibility.YearInputs],
    arrays: Sequence[Arrays],
    nominal: Callable[[Arrays], pd.Series],
    fit: Callable[[Sequence[Arrays], pd.Series, pd.Series], Fitted],
    energy: Callable[[Arrays, Fitted], pd.Series],
) -> Report:
    """Report of the module docstring, from the `nominal` output of the arrays of a year, the
    `fit` of calibration and the `energy` of the potential with fitted parameters."""
    calendar = _calendar_years(years)
    fold_years = [year for year in calendar if year in FOLD_YEARS[technology]]
    if len(fold_years) < 2:
        raise ValueError(f"leave-one-year-out needs two of the fold years {FOLD_YEARS[technology]}")
    arrays_of = dict(zip(calendar, arrays, strict=True))
    covariates = pd.concat([part.covariates for part in arrays])
    hours = pd.DatetimeIndex(covariates.index)
    eligible = pd.concat([inputs.eligibility["eligible"] for inputs in years])
    observed = pd.DataFrame(
        {
            "published_mwh": pd.concat([inputs.published["energy_mwh"] for inputs in years]),
            "capacity_mw": covariates["capacity_mw"],
            "eligible": eligible & hours.year.isin(fold_years),
        }
    )
    season = pd.Series(hours.month, index=hours).map(evaluation.SEASON_OF_MONTH)
    sky_class = days.sky_classes(covariates)

    def potential_mwh(fitted: Fitted, selected: Sequence[int]) -> pd.Series:
        potential: pd.Series = pd.concat([energy(arrays_of[year], fitted) for year in selected])
        return potential

    def learn(selected: Sequence[int]) -> Fitted:
        rows = hours.year.isin(selected)
        parts = [arrays_of[year] for year in selected]
        return fit(parts, observed.loc[rows, "published_mwh"], observed.loc[rows, "eligible"])

    def learn_strata(fitted: Fitted, selected: Sequence[int]) -> strata.StratumRatios:
        rows = hours.year.isin(selected)
        return strata.fit_ratios(
            observed.loc[rows, "published_mwh"],
            potential_mwh(fitted, selected),
            covariates[rows],
            observed.loc[rows, "eligible"],
        )

    def predict(held_out: int, fitted: Fitted, ratios: strata.StratumRatios) -> pd.DataFrame:
        rows = hours.year == held_out
        in_year = observed[rows]
        calibrated = energy(arrays_of[held_out], fitted)
        with_strata = strata.apply_ratios(ratios, calibrated, covariates[rows])
        level = strata.cross_fitted_levels(
            in_year["published_mwh"], with_strata, in_year["eligible"]
        )
        unleveled = in_year.index[in_year["eligible"].to_numpy() & level.isna().to_numpy()]
        if not unleveled.empty:
            raise ValueError(f"eligible hour {unleveled[0]} has no cross-fitted level")
        return pd.DataFrame(
            {
                "held_out_year": held_out,
                "published_mwh": in_year["published_mwh"],
                "capacity_mw": in_year["capacity_mw"],
                "raw_mwh": nominal(arrays_of[held_out]),
                "climatology_mwh": evaluation.climatology(observed, held_out),
                "calibrated_mwh": calibrated,
                "calibrated_strata_mwh": with_strata,
                "calibrated_level_mwh": with_strata * level,
                "eligible": in_year["eligible"],
                "season": season[rows],
                "sky_class": sky_class[rows],
            }
        )

    folds = {}
    held_out_years = []
    for held_out in fold_years:
        training = [year for year in fold_years if year != held_out]
        fitted = learn(training)
        ratios = learn_strata(fitted, training)
        folds[held_out] = Potential(fitted, ratios)
        held_out_years.append(predict(held_out, fitted, ratios))
    predictions = evaluation.Predictions.validate(pd.concat(held_out_years))
    metrics = evaluation.metrics(predictions, technology)

    return Report(folds, predictions, metrics, evaluation.period_summary(metrics))


def _calendar_years(years: Sequence[eligibility.YearInputs]) -> list[int]:
    """UTC year of each input. Raises ValueError unless the frames of each input are on the same
    hours, all in one year of evaluation.TRAINING_YEARS, and the years increase."""
    calendar = []
    for inputs in years:
        hours = pd.DatetimeIndex(inputs.arrays.covariates.index)
        frames = [inputs.published, inputs.price_classes, inputs.eligibility]
        if not all(frame.index.equals(hours) for frame in frames):
            raise ValueError("the frames of an input are not on the same hours")
        found = hours.year.unique().tolist()
        if len(found) != 1:
            raise ValueError(f"an input holds the hours of {found}, not of one UTC year")
        calendar += found
    if calendar != sorted(set(calendar)):
        raise ValueError(f"expected one input per UTC year in increasing order, not {calendar}")
    outside = [year for year in calendar if year not in evaluation.TRAINING_YEARS]
    if outside:
        first, last = evaluation.TRAINING_YEARS[0], evaluation.TRAINING_YEARS[-1]
        raise ValueError(f"{outside[0]} lies outside the training years {first}-{last}")
    return calendar
