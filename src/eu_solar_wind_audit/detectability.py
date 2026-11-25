"""Minimal detectable effect of the modulation estimate (evaluation plan, axis 4, and Deviations).

The error of the potential on the N hours is measured on the P hours that resemble them: in the
held-out years of the MDE_YEARS of the country, the eligible hours of clear-sky days with the sun at
MIN_ELEVATION_DEG or more. The relative error of a Europe/Paris day is sum(y - p) / sum(p) over
these hours, with y the published output and p the potential with strata and cross-fitted level
of folds. With s the standard deviation of these daily errors and P_d the potential at the N hours
of the day d, the minimal detectable effect of a year is

    Z_SUM * s * sqrt(sum over the days of the year of P_d^2),

Z_SUM = z(0.95) + z(0.80): a one-sided test at 5 % detects such an effect with 80 % power. An
estimate below it is inconclusive.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm

from eu_solar_wind_audit import days, evaluation, modulation, physics

# The fold years of 2023-2025: French onshore wind has no fold in 2023 (folds.FOLD_YEARS).
MDE_YEARS: dict[str, dict[physics.Technology, tuple[int, ...]]] = {
    "FR": {"solar": (2023, 2024, 2025), "wind_onshore": (2024, 2025)},
    "DE": {"solar": (2023, 2024, 2025), "wind_onshore": (2023, 2024, 2025)},
    "DK1": {"solar": (2023, 2024, 2025), "wind_onshore": (2023, 2024, 2025)},
    "DK2": {"solar": (2023, 2024, 2025), "wind_onshore": (2023, 2024, 2025)},
}
MIN_ELEVATION_DEG = 30.0
Z_SUM = float(norm.ppf(0.95) + norm.ppf(0.80))


def daily_errors(
    predictions: pd.DataFrame,
    covariates: pd.DataFrame,
    technology: physics.Technology,
    country: str,
) -> "pd.Series[float]":
    """Relative error of each Europe/Paris day with selected hours, from `predictions`, the
    evaluation.Predictions of the folds of `technology` in `country`, and `covariates`, the
    precompute.Covariates of their hours."""
    table = evaluation.Predictions.validate(predictions)
    elevation = covariates["sun_elevation_deg"].reindex(table.index)
    selected = table[
        table["held_out_year"].isin(MDE_YEARS[country][technology]).to_numpy()
        & table["eligible"].to_numpy()
        & table["sky_class"].eq("clear").to_numpy()
        & (elevation >= MIN_ELEVATION_DEG).to_numpy()
    ]
    day = days.local_days(pd.DatetimeIndex(selected.index))
    error = (selected["published_mwh"] - selected["calibrated_level_mwh"]).groupby(day).sum()
    potential = selected["calibrated_level_mwh"].groupby(day).sum()
    relative: pd.Series[float] = (error / potential).rename("relative_error")
    return relative


def minimal_detectable_effects(errors: pd.Series, hours: pd.DataFrame) -> "pd.Series[float]":
    """Minimal detectable effect in GWh of each Europe/Paris year of `hours`, a
    modulation.NegativeHours frame, from the daily relative `errors`."""
    table = modulation.NegativeHours.validate(hours)
    local = days.local_days(pd.DatetimeIndex(table.index))
    per_day = table["potential_mwh"].groupby(local).sum()
    year = pd.DatetimeIndex(per_day.index).year.astype(str).rename("year")
    spread = np.sqrt((per_day**2).groupby(year).sum())
    effects: pd.Series[float] = Z_SUM * float(errors.std()) * spread / modulation.MWH_PER_GWH
    return effects.rename("mde_gwh")
