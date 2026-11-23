"""Bootstrap interval of the modulation of mainland France during negative prices (evaluation plan,
axis 4, "Inference", and Deviations).

A replicate resamples the Europe/Paris days of the P hours: within each calendar month, a
stationary bootstrap draws as many days as the month has, in blocks of mean length MEAN_BLOCK_DAYS
that wrap around the month, and each hour counts as many times as its day is drawn. The replicate
also draws the DC/AC ratio, for solar, and one capacity factor per month, refits the calibration,
the strata and the annual levels with these weights as folds and modulation fit them, and gives
M-hat of each month over the N hours of the data, which never change. The interval of a month runs
from the 5th to the 95th percentile of its replicates; a year takes the sums of its months.

A capacity factor f scales the capacity behind the potential: the potential and the capacity of
each hour of its month are f times larger, so the calibration fits the published output over f,
which gives the same residuals.
"""

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

from eu_solar_wind_audit import (
    calibration,
    days,
    eligibility,
    folds,
    modulation,
    physics,
    precompute,
    strata,
)
from eu_solar_wind_audit.pv import PVParameters

REPLICATES = 999
SEED = 2026
MEAN_BLOCK_DAYS = 7
# Ratios of precompute.DC_AC_GRID: 1.0 to 1.3, and 1.0 to 1.5 for the sensitivity (evaluation plan,
# axis 4, and Deviations).
DC_AC_RATIOS = np.round(np.arange(100, 131) / 100, 2)
DC_AC_SENSITIVITY = np.round(np.arange(100, 151) / 100, 2)
CAPACITY_RANGE = (0.9, 1.1)
PERCENTILES = (0.05, 0.95)


@dataclass(frozen=True)
class Draw:
    """What a replicate draws: the weight of each hour, the number of times its Europe/Paris day
    is drawn; the DC/AC ratio; and the capacity factor of each hour, that of its month."""

    weights: pd.Series
    dc_ac_ratio: float
    capacity_factors: pd.Series


def draw(
    hours: pd.DatetimeIndex,
    rng: np.random.Generator,
    dc_ac_ratios: npt.NDArray[np.float64] = DC_AC_RATIOS,
) -> Draw:
    """A `Draw` on the UTC `hours`, with a DC/AC ratio among `dc_ac_ratios` and capacity factors
    uniform in CAPACITY_RANGE, from `rng`."""
    local = days.local_days(hours)
    unique_days = local.unique()
    day_months = unique_days.to_period("M")
    counts = np.zeros(len(unique_days), dtype=np.int64)
    for month in day_months.unique():
        positions = np.flatnonzero(day_months == month)
        drawn = _stationary(len(positions), rng)
        counts[positions] = np.bincount(drawn, minlength=len(positions))
    weights = pd.Series(counts[unique_days.get_indexer(local)], index=hours, name="weight")
    dc_ac_ratio = float(rng.choice(dc_ac_ratios))
    hour_months = local.to_period("M")
    months = hour_months.unique()
    factors = pd.Series(rng.uniform(*CAPACITY_RANGE, len(months)), index=months)
    capacity_factors = pd.Series(factors.reindex(hour_months).to_numpy(), index=hours)
    return Draw(weights, dc_ac_ratio, capacity_factors)


def replicate(
    years: Sequence[eligibility.YearInputs],
    technology: physics.Technology,
    pv: PVParameters,
    drawn: Draw,
    targets: pd.Series | None = None,
) -> pd.Series:
    """M-hat in GWh of each Europe/Paris month, `YYYY-MM`, with estimated hours in `years`, the
    yearly inputs of `technology`, for `drawn`. `pv` are the PV parameters of the solar arrays, at
    the ratio that `drawn` gives; onshore wind ignores them. `targets`, on the hours of `years`,
    flags the estimated hours, the N hours without it: they leave the fit, the strata and the
    annual levels.

    Raises ValueError if a year has estimated hours but no eligible hour of positive weight to give
    its annual level.
    """
    arrays = [inputs.arrays for inputs in years]
    covariates = pd.concat([part.covariates for part in arrays])
    hours = pd.DatetimeIndex(covariates.index)
    published = pd.concat([inputs.published["energy_mwh"] for inputs in years])
    estimated = (
        pd.concat([eligibility.negative_price_hours(inputs) for inputs in years])
        if targets is None
        else targets
    )
    eligible = pd.concat([inputs.eligibility["eligible"] for inputs in years]) & ~estimated
    fitted_on = eligible & hours.year.isin(folds.FOLD_YEARS[technology])
    factor = drawn.capacity_factors

    fitted: calibration.SolarCalibration | calibration.WindCalibration
    if technology == "solar":
        solar = [part for part in arrays if isinstance(part, precompute.SolarArrays)]
        fitted = calibration.fit_solar(
            solar,
            published / factor,
            fitted_on,
            dataclasses.replace(pv, dc_ac_ratio=drawn.dc_ac_ratio),
            weights=drawn.weights,
        )
    else:
        wind = [part for part in arrays if isinstance(part, precompute.WindArrays)]
        fitted = calibration.fit_wind(wind, published / factor, fitted_on, weights=drawn.weights)
    potential = pd.concat([calibration.potential_mwh(part, fitted) for part in arrays]) * factor
    scaled = covariates.assign(capacity_mw=covariates["capacity_mw"] * factor)
    ratios = strata.fit_ratios(published, potential, scaled, fitted_on, weights=drawn.weights)
    with_strata = strata.apply_ratios(ratios, potential, scaled)
    levels = strata.annual_levels(published, with_strata, eligible, weights=drawn.weights)
    level = levels.reindex(hours.year).to_numpy()
    unleveled = hours[estimated.to_numpy() & np.isnan(level)]
    if not unleveled.empty:
        raise ValueError(f"{unleveled[0].year} has estimated hours but no eligible hour")
    gap = (with_strata * level - published)[estimated.to_numpy()]
    local = pd.DatetimeIndex(gap.index).tz_convert(modulation.TIMEZONE)
    month = pd.Index(local.strftime("%Y-%m"), name="month")
    by_month: pd.Series = gap.groupby(month).sum() / modulation.MWH_PER_GWH
    return by_month


def replicates(
    years: Sequence[eligibility.YearInputs],
    technology: physics.Technology,
    pv: PVParameters,
    *,
    count: int = REPLICATES,
    seed: int = SEED,
    dc_ac_ratios: npt.NDArray[np.float64] = DC_AC_RATIOS,
    targets: pd.Series | None = None,
) -> pd.DataFrame:
    """M-hat in GWh of each month in `count` replicates of `replicate` on the hours of `targets`,
    one row each, drawn from the generator seeded with `seed`, with the DC/AC ratio of each
    replicate in a last column."""
    hours = pd.DatetimeIndex(pd.concat([inputs.arrays.covariates for inputs in years]).index)
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(count):
        drawn = draw(hours, rng, dc_ac_ratios)
        rows.append(replicate(years, technology, pv, drawn, targets).rename(None))
        rows[-1]["dc_ac_ratio"] = drawn.dc_ac_ratio
    table = pd.DataFrame(rows).reset_index(drop=True)
    table.index.name = "replicate"
    return table


def intervals(replicated: pd.DataFrame) -> pd.DataFrame:
    """The 90 % interval of each month and year of `replicated`, rows of replicates whose columns
    are months, `YYYY-MM`: the 5th and 95th percentiles of the replicates of a month, and of the
    sums of its months for a year, `YYYY`, in GWh."""
    months = replicated.drop(columns="dc_ac_ratio", errors="ignore")
    by_year = months.T.groupby(months.columns.str[:4]).sum().T
    both = pd.concat([months, by_year], axis="columns")
    low, high = PERCENTILES
    table = pd.DataFrame({"low_gwh": both.quantile(low), "high_gwh": both.quantile(high)})
    table.index.name = "period"
    return table


def _stationary(days: int, rng: np.random.Generator) -> npt.NDArray[np.int64]:
    """Positions of `days` draws among `days` consecutive days: the first starts a block at a
    uniform day, and each next one starts a new block with probability 1 / MEAN_BLOCK_DAYS, or
    else takes the day after the last one, wrapping around."""
    starts = rng.integers(days, size=days)
    new_block = rng.random(days) < 1 / MEAN_BLOCK_DAYS
    positions = np.empty(days, dtype=np.int64)
    positions[0] = starts[0]
    for draw_index in range(1, days):
        following = (positions[draw_index - 1] + 1) % days
        positions[draw_index] = starts[draw_index] if new_block[draw_index] else following
    return positions
