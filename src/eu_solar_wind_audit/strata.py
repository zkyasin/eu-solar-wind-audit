"""Strata and annual level of the calibrated potential (evaluation plan, "Two model objects",
item 2, and Deviations): the published output over the potential is estimated in each stratum of
hours, then over each year.

The stratum of an hour is the band of its sun elevation, the decile of the capacity factor of the
potential and the type of its Europe/Paris day (days.day_types). The elevation is the
`sun_elevation_deg` covariate of precompute: capacity-weighted for solar, at the capacity centroid
of the fleet for onshore wind. The decile edges are the 10 % to 90 % quantiles of the capacity
factor over the eligible hours of the training years; hours below or above them go to the first
or last decile. The published output never defines a stratum, since it depends on modulation.

Over the eligible hours of the training years, the raw ratio of a stratum is the published energy
of its hours over their potential, and the pooled ratio the same over all these hours. The shrunk
ratio of a stratum is

    w * raw ratio + (1 - w) * pooled ratio,  w = n / (n + 7),

with n the number of distinct Europe/Paris days with eligible hours in the stratum; a stratum
without such hours takes the pooled ratio. The potential with strata is the potential times the
shrunk ratio of the stratum of each hour.

The annual level of a UTC year is the published energy over the potential with strata, both summed
over the eligible hours of the year. Cross-fitted within the year, the hours of a week take the
level of the other weeks of their year.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import pandas as pd
import pandera.pandas as pa
from pandera.typing import DataFrame, Index, Series

from eu_solar_wind_audit import days, evaluation, grid, precompute
from eu_solar_wind_audit.precompute import BoolArray

# Sun elevation in degrees: the night band up to 0, then a band from each edge to the next, the
# last without an upper edge.
ELEVATION_EDGES = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]
ELEVATION_BANDS = ["night", "0-10", "10-20", "20-30", "30-40", "40-50", "50+"]
DECILES = list(range(1, 11))
# Quantile levels of the 9 decile edges.
DECILE_LEVELS = np.arange(1, 10) / 10
STRATUM = ["elevation_band", "decile", "day_type"]
STRATA = pd.MultiIndex.from_product([ELEVATION_BANDS, DECILES, days.DAY_TYPES], names=STRATUM)
# Weight n / (n + SHRINKAGE_DAYS) of the raw ratio of a stratum, n counting its distinct days:
# the hours of a day are not independent (evaluation plan, Deviations).
SHRINKAGE_DAYS = 7
# Weeks of the cross-fitted level: blocks of 7 days from 1 January 00:00 UTC.
WEEK_DAYS = 7


class HourStrata(pa.DataFrameModel):
    """Stratum of the hour starting at `time`: the band of its sun elevation, the decile of the
    capacity factor of the potential, from 1 to 10, and the type of its day. The band is null
    without a sun elevation, the decile without a capacity factor."""

    time: Index[grid.UtcTime] = pa.Field(unique=True, check_name=True)
    elevation_band: Series[str] = pa.Field(isin=ELEVATION_BANDS, nullable=True)
    decile: Series[pd.Int64Dtype] = pa.Field(isin=DECILES, nullable=True)
    day_type: Series[str] = pa.Field(isin=days.DAY_TYPES)

    class Config:
        strict = True


class StratumTable(pa.DataFrameModel):
    """Ratios of each stratum, one row per combination of elevation band, decile and day type:
    the eligible hours of the training years in the stratum, their distinct Europe/Paris days, the
    raw ratio of their published energy to their potential, null without such hours, and the
    shrunk ratio."""

    elevation_band: Series[str] = pa.Field(isin=ELEVATION_BANDS)
    decile: Series[int] = pa.Field(isin=DECILES)
    day_type: Series[str] = pa.Field(isin=days.DAY_TYPES)
    training_hours: Series[int] = pa.Field(ge=0)
    distinct_days: Series[int] = pa.Field(ge=0)
    raw_ratio: Series[float] = pa.Field(nullable=True)
    shrunk_ratio: Series[float]

    @pa.dataframe_check
    @classmethod
    def every_stratum(cls, frame: pd.DataFrame) -> bool:
        return len(frame) == len(STRATA)

    class Config:
        strict = True
        unique: ClassVar[list[str]] = STRATUM


@dataclass(frozen=True)
class StratumRatios:
    """Strata fitted on the eligible hours of the training years: the 9 decile edges of the
    capacity factor of the potential, the pooled ratio and the ratios of each stratum."""

    decile_edges: tuple[float, ...]
    pooled_ratio: float
    table: DataFrame[StratumTable]


def classify_hours(
    potential_mwh: pd.Series, covariates: pd.DataFrame, decile_edges: Sequence[float]
) -> DataFrame[HourStrata]:
    """Stratum of each hour of `covariates`, a precompute.Covariates frame, for the potential
    `potential_mwh` in MWh on the same hours and the 9 increasing `decile_edges`. An hour on an
    edge goes to the band or decile above it, except at 0 degrees, which is night."""
    precompute.Covariates.validate(covariates)
    hours = pd.DatetimeIndex(covariates.index)
    if not potential_mwh.index.equals(hours):
        raise ValueError("the potential is not on the hours of the covariates")
    edges = np.asarray(decile_edges, dtype=float)
    if edges.shape != DECILE_LEVELS.shape or (np.diff(edges) < 0).any():
        raise ValueError(f"expected {DECILE_LEVELS.size} increasing decile edges")

    elevation = covariates["sun_elevation_deg"].to_numpy()
    band = np.where(elevation <= 0, 0, np.searchsorted(ELEVATION_EDGES, elevation, side="right"))
    factor = (potential_mwh / covariates["capacity_mw"]).to_numpy()
    decile = np.searchsorted(edges, factor, side="right") + 1
    strata = pd.DataFrame(
        {
            "elevation_band": pd.Series(
                np.take(ELEVATION_BANDS, band), index=hours, dtype="str"
            ).where(~np.isnan(elevation)),
            "decile": pd.Series(decile, index=hours, dtype="Int64").where(~np.isnan(factor)),
            "day_type": days.day_types(hours),
        }
    )
    return HourStrata.validate(strata)


def fit_ratios(
    published_mwh: pd.Series,
    potential_mwh: pd.Series,
    covariates: pd.DataFrame,
    eligible: pd.Series,
) -> StratumRatios:
    """Strata fitted on the eligible hours of `covariates`, a precompute.Covariates frame of the
    training years of a fold, which need not be consecutive. `published_mwh`, the potential
    `potential_mwh` in MWh and the boolean `eligible` are on the hours of `covariates`.

    Raises ValueError if an hour lies outside evaluation.TRAINING_YEARS, if the inputs are not on
    the same hours, or if an eligible hour has no published energy, potential, capacity or sun
    elevation.
    """
    precompute.Covariates.validate(covariates)
    hours = pd.DatetimeIndex(covariates.index)
    outside = hours[~hours.year.isin(evaluation.TRAINING_YEARS)]
    if not outside.empty:
        first, last = evaluation.TRAINING_YEARS[0], evaluation.TRAINING_YEARS[-1]
        raise ValueError(f"{outside[0]} lies outside the training years {first}-{last}")
    capacity_mw = covariates["capacity_mw"]
    training = _eligible_hours(
        eligible,
        {
            "published energy": published_mwh,
            "potential": potential_mwh,
            "capacity": capacity_mw.where(capacity_mw > 0),
            "sun elevation": covariates["sun_elevation_deg"],
        },
    )

    factor = (potential_mwh / capacity_mw)[training]
    edges = tuple(float(edge) for edge in np.quantile(factor, DECILE_LEVELS))
    strata = classify_hours(potential_mwh, covariates, edges)[training]
    by_stratum = strata.assign(
        published_mwh=published_mwh[training],
        potential_mwh=potential_mwh[training],
        day=days.local_days(pd.DatetimeIndex(strata.index)),
    ).groupby(STRATUM)
    sums = pd.DataFrame(
        {
            "training_hours": by_stratum.size(),
            "distinct_days": by_stratum["day"].nunique(),
            "published_mwh": by_stratum["published_mwh"].sum(),
            "potential_mwh": by_stratum["potential_mwh"].sum(),
        }
    ).reindex(STRATA, fill_value=0)
    pooled = float(published_mwh[training].sum() / potential_mwh[training].sum())
    raw = sums["published_mwh"] / sums["potential_mwh"]
    weight = sums["distinct_days"] / (sums["distinct_days"] + SHRINKAGE_DAYS)
    shrunk = (weight * raw + (1 - weight) * pooled).where(sums["training_hours"] > 0, pooled)
    table = sums[["training_hours", "distinct_days"]].assign(raw_ratio=raw, shrunk_ratio=shrunk)
    return StratumRatios(edges, pooled, StratumTable.validate(table.reset_index()))


def apply_ratios(
    ratios: StratumRatios, potential_mwh: pd.Series, covariates: pd.DataFrame
) -> "pd.Series[float]":
    """Potential with strata: the potential `potential_mwh` in MWh on the hours of `covariates`, a
    precompute.Covariates frame, times the shrunk ratio of the stratum of each hour. NaN where the
    hour has no stratum."""
    strata = classify_hours(potential_mwh, covariates, ratios.decile_edges)
    shrunk = ratios.table.set_index(STRATUM)["shrunk_ratio"]
    ratio = shrunk.reindex(pd.MultiIndex.from_frame(strata[STRATUM])).to_numpy()
    return pd.Series(potential_mwh.to_numpy() * ratio, index=strata.index, name="energy_mwh")


def annual_levels(
    published_mwh: pd.Series, potential_mwh: pd.Series, eligible: pd.Series
) -> "pd.Series[float]":
    """Annual level of each UTC year of the hours of `eligible`: the published energy over the
    potential with strata of `apply_ratios`, both in MWh and summed over the eligible hours of the
    year. NaN for a year without eligible hours. Indexed by year.

    Raises ValueError unless the inputs are on the same UTC hours, `eligible` is boolean with an
    eligible hour, and every eligible hour has both energies.
    """
    sums = _eligible_sums(published_mwh, potential_mwh, eligible)
    by_year = sums.groupby(pd.DatetimeIndex(sums.index).year.rename("year")).sum()
    return (by_year["published_mwh"] / by_year["potential_mwh"]).rename("annual_level")


def cross_fitted_levels(
    published_mwh: pd.Series, potential_mwh: pd.Series, eligible: pd.Series
) -> "pd.Series[float]":
    """Annual level of each hour of `eligible` cross-fitted within its UTC year: the level of
    `annual_levels` over the eligible hours of the other weeks of the year. Weeks are blocks of 7
    days from 1 January 00:00 UTC; the last block of a year has 1 or 2 days. NaN where the other
    weeks have no eligible hours.

    Raises ValueError as `annual_levels` does.
    """
    sums = _eligible_sums(published_mwh, potential_mwh, eligible)
    hours = pd.DatetimeIndex(sums.index)
    weeks = [hours.year, (hours.day_of_year - 1) // WEEK_DAYS]
    by_week = sums.groupby(weeks).sum()
    other_weeks = by_week.groupby(level=0).transform("sum") - by_week
    level = other_weeks["published_mwh"] / other_weeks["potential_mwh"]
    by_hour = level.reindex(pd.MultiIndex.from_arrays(weeks)).to_numpy()
    return pd.Series(by_hour, index=hours, name="level")


def _eligible_sums(
    published_mwh: pd.Series, potential_mwh: pd.Series, eligible: pd.Series
) -> pd.DataFrame:
    """Both energies at the eligible hours, zero at the others, after the checks of
    `annual_levels`."""
    hours = eligible.index
    if not isinstance(hours, pd.DatetimeIndex) or str(hours.tz) != "UTC":
        raise ValueError("expected a DatetimeIndex in UTC")
    mask = _eligible_hours(
        eligible, {"published energy": published_mwh, "potential": potential_mwh}
    )
    return pd.DataFrame(
        {
            "published_mwh": published_mwh.where(mask, 0.0),
            "potential_mwh": potential_mwh.where(mask, 0.0),
        }
    )


def _eligible_hours(eligible: pd.Series, values: dict[str, pd.Series]) -> BoolArray:
    """Mask of the eligible hours. Raises ValueError unless `eligible` is boolean, with at least
    one eligible hour, and each series of `values` is on its hours, with a value at every eligible
    hour."""
    hours = pd.DatetimeIndex(eligible.index)
    if not all(series.index.equals(hours) for series in values.values()):
        raise ValueError("the inputs are not on the same hours")
    if eligible.dtype != bool:
        raise ValueError("eligibility must be boolean")
    mask: BoolArray = eligible.to_numpy()
    if not mask.any():
        raise ValueError("no eligible hours")
    for name, series in values.items():
        missing = hours[mask & series.isna().to_numpy()]
        if not missing.empty:
            raise ValueError(f"eligible hour {missing[0]} has no {name}")
    return mask
