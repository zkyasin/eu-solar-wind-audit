"""DC/AC ratio of the solar fleet (evaluation plan, "Two model objects", item 1, and Deviations):
the ratios that the clear-sky profile of the calibrated potential allows. The result is reported
only: the calibrated potential keeps the configured ratio.

Sample: the eligible hours whose capacity-weighted sun elevation is SAMPLE_ELEVATION_DEG or more,
on the clear days of days.sky_classes whose hours with mid-hour within TRANSIT_WINDOW of solar
transit all have the reference price class. The transit is that of the Europe/Paris day, from
pvlib (SPA) at the capacity centroid of the fleet over the period of each array. Whole days keep
the sample from leaning towards morning and evening hours where prices take out sunny peak hours.

Profile: with c the scale of calibration.fit_solar on the sample hours, at the configured ratio,
x_r minimises the cost F(x; c) of calibration at each ratio r of precompute.DC_AC_GRID, in one
pass of calibration.huber_pass from calibration.SOLAR_START, and cost(r) = F(x_r; c). The minimum
is the smallest ratio with the lowest cost, flagged when it is an end of the grid.

Subsets: the profile, each with its own scale, of the sample hours in each band of
capacity-weighted air temperature, where warm modules lose output at peak hours as clipping does,
and in each season of the UTC month.

Bootstrap: each of REPLICATES replicates draws the days of the sample with replacement, from a
generator seeded with SEED, and holds every hour of a drawn day as many times as the day is drawn.
Its minimum comes from its profile with the scale of the sample, each ratio of the grid starting
from the solution at the ratio before it. The QUANTILES of the minima are minima of replicates:
for 999 replicates, the 50th, 500th and 950th smallest.

Report: the sample hours and days of each UTC year, and each minimum stated at REPORT_STEP, as the
ratio r and as r (1 - L) with L the DC losses of the PV parameters: away from clipping hours, the
data identify the level times r (1 - L), not r.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import numpy.typing as npt
import pandas as pd
import pandera.pandas as pa
import pvlib
from pandera.typing import DataFrame, Index, Series

from eu_solar_wind_audit import calibration, days, evaluation, precompute, pv
from eu_solar_wind_audit.precompute import DC_AC_GRID, BoolArray, FloatArray
from eu_solar_wind_audit.pv import PVParameters

IntArray = npt.NDArray[np.intp]

# Hours with the sun at 20 degrees or more (calibration design), on clear days whose hours within
# 3 h of solar noon all have the reference price (evaluation plan, Deviations).
SAMPLE_ELEVATION_DEG = 20.0
TRANSIT_WINDOW = pd.Timedelta(hours=3)
# Bootstrap over days (evaluation plan, Deviations), with the 999 replicates of the inference of
# the plan and a fixed seed.
REPLICATES = 999
SEED = 20261123
QUANTILES = (0.05, 0.5, 0.95)
# The profile does not locate the ratio more precisely (calibration critique).
REPORT_STEP = 0.05
# Bands [low, high) of capacity-weighted air temperature in degC (calibration critique).
AIR_TEMPERATURE_BANDS = {
    "air_below_15_c": (-np.inf, 15.0),
    "air_15_to_22_c": (15.0, 22.0),
    "air_from_22_c": (22.0, np.inf),
}


@dataclass(frozen=True)
class Profile:
    """Profile on `hours` hours of `days` Europe/Paris days with the scale `f_scale`: at each ratio
    DC_AC_GRID[i], `costs[i]` is the cost F(x_i; f_scale) and `parameters[i]` the level,
    horizontal share and seasonal amplitude x_i. `minimum` is the smallest ratio with the lowest
    cost, and `at_grid_end` tells whether it is the first or the last ratio of the grid."""

    costs: FloatArray
    parameters: FloatArray
    f_scale: float
    minimum: float
    at_grid_end: bool
    hours: int
    days: int


@dataclass(frozen=True)
class Bootstrap:
    """Minima of the profiles of `minima.size` replicates drawn with the generator seeded with
    `seed`, and their QUANTILES."""

    minima: FloatArray
    quantiles: tuple[float, ...]
    seed: int


class SampleSize(pa.DataFrameModel):
    """Sample hours and their Europe/Paris days in each UTC year with sample hours."""

    year: Index[int] = pa.Field(unique=True, check_name=True)
    hours: Series[int] = pa.Field(gt=0)
    days: Series[int] = pa.Field(gt=0)

    class Config:
        strict = True


class Minima(pa.DataFrameModel):
    """The minimum of the profile of the sample (estimate "sample") and of each of its subsets,
    and the QUANTILES of the bootstrap minima (estimates "bootstrap_5", "bootstrap_50" and
    "bootstrap_95"), with the hours and days of the sample they come from. `dc_ac_ratio` is on
    DC_AC_GRID and `at_grid_end` tells whether it is an end of the grid; `stated_ratio` is the
    ratio at REPORT_STEP, and `stated_ratio_after_losses` the ratio times (1 - DC losses) at
    REPORT_STEP."""

    estimate: Index[str] = pa.Field(unique=True, check_name=True)
    hours: Series[int] = pa.Field(gt=0)
    days: Series[int] = pa.Field(gt=0)
    dc_ac_ratio: Series[float]
    at_grid_end: Series[bool]
    stated_ratio: Series[float]
    stated_ratio_after_losses: Series[float]

    class Config:
        strict = True


@dataclass(frozen=True)
class Report:
    """The size of the sample, its profile, the profiles of its subsets, the bootstrap over its
    days, and their minima stated at REPORT_STEP."""

    sample_size: DataFrame[SampleSize]
    profile: Profile
    subsets: dict[str, Profile]
    bootstrap: Bootstrap
    minima: DataFrame[Minima]


class _Sample(NamedTuple):
    """Sample hours, in the order of the arrays: the capacity factor of the whole fleet mounted as
    configured, `fleet`, and as horizontal modules, `horizontal`, at each ratio of DC_AC_GRID,
    shape (hour, ratio); the seasonal sine of precompute; the published capacity factor; and the
    Europe/Paris day, numbered from 0 in time order."""

    fleet: FloatArray
    horizontal: FloatArray
    sine: FloatArray
    published: FloatArray
    day: IntArray


def sample_hours(
    arrays: Sequence[precompute.SolarArrays], eligible: pd.Series, price_classes: pd.DataFrame
) -> pd.Series:
    """Whether each hour of `arrays` is in the sample of the module docstring.

    `arrays` cover distinct periods in time order, such as consecutive UTC years, so that the days
    across their ends are whole. `eligible`, the eligible column of eligibility.Eligibility, and
    `price_classes`, the classes of regimes.classify_hours, are on the hours of the arrays, in the
    same order.
    """
    covariates = pd.concat([part.covariates for part in arrays])
    hours = pd.DatetimeIndex(covariates.index)
    if not (eligible.index.equals(hours) and price_classes.index.equals(hours)):
        raise ValueError("eligibility and price classes must be on the hours of the arrays")
    day = days.local_days(hours)
    near_transit = np.concatenate([_near_transit(part) for part in arrays])
    off_reference = (price_classes["regime"] != "reference").to_numpy()
    in_sample = (
        eligible.to_numpy()
        & (covariates["sun_elevation_deg"] >= SAMPLE_ELEVATION_DEG).to_numpy()
        & (days.sky_classes(covariates) == "clear").to_numpy()
        & ~day.isin(day[near_transit & off_reference])
    )
    return pd.Series(in_sample, index=hours, name="sample")


def subsets(arrays: Sequence[precompute.SolarArrays], sample: pd.Series) -> dict[str, pd.Series]:
    """The hours of `arrays` where the boolean `sample` is true, split by the capacity-weighted air
    temperature of the hour into AIR_TEMPERATURE_BANDS, then by the season of its UTC month into
    evaluation.SEASONS, as boolean Series on the hours of the arrays. A subset without sample
    hours, such as winter can be, is left out."""
    covariates = pd.concat([part.covariates for part in arrays])
    if not sample.index.equals(covariates.index):
        raise ValueError("the sample must be on the hours of the arrays")
    temperature = covariates["air_temperature_c"]
    month = pd.Series(pd.DatetimeIndex(covariates.index).month, index=covariates.index)
    season = month.map(evaluation.SEASON_OF_MONTH)
    bands = {
        name: (temperature >= low) & (temperature < high)
        for name, (low, high) in AIR_TEMPERATURE_BANDS.items()
    }
    seasons = {name: season == name for name in evaluation.SEASONS}
    split = {name: sample & hours for name, hours in (bands | seasons).items()}
    return {name: hours for name, hours in split.items() if hours.any()}


def _near_transit(arrays: precompute.SolarArrays) -> BoolArray:
    """Whether the middle of each hour of `arrays` lies within TRANSIT_WINDOW of the solar transit
    of its Europe/Paris day at the capacity centroid of the fleet."""
    hours = pd.DatetimeIndex(arrays.covariates.index)
    # pvlib gives each stamp the transit of its local day.
    transit = pvlib.solarposition.sun_rise_set_transit_spa(
        hours.tz_convert(days.TIMEZONE), *arrays.centroid
    )["transit"]
    offset = hours + pv.HALF_HOUR - pd.DatetimeIndex(transit).tz_convert("UTC")
    near: BoolArray = (offset >= -TRANSIT_WINDOW) & (offset <= TRANSIT_WINDOW)
    return near


def profile(
    arrays: Sequence[precompute.SolarArrays],
    published_mwh: pd.Series,
    sample: pd.Series,
    parameters: PVParameters,
) -> Profile:
    """Profile of the module docstring on the hours of `arrays` where the boolean `sample` is
    true, as sample_hours gives it. `published_mwh` is on the hours of the arrays, in the same
    order, and `parameters` are the PV parameters that the arrays were computed with.

    Raises ValueError as calibration.fit_solar does.
    """
    # The two passes at the configured ratio also check the inputs.
    f_scale = calibration.fit_solar(arrays, published_mwh, sample, parameters).fit.f_scale
    return _profile(_sample(arrays, published_mwh, sample), f_scale)


def bootstrap(
    arrays: Sequence[precompute.SolarArrays],
    published_mwh: pd.Series,
    sample: pd.Series,
    parameters: PVParameters,
    *,
    replicates: int = REPLICATES,
    seed: int = SEED,
) -> Bootstrap:
    """Bootstrap of the module docstring over the days of the hours of `arrays` where `sample` is
    true, from the inputs of `profile`.

    Raises ValueError as calibration.fit_solar does.
    """
    f_scale = calibration.fit_solar(arrays, published_mwh, sample, parameters).fit.f_scale
    base = _sample(arrays, published_mwh, sample)
    day_count = np.unique(base.day).size
    generator = np.random.default_rng(seed)
    minima = np.empty(replicates)
    for replicate in range(replicates):
        draws = np.bincount(generator.integers(day_count, size=day_count), minlength=day_count)
        rows = np.repeat(np.arange(base.day.size), draws[base.day])
        drawn = _Sample(
            fleet=base.fleet[rows],
            horizontal=base.horizontal[rows],
            sine=base.sine[rows],
            published=base.published[rows],
            day=base.day[rows],
        )
        minima[replicate] = _profile(drawn, f_scale, warm=True).minimum
    quantiles = np.quantile(minima, QUANTILES, method="inverted_cdf")
    return Bootstrap(minima=minima, quantiles=tuple(quantiles.tolist()), seed=seed)


def report(
    arrays: Sequence[precompute.SolarArrays],
    published_mwh: pd.Series,
    eligible: pd.Series,
    price_classes: pd.DataFrame,
    parameters: PVParameters,
    *,
    replicates: int = REPLICATES,
    seed: int = SEED,
) -> Report:
    """Report of the module docstring on the sample of `arrays`, from the inputs of sample_hours
    and profile."""
    sample = sample_hours(arrays, eligible, price_classes)
    sample_profile = profile(arrays, published_mwh, sample, parameters)
    subset_profiles = {
        name: profile(arrays, published_mwh, hours, parameters)
        for name, hours in subsets(arrays, sample).items()
    }
    resampled = bootstrap(
        arrays, published_mwh, sample, parameters, replicates=replicates, seed=seed
    )
    estimates = {
        name: (found.hours, found.days, found.minimum)
        for name, found in {"sample": sample_profile, **subset_profiles}.items()
    }
    for quantile, ratio in zip(QUANTILES, resampled.quantiles, strict=True):
        estimates[f"bootstrap_{round(100 * quantile)}"] = (
            sample_profile.hours,
            sample_profile.days,
            ratio,
        )
    minima = pd.DataFrame.from_dict(
        estimates, orient="index", columns=["hours", "days", "dc_ac_ratio"]
    ).rename_axis("estimate")
    minima["at_grid_end"] = minima["dc_ac_ratio"].isin([DC_AC_GRID[0], DC_AC_GRID[-1]])
    minima["stated_ratio"] = _stated(minima["dc_ac_ratio"])
    minima["stated_ratio_after_losses"] = _stated(
        minima["dc_ac_ratio"] * (1 - parameters.dc_losses)
    )
    return Report(
        sample_size=sample_size(sample),
        profile=sample_profile,
        subsets=subset_profiles,
        bootstrap=resampled,
        minima=Minima.validate(minima),
    )


def sample_size(sample: pd.Series) -> DataFrame[SampleSize]:
    """The hours where the boolean `sample` is true, and their Europe/Paris days, counted in each
    UTC year."""
    hours = pd.DatetimeIndex(sample.index)[sample.to_numpy()]
    year = pd.Index(hours.year, dtype="int64", name="year")
    by_year = pd.Series(days.local_days(hours), index=hours).groupby(year)
    return SampleSize.validate(pd.DataFrame({"hours": by_year.size(), "days": by_year.nunique()}))


def _stated(ratio: pd.Series) -> pd.Series:
    """`ratio` rounded to the nearest multiple of REPORT_STEP."""
    return ((ratio / REPORT_STEP).round() * REPORT_STEP).round(2)


def _sample(
    arrays: Sequence[precompute.SolarArrays], published_mwh: pd.Series, sample: pd.Series
) -> _Sample:
    selected = sample.to_numpy()
    covariates = pd.concat([part.covariates for part in arrays])
    hours = pd.DatetimeIndex(covariates.index)[selected]
    capacity_mw = covariates["capacity_mw"].to_numpy()[selected]
    fleet = np.concatenate([np.einsum("hsr,s->hr", part.fleet_mwh, part.shares) for part in arrays])
    horizontal = np.concatenate([part.horizontal_mwh for part in arrays])
    _, day = np.unique(days.local_days(hours), return_inverse=True)
    return _Sample(
        fleet=fleet[selected] / capacity_mw[:, np.newaxis],
        horizontal=horizontal[selected] / capacity_mw[:, np.newaxis],
        sine=precompute.seasonal_sine(hours),
        published=published_mwh.to_numpy()[selected] / capacity_mw,
        day=day,
    )


def _profile(sample: _Sample, f_scale: float, warm: bool = False) -> Profile:
    """Profile on the hours of `sample`, which can repeat. Each ratio starts from
    calibration.SOLAR_START or, when `warm`, from the solution at the ratio before it."""
    costs = np.empty(DC_AC_GRID.size)
    fitted = np.empty((DC_AC_GRID.size, len(calibration.SOLAR_START)))
    start = np.array(calibration.SOLAR_START)
    for column in range(DC_AC_GRID.size):
        result = calibration.huber_pass(
            _residuals(sample, column),
            start,
            f_scale,
            calibration.SOLAR_LOWER,
            calibration.SOLAR_UPPER,
            calibration.SOLAR_X_SCALE,
        )
        costs[column], fitted[column] = result.cost, result.x
        if warm:
            start = result.x
    # argmin keeps the first of equal costs, so ties go to the smallest ratio.
    lowest = int(np.argmin(costs))
    return Profile(
        costs=costs,
        parameters=fitted,
        f_scale=f_scale,
        minimum=float(DC_AC_GRID[lowest]),
        at_grid_end=lowest in (0, DC_AC_GRID.size - 1),
        hours=sample.day.size,
        days=np.unique(sample.day).size,
    )


def _residuals(sample: _Sample, column: int) -> Callable[[FloatArray], FloatArray]:
    """Residuals r_t(x) of calibration on the sample hours at the ratio DC_AC_GRID[column]."""
    fleet, horizontal = sample.fleet[:, column], sample.horizontal[:, column]

    def residuals(x: FloatArray) -> FloatArray:
        level, horizontal_share, seasonal_amplitude = x
        # The potential is linear in the output of the fleet, so it takes capacity factors.
        potential = precompute.solar_potential(
            fleet, horizontal, sample.sine, level, horizontal_share, seasonal_amplitude
        )
        return potential - sample.published

    return residuals
