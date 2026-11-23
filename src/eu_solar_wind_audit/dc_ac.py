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
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import numpy.typing as npt
import pandas as pd
import pvlib

from eu_solar_wind_audit import calibration, days, precompute, pv
from eu_solar_wind_audit.precompute import DC_AC_GRID, BoolArray, FloatArray
from eu_solar_wind_audit.pv import PVParameters

IntArray = npt.NDArray[np.intp]

# Hours with the sun at 20 degrees or more (calibration design), on clear days whose hours within
# 3 h of solar noon all have the reference price (evaluation plan, Deviations).
SAMPLE_ELEVATION_DEG = 20.0
TRANSIT_WINDOW = pd.Timedelta(hours=3)


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


class _Sample(NamedTuple):
    """Sample hours, in the order of the arrays: the capacity factor of the whole fleet mounted as
    configured, `fleet`, and as horizontal modules, `horizontal`, at each ratio of DC_AC_GRID,
    shape (hour, ratio); the seasonal sine of precompute; the published capacity factor; and the
    number of the Europe/Paris day."""

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


def _profile(sample: _Sample, f_scale: float) -> Profile:
    costs = np.empty(DC_AC_GRID.size)
    fitted = np.empty((DC_AC_GRID.size, len(calibration.SOLAR_START)))
    for column in range(DC_AC_GRID.size):
        result = calibration.huber_pass(
            _residuals(sample, column),
            np.array(calibration.SOLAR_START),
            f_scale,
            calibration.SOLAR_LOWER,
            calibration.SOLAR_UPPER,
            calibration.SOLAR_X_SCALE,
        )
        costs[column], fitted[column] = result.cost, result.x
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
