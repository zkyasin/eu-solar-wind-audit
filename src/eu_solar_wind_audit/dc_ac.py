"""DC/AC ratio of the solar fleet (evaluation plan, "Two model objects", item 1, and Deviations):
the ratios that the clear-sky profile of the calibrated potential allows. The result is reported
only: the calibrated potential keeps the configured ratio.

Sample: the eligible hours whose capacity-weighted sun elevation is SAMPLE_ELEVATION_DEG or more,
on the clear days of days.sky_classes whose hours with mid-hour within TRANSIT_WINDOW of solar
transit all have the reference price class. The transit is that of the Europe/Paris day, from
pvlib (SPA) at the capacity centroid of the fleet over the period of each array. Whole days keep
the sample from leaning towards morning and evening hours where prices take out sunny peak hours.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd
import pvlib

from eu_solar_wind_audit import days, precompute, pv
from eu_solar_wind_audit.precompute import BoolArray

# Hours with the sun at 20 degrees or more (calibration design), on clear days whose hours within
# 3 h of solar noon all have the reference price (evaluation plan, Deviations).
SAMPLE_ELEVATION_DEG = 20.0
TRANSIT_WINDOW = pd.Timedelta(hours=3)


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
