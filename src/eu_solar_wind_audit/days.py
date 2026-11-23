"""Days of mainland France in Europe/Paris time, as the calibrated potential uses them
(evaluation plan, "Two model objects" and Deviations): the type of each day for the strata, and
its sky class from the fleet clear-sky index.

Hours are UTC hour starts; each takes the type and the sky class of the local day it starts in.
"""

import numpy as np
import pandas as pd
from pandas.tseries.holiday import AbstractHolidayCalendar, EasterMonday, Holiday
from pandas.tseries.offsets import Day, Easter

from eu_solar_wind_audit import grid, precompute

TIME_ZONE = "Europe/Paris"

# The 11 public holidays of the Code du travail, article L3133-1. The two more days of Alsace and
# Moselle, Good Friday and 26 December, are left out.
PUBLIC_HOLIDAYS = AbstractHolidayCalendar(
    "France",
    rules=[
        Holiday("New Year's Day", month=1, day=1),
        EasterMonday,
        Holiday("Labour Day", month=5, day=1),
        Holiday("Victory in Europe Day", month=5, day=8),
        # 39 and 50 days after Easter Sunday.
        Holiday("Ascension Day", month=1, day=1, offset=[Easter(), Day(39)]),
        Holiday("Whit Monday", month=1, day=1, offset=[Easter(), Day(50)]),
        Holiday("National Day", month=7, day=14),
        Holiday("Assumption Day", month=8, day=15),
        Holiday("All Saints' Day", month=11, day=1),
        Holiday("Armistice Day", month=11, day=11),
        Holiday("Christmas Day", month=12, day=25),
    ],
)

DAY_TYPES = ["working_day", "saturday", "sunday_or_holiday"]

# Fleet clear-sky index from which a day is clear, and below which it is overcast; days in between
# are mixed. Chosen on 2019-2022 (evaluation plan, Deviations).
CLEAR_FROM = 0.9
OVERCAST_BELOW = 0.6


def day_types(hours: pd.DatetimeIndex) -> pd.Series:
    """Type of the Europe/Paris day of each of `hours`, in UTC: one of `DAY_TYPES`. A public
    holiday that falls on a Saturday is a holiday."""
    days = local_days(hours)
    sunday_or_holiday = (days.day_of_week == 6) | days.isin(PUBLIC_HOLIDAYS.holidays())
    types = np.select(
        [sunday_or_holiday, days.day_of_week == 5],
        ["sunday_or_holiday", "saturday"],
        default="working_day",
    )
    return pd.Series(types, index=hours, name="day_type", dtype="str")


def clear_sky_index(covariates: pd.DataFrame) -> "pd.Series[float]":
    """Fleet clear-sky index of each Europe/Paris day whose hours are all in `covariates`, a
    precompute.Covariates frame: the global over the clear-sky irradiance, each summed over the
    hours of the day. NaN for a day without clear-sky irradiance or with a missing value. Indexed
    by the day, as a naive date."""
    precompute.Covariates.validate(covariates)
    irradiance = covariates[["ghi_w_m2", "clear_sky_ghi_w_m2"]]
    by_day = irradiance.groupby(local_days(pd.DatetimeIndex(covariates.index)).rename("day"))
    sums = by_day.sum(skipna=False)
    days = pd.DatetimeIndex(sums.index)
    # 23, 24 or 25 hours.
    length = (days + pd.Timedelta(days=1)).tz_localize(TIME_ZONE) - days.tz_localize(TIME_ZONE)
    whole = by_day.size() == pd.Series(length / grid.HOUR, index=days)
    clear_sky = sums["clear_sky_ghi_w_m2"]
    index = sums["ghi_w_m2"] / clear_sky.where(clear_sky > 0)
    return index[whole].rename("clear_sky_index")


def sky_classes(covariates: pd.DataFrame) -> pd.Series:
    """Sky class of the Europe/Paris day of each hour of `covariates`, a precompute.Covariates
    frame, from the `clear_sky_index` of the day: clear from CLEAR_FROM, overcast below
    OVERCAST_BELOW, mixed in between. Null for the hours of a day without an index.

    Covariates come one UTC year at a time: consecutive years are joined first, so that the days
    across New Year are whole.
    """
    index = clear_sky_index(covariates)
    bins = [-np.inf, OVERCAST_BELOW, CLEAR_FROM, np.inf]
    labels = ["overcast", "mixed", "clear"]
    by_day = pd.cut(index, bins, right=False, labels=labels).astype("str")
    hours = pd.DatetimeIndex(covariates.index)
    classes = by_day.reindex(local_days(hours)).to_numpy()
    return pd.Series(classes, index=hours, name="sky_class", dtype="str")


def local_days(hours: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Europe/Paris day of each of `hours`, in UTC, as a naive date."""
    if str(hours.tz) != "UTC":
        raise ValueError("expected a DatetimeIndex in UTC")
    return hours.tz_convert(TIME_ZONE).tz_localize(None).normalize()
