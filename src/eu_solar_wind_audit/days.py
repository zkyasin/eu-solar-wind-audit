"""Days of mainland France, Germany, the Danish price areas and the Netherlands, as the calibrated
potential uses them (evaluation plan, "Two model objects" and Deviations): the type of each day for
the strata, and its sky class from the fleet clear-sky index.

The four countries keep Central European Time, so their days are those of Europe/Paris. Hours are
UTC hour starts; each takes the type and the sky class of the local day it starts in.
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from pandas.tseries.holiday import AbstractHolidayCalendar, EasterMonday, GoodFriday, Holiday
from pandas.tseries.offsets import Day, Easter

from eu_solar_wind_audit import grid

TIMEZONE = "Europe/Paris"

# 39 and 50 days after Easter Sunday.
ASCENSION_DAY = Holiday("Ascension Day", month=1, day=1, offset=[Easter(), Day(39)])
WHIT_MONDAY = Holiday("Whit Monday", month=1, day=1, offset=[Easter(), Day(50)])

# The 11 public holidays of the Code du travail, article L3133-1. Alsace and Moselle have two
# more, Good Friday and 26 December, which are left out.
FRANCE = AbstractHolidayCalendar(
    "France",
    rules=[
        Holiday("New Year's Day", month=1, day=1),
        EasterMonday,
        Holiday("Labour Day", month=5, day=1),
        Holiday("Victory in Europe Day", month=5, day=8),
        ASCENSION_DAY,
        WHIT_MONDAY,
        Holiday("National Day", month=7, day=14),
        Holiday("Assumption Day", month=8, day=15),
        Holiday("All Saints' Day", month=11, day=1),
        Holiday("Armistice Day", month=11, day=11),
        Holiday("Christmas Day", month=12, day=25),
    ],
)
# The 9 public holidays of every Land. The others hold in some Länder only, as Corpus Christi and
# All Saints' Day in the south and west, and are left out.
GERMANY = AbstractHolidayCalendar(
    "Germany",
    rules=[
        Holiday("New Year's Day", month=1, day=1),
        GoodFriday,
        EasterMonday,
        Holiday("Labour Day", month=5, day=1),
        ASCENSION_DAY,
        WHIT_MONDAY,
        Holiday("German Unity Day", month=10, day=3),
        Holiday("Christmas Day", month=12, day=25),
        Holiday("Second Day of Christmas", month=12, day=26),
    ],
)
# The public holidays of the Danish Church, which the Holidays Act protects. Store bededag, the
# fourth Friday after Easter, ceased to be one on 1 January 2024. Constitution Day and Christmas
# Eve are days off under many agreements, not public holidays, and are left out.
DENMARK = AbstractHolidayCalendar(
    "Denmark",
    rules=[
        Holiday("New Year's Day", month=1, day=1),
        Holiday("Maundy Thursday", month=1, day=1, offset=[Easter(), Day(-3)]),
        GoodFriday,
        EasterMonday,
        Holiday("Store bededag", month=1, day=1, offset=[Easter(), Day(26)], end_date="2023-12-31"),
        ASCENSION_DAY,
        WHIT_MONDAY,
        Holiday("Christmas Day", month=12, day=25),
        Holiday("Second Day of Christmas", month=12, day=26),
    ],
)


def _sunday_to_saturday(day: datetime) -> datetime:
    return day - timedelta(days=1) if day.weekday() == 6 else day


# The public holidays of the Algemene termijnenwet, article 3. King's Day moves to the Saturday
# before when 27 April is a Sunday. 5 May is a day off under many agreements only every fifth
# year; it is kept every year, as the law lists it. Good Friday is not among them.
NETHERLANDS = AbstractHolidayCalendar(
    "Netherlands",
    rules=[
        Holiday("New Year's Day", month=1, day=1),
        EasterMonday,
        Holiday(
            "King's Day",
            month=4,
            day=27,
            observance=_sunday_to_saturday,
            start_date="2014-01-01",
        ),
        Holiday("Liberation Day", month=5, day=5),
        ASCENSION_DAY,
        WHIT_MONDAY,
        Holiday("Christmas Day", month=12, day=25),
        Holiday("Second Day of Christmas", month=12, day=26),
    ],
)
# Keyed by the zones that calibration and detection runs name.
PUBLIC_HOLIDAYS = {
    "FR": FRANCE,
    "DE": GERMANY,
    "DK1": DENMARK,
    "DK2": DENMARK,
    "NL": NETHERLANDS,
}

DAY_TYPES = ["working_day", "saturday", "sunday_or_holiday"]

# Fleet clear-sky index from which a day is clear, and below which it is overcast; days in between
# are mixed. Chosen on 2019-2022 (evaluation plan, Deviations).
CLEAR_FROM = 0.9
OVERCAST_BELOW = 0.6


def day_types(hours: pd.DatetimeIndex, country: str) -> pd.Series:
    """Type of the local day of each of `hours`, in UTC, in `country`, a key of PUBLIC_HOLIDAYS:
    one of `DAY_TYPES`. A public holiday that falls on a Saturday is a holiday."""
    days = local_days(hours)
    holidays = PUBLIC_HOLIDAYS[country].holidays()
    sunday_or_holiday = (days.day_of_week == 6) | days.isin(holidays)
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
    irradiance = covariates[["ghi_w_m2", "clear_sky_ghi_w_m2"]]
    by_day = irradiance.groupby(local_days(pd.DatetimeIndex(covariates.index)).rename("day"))
    sums = by_day.sum(skipna=False)
    days = pd.DatetimeIndex(sums.index)
    # 23, 24 or 25 hours.
    length = (days + pd.Timedelta(days=1)).tz_localize(TIMEZONE) - days.tz_localize(TIMEZONE)
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
    return hours.tz_convert(TIMEZONE).tz_localize(None).normalize()
