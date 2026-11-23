"""Days of mainland France in Europe/Paris time, as the strata of the calibrated potential use
them (evaluation plan, "Two model objects"): the type of each day.

Hours are UTC hour starts; each takes the type of the local day it starts in.
"""

import numpy as np
import pandas as pd
from pandas.tseries.holiday import AbstractHolidayCalendar, EasterMonday, Holiday
from pandas.tseries.offsets import Day, Easter

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


def local_days(hours: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Europe/Paris day of each of `hours`, in UTC, as a naive date."""
    if str(hours.tz) != "UTC":
        raise ValueError("expected a DatetimeIndex in UTC")
    return hours.tz_convert(TIME_ZONE).tz_localize(None).normalize()
