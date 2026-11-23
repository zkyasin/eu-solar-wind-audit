import pandas as pd
import pytest

from eu_solar_wind_audit import days

FIXED_HOLIDAYS = ["01-01", "05-01", "05-08", "07-14", "08-15", "11-01", "11-11", "12-25"]
# Easter Monday, Ascension Day and Whit Monday. Easter Sunday fell on 21 April 2019, 12 April
# 2020, 4 April 2021 and 17 April 2022.
EASTER_HOLIDAYS = {
    2019: ["04-22", "05-30", "06-10"],
    2020: ["04-13", "05-21", "06-01"],
    2021: ["04-05", "05-13", "05-24"],
    2022: ["04-18", "05-26", "06-06"],
}


def utc_hours(start: str, end: str) -> pd.DatetimeIndex:
    return pd.date_range(start, end, freq="h", inclusive="left", tz="UTC", unit="ns", name="time")


@pytest.mark.parametrize("year", [2019, 2020, 2021, 2022])
def test_the_eleven_public_holidays_of_a_year(year: int) -> None:
    holidays = days.PUBLIC_HOLIDAYS.holidays(f"{year}-01-01", f"{year}-12-31")

    expected = sorted(FIXED_HOLIDAYS + EASTER_HOLIDAYS[year])
    assert holidays.strftime("%m-%d").tolist() == expected
    assert holidays.year.unique().tolist() == [year]


@pytest.mark.parametrize(
    ("start", "end", "expected"),
    [
        # Clocks go forward on Sunday 27 March 2022 at 01:00 UTC: that day has 23 hours, from
        # 23:00 UTC on Saturday to 22:00 UTC.
        pytest.param(
            "2022-03-26 21:00",
            "2022-03-27 23:00",
            [("saturday", 2), ("sunday_or_holiday", 23), ("working_day", 1)],
            id="spring-clock-change",
        ),
        # Clocks go back on Sunday 30 October 2022, which has 25 hours. Tuesday 1 November is All
        # Saints' Day.
        pytest.param(
            "2022-10-29 21:00",
            "2022-11-02 00:00",
            [
                ("saturday", 1),
                ("sunday_or_holiday", 25),
                ("working_day", 24),
                ("sunday_or_holiday", 24),
                ("working_day", 1),
            ],
            id="autumn-clock-change",
        ),
        # New Year's Day 2022 is a Saturday: the holiday wins.
        pytest.param(
            "2021-12-31 21:00",
            "2022-01-03 00:00",
            [("working_day", 2), ("sunday_or_holiday", 48), ("working_day", 1)],
            id="new-year",
        ),
    ],
)
def test_hours_take_the_type_of_their_day_in_paris(
    start: str, end: str, expected: list[tuple[str, int]]
) -> None:
    hours = utc_hours(start, end)

    types = days.day_types(hours)

    assert types.index.equals(hours)
    assert types.tolist() == [day_type for day_type, count in expected for _ in range(count)]
    assert set(types) <= set(days.DAY_TYPES)


def test_day_types_need_utc_hours() -> None:
    hours = pd.date_range("2022-03-26", periods=3, freq="h", tz="Europe/Paris")

    with pytest.raises(ValueError, match="UTC"):
        days.day_types(hours)
