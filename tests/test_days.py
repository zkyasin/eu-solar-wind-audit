import numpy as np
import pandas as pd
import pytest
from numpy.typing import ArrayLike

from eu_solar_wind_audit import days, evaluation

FIXED_HOLIDAYS = ["01-01", "05-01", "05-08", "07-14", "08-15", "11-01", "11-11", "12-25"]
# Easter Monday, Ascension Day and Whit Monday. Easter Sunday fell on 21 April 2019, 12 April
# 2020, 4 April 2021 and 17 April 2022.
EASTER_HOLIDAYS = {
    2019: ["04-22", "05-30", "06-10"],
    2020: ["04-13", "05-21", "06-01"],
    2021: ["04-05", "05-13", "05-24"],
    2022: ["04-18", "05-26", "06-06"],
}
GERMAN_FIXED_HOLIDAYS = ["01-01", "05-01", "10-03", "12-25", "12-26"]
# Good Friday, Easter Monday, Ascension Day and Whit Monday.
GERMAN_EASTER_HOLIDAYS = {
    2019: ["04-19", "04-22", "05-30", "06-10"],
    2020: ["04-10", "04-13", "05-21", "06-01"],
    2021: ["04-02", "04-05", "05-13", "05-24"],
    2022: ["04-15", "04-18", "05-26", "06-06"],
}


def utc_hours(start: str, end: str) -> pd.DatetimeIndex:
    return pd.date_range(start, end, freq="h", inclusive="left", tz="UTC", unit="ns", name="time")


def paris_hours(first_day: str, end_day: str) -> pd.DatetimeIndex:
    """UTC hours of the Europe/Paris days from `first_day` to the day before `end_day`."""
    start, end = (pd.Timestamp(day, tz=days.TIMEZONE) for day in (first_day, end_day))
    hours = pd.date_range(start, end, freq="h", inclusive="left", unit="ns", name="time")
    return hours.tz_convert("UTC")


def covariates(
    hours: pd.DatetimeIndex, ghi: ArrayLike, clear_sky_ghi: ArrayLike = 100.0
) -> pd.DataFrame:
    """precompute.Covariates on `hours` with the given global and clear-sky irradiance."""
    return pd.DataFrame(
        {
            "complete": True,
            "capacity_mw": 100.0,
            "capacity_provisional": False,
            "era5_source": "ERA5",
            "sun_elevation_deg": 10.0,
            "ghi_w_m2": ghi,
            "clear_sky_ghi_w_m2": clear_sky_ghi,
            "air_temperature_c": 20.0,
        },
        index=hours,
    )


def sky(hours: pd.DatetimeIndex, classes: list[str | None]) -> pd.Series:
    return pd.Series(classes, index=hours, name="sky_class", dtype="str")


@pytest.mark.parametrize("year", [2019, 2020, 2021, 2022])
def test_the_eleven_public_holidays_of_a_year(year: int) -> None:
    holidays = days.PUBLIC_HOLIDAYS["FR"].holidays(f"{year}-01-01", f"{year}-12-31")

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

    types = days.day_types(hours, "FR")

    assert types.index.equals(hours)
    assert types.tolist() == [day_type for day_type, count in expected for _ in range(count)]
    assert set(types) <= set(days.DAY_TYPES)


@pytest.mark.parametrize("year", [2019, 2020, 2021, 2022])
def test_the_nine_public_holidays_of_every_german_land(year: int) -> None:
    holidays = days.PUBLIC_HOLIDAYS["DE"].holidays(f"{year}-01-01", f"{year}-12-31")

    expected = sorted(GERMAN_FIXED_HOLIDAYS + GERMAN_EASTER_HOLIDAYS[year])
    assert holidays.strftime("%m-%d").tolist() == expected


@pytest.mark.parametrize(
    ("year", "easter_holidays"),
    [
        (2019, ["04-18", "04-19", "04-22", "05-17", "05-30", "06-10"]),
        (2023, ["04-06", "04-07", "04-10", "05-05", "05-18", "05-29"]),
        (2024, ["03-28", "03-29", "04-01", "05-09", "05-20"]),
        (2025, ["04-17", "04-18", "04-21", "05-29", "06-09"]),
    ],
    ids=["2019", "2023 last store bededag", "2024", "2025"],
)
def test_danish_public_holidays(year: int, easter_holidays: list[str]) -> None:
    holidays = days.PUBLIC_HOLIDAYS["DK1"].holidays(f"{year}-01-01", f"{year}-12-31")

    assert holidays.strftime("%m-%d").tolist() == ["01-01", *easter_holidays, "12-25", "12-26"]
    assert days.PUBLIC_HOLIDAYS["DK2"] is days.PUBLIC_HOLIDAYS["DK1"]


def test_each_country_has_its_own_holidays() -> None:
    # Thursday 14 July 2022 is the French National Day, Monday 3 October German Unity Day and
    # Thursday 6 April 2023 Maundy Thursday.
    hours = pd.DatetimeIndex(["2022-07-14 10:00", "2022-10-03 10:00", "2023-04-06 10:00"], tz="UTC")

    assert days.day_types(hours, "FR").tolist() == ["sunday_or_holiday", *["working_day"] * 2]
    assert days.day_types(hours, "DE").tolist() == [
        "working_day",
        "sunday_or_holiday",
        "working_day",
    ]
    assert days.day_types(hours, "DK1").tolist() == [*["working_day"] * 2, "sunday_or_holiday"]


def test_day_types_need_utc_hours() -> None:
    hours = pd.date_range("2022-03-26", periods=3, freq="h", tz="Europe/Paris")

    with pytest.raises(ValueError, match="UTC"):
        days.day_types(hours, "FR")


@pytest.mark.parametrize(
    ("ghi", "sky_class"),
    [
        (120.0, "clear"),
        (90.0, "clear"),
        (89.99, "mixed"),
        (60.0, "mixed"),
        (59.99, "overcast"),
        (0.0, "overcast"),
    ],
)
def test_sky_class_edges(ghi: float, sky_class: str) -> None:
    hours = paris_hours("2022-06-15", "2022-06-16")

    classes = days.sky_classes(covariates(hours, ghi))

    pd.testing.assert_series_equal(classes, sky(hours, [sky_class] * 24))
    assert sky_class in evaluation.SKY_CLASSES


def test_only_whole_local_days_get_a_sky_class() -> None:
    # Saturday 26 to Monday 28 March 2022 in Paris, with an hour of the days before and after.
    # The Sunday has 23 hours.
    hours = utc_hours("2022-03-25 22:00", "2022-03-28 23:00")
    day = days.local_days(hours)
    ghi = day.day.map({25: 95.0, 26: 95.0, 27: 70.0, 28: 30.0, 29: 95.0})

    found = covariates(hours, ghi)

    index = days.clear_sky_index(found)
    assert index.index.equals(pd.DatetimeIndex(["2022-03-26", "2022-03-27", "2022-03-28"]))
    np.testing.assert_allclose(index, [0.95, 0.70, 0.30])
    expected = [None] + ["clear"] * 24 + ["mixed"] * 23 + ["overcast"] * 24 + [None]
    pd.testing.assert_series_equal(days.sky_classes(found), sky(hours, expected))


def test_a_day_with_a_missing_hour_gets_no_sky_class() -> None:
    hours = paris_hours("2022-03-26", "2022-03-28")
    kept = hours.drop([pd.Timestamp("2022-03-27 12:00", tz="UTC")])

    classes = days.sky_classes(covariates(kept, 95.0))

    pd.testing.assert_series_equal(classes, sky(kept, ["clear"] * 24 + [None] * 22))


def test_a_day_without_clear_sky_irradiance_gets_no_sky_class() -> None:
    hours = paris_hours("2022-06-15", "2022-06-17")
    clear_sky_ghi = np.where(hours < pd.Timestamp("2022-06-15 22:00", tz="UTC"), 0.0, 100.0)

    # Global irradiance on the first day too, whose ratio of sums would be infinite.
    found = covariates(hours, 50.0, clear_sky_ghi)

    assert np.isnan(days.clear_sky_index(found).iloc[0])
    pd.testing.assert_series_equal(
        days.sky_classes(found), sky(hours, [None] * 24 + ["overcast"] * 24)
    )


def test_a_day_with_a_missing_irradiance_gets_no_sky_class() -> None:
    hours = paris_hours("2022-06-15", "2022-06-17")
    ghi = np.full(hours.size, 95.0)
    ghi[30] = np.nan

    classes = days.sky_classes(covariates(hours, ghi))

    pd.testing.assert_series_equal(classes, sky(hours, ["clear"] * 24 + [None] * 24))


def test_years_are_joined_to_classify_new_year_day() -> None:
    # 1 January 2022 in Paris starts at 23:00 UTC on 31 December.
    last_year = covariates(utc_hours("2021-12-31 00:00", "2022-01-01 00:00"), 95.0)
    this_year = covariates(utc_hours("2022-01-01 00:00", "2022-01-02 00:00"), 95.0)

    alone = days.sky_classes(this_year)
    joined = days.sky_classes(pd.concat([last_year, this_year]))

    assert alone.isna().all()
    new_year_day = utc_hours("2021-12-31 23:00", "2022-01-01 23:00")
    assert joined[joined.notna()].index.equals(new_year_day)
    assert set(joined.dropna()) == {"clear"}
