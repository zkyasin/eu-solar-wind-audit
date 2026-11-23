import numpy as np
import pandas as pd
import pytest
from numpy.typing import ArrayLike

from eu_solar_wind_audit import days, dc_ac, precompute, regimes

CAPACITY_MW = 100.0
CENTROID = (46.0, 2.0)


def utc(stamp: str) -> pd.Timestamp:
    return pd.Timestamp(stamp, tz="UTC")


def paris_hours(first_day: str, end_day: str) -> pd.DatetimeIndex:
    """UTC hours of the Europe/Paris days from `first_day` to the day before `end_day`."""
    start, end = (pd.Timestamp(day, tz=days.TIMEZONE) for day in (first_day, end_day))
    hours = pd.date_range(start, end, freq="h", inclusive="left", unit="ns", name="time")
    return hours.tz_convert("UTC")


def sky_arrays(
    hours: pd.DatetimeIndex, clear_sky_index: ArrayLike, centroid: tuple[float, float] = CENTROID
) -> precompute.SolarArrays:
    """Arrays on `hours` with the sun at 60 degrees at 12:00 UTC and 10 degrees lower for each
    hour away from it, and the given clear-sky index in each hour. Their output is zero, which the
    sample does not read."""
    covariates = pd.DataFrame(
        {
            "complete": True,
            "capacity_mw": CAPACITY_MW,
            "capacity_provisional": False,
            "era5_source": "ERA5",
            "sun_elevation_deg": 60.0 - 10.0 * np.abs(hours.hour.to_numpy() - 12),
            "ghi_w_m2": 100.0 * np.asarray(clear_sky_index, dtype=float),
            "clear_sky_ghi_w_m2": 100.0,
            "air_temperature_c": 20.0,
        },
        index=hours,
    )
    return precompute.SolarArrays(
        fleet_mwh=np.zeros((hours.size, 1, precompute.DC_AC_GRID.size)),
        horizontal_mwh=np.zeros((hours.size, precompute.DC_AC_GRID.size)),
        shares=np.array([1.0]),
        covariates=precompute.Covariates.validate(covariates),
        centroid=centroid,
    )


def utc_hours(day: str, hours: list[int]) -> pd.DatetimeIndex:
    return pd.DatetimeIndex([utc(f"{day} {hour:02d}:00") for hour in hours], name="time")


def test_the_sample_keeps_clear_days_with_reference_prices_around_transit() -> None:
    # Four days of June 2022 in Paris. The sun stands at 20 degrees or more from 08:00 to 16:00
    # UTC, and solar transit at 2 E comes at 11:52 UTC: the hours from 09:00 to 14:00 UTC have
    # their middle within 3 h of it.
    hours = paris_hours("2022-06-14", "2022-06-18")
    clear_sky_index = days.local_days(hours).day.map({14: 0.95, 15: 0.95, 16: 0.8, 17: 0.95})
    arrays = sky_arrays(hours, clear_sky_index)
    prices = pd.Series(50.0, index=hours)
    # A near-zero price within 3 h of transit on 15 June, beyond them on 17 June.
    prices.loc[[utc("2022-06-15 10:00"), utc("2022-06-17 15:00")]] = 2.0
    classes = regimes.classify_hours(prices)
    eligible = classes["regime"] == "reference"
    # An hour left out by another rule leaves the rest of its day in the sample.
    eligible.loc[[utc("2022-06-14 12:00")]] = False

    sample = dc_ac.sample_hours([arrays], eligible, classes)

    assert sample.index.equals(hours)
    # 16 June is mixed.
    expected = utc_hours("2022-06-14", [8, 9, 10, 11, 13, 14, 15, 16]).append(
        utc_hours("2022-06-17", [8, 9, 10, 11, 12, 13, 14, 16])
    )
    assert sample.index[sample.to_numpy()].equals(expected)


def test_the_transit_is_that_of_the_centroid_of_each_array() -> None:
    first, second = paris_hours("2022-06-14", "2022-06-15"), paris_hours("2022-06-15", "2022-06-16")
    # Transit comes at 11:52 UTC at 2 E and an hour earlier at 17 E, where the middle of the hour
    # from 08:00 UTC lies within 3 h of it.
    arrays = [
        sky_arrays(first, 0.95, centroid=(46.0, 2.0)),
        sky_arrays(second, 0.95, centroid=(46.0, 17.0)),
    ]
    hours = first.append(second)
    prices = pd.Series(50.0, index=hours)
    prices.loc[[utc("2022-06-14 08:00"), utc("2022-06-15 08:00")]] = 2.0
    classes = regimes.classify_hours(prices)

    sample = dc_ac.sample_hours(arrays, classes["regime"] == "reference", classes)

    assert sample.index[sample.to_numpy()].equals(utc_hours("2022-06-14", list(range(9, 17))))


def test_the_sample_needs_inputs_on_the_hours_of_the_arrays() -> None:
    hours = paris_hours("2022-06-14", "2022-06-15")
    arrays = sky_arrays(hours, 0.95)
    classes = regimes.classify_hours(pd.Series(50.0, index=hours))
    eligible = classes["regime"] == "reference"

    with pytest.raises(ValueError, match="on the hours of the arrays"):
        dc_ac.sample_hours([arrays], eligible.iloc[1:], classes)
    with pytest.raises(ValueError, match="on the hours of the arrays"):
        dc_ac.sample_hours([arrays], eligible, classes.iloc[1:])
