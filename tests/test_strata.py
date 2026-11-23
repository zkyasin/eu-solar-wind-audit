from collections.abc import Callable
from typing import NamedTuple

import numpy as np
import pandas as pd
import pytest
from numpy.typing import ArrayLike
from pandera.errors import SchemaError
from pandera.typing import DataFrame

from eu_solar_wind_audit import days, precompute, strata

CAPACITY_MW = 100.0
# Capacity factors 0.1 to 0.9.
EDGES = tuple(np.arange(1, 10) / 10)


def utc_hours(start: str, periods: int) -> pd.DatetimeIndex:
    return pd.date_range(start, periods=periods, freq="h", tz="UTC", unit="ns", name="time")


def utc_stamps(stamps: list[str]) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(stamps, tz="UTC", name="time").as_unit("ns")


def covariates(
    hours: pd.DatetimeIndex, sun_elevation_deg: ArrayLike = 30.0
) -> DataFrame[precompute.Covariates]:
    """Covariates of 100 MW on `hours`, with the given sun elevation."""
    frame = pd.DataFrame(
        {
            "complete": True,
            "capacity_mw": CAPACITY_MW,
            "capacity_provisional": False,
            "era5_source": "ERA5",
            "sun_elevation_deg": sun_elevation_deg,
            "ghi_w_m2": 500.0,
            "clear_sky_ghi_w_m2": 700.0,
        },
        index=hours,
    )
    return precompute.Covariates.validate(frame)


def test_elevation_bands_start_at_their_lower_edge_and_night_holds_zero() -> None:
    elevations = [-20.0, 0.0, 0.01, 9.99, 10.0, 19.99, 20.0, 39.99, 40.0, 49.99, 50.0, 70.0]
    hours = utc_hours("2021-06-01 00:00", len(elevations))

    found = strata.classify_hours(
        pd.Series(50.0, index=hours), covariates(hours, elevations), EDGES
    )

    assert found["elevation_band"].tolist() == [
        "night",
        "night",
        "0-10",
        "0-10",
        "10-20",
        "10-20",
        "20-30",
        "30-40",
        "40-50",
        "40-50",
        "50+",
        "50+",
    ]


def test_hours_below_or_above_the_edges_go_to_the_first_or_last_decile() -> None:
    # Capacity factors 0, 0.05, 0.1, ..., 1.3 on 100 MW.
    potential_mwh = [0.0, 5.0, 10.0, 15.0, 50.0, 85.0, 90.0, 130.0]
    hours = utc_hours("2021-06-01 08:00", len(potential_mwh))

    found = strata.classify_hours(pd.Series(potential_mwh, index=hours), covariates(hours), EDGES)

    # An hour on an edge goes to the decile above it.
    assert found["decile"].tolist() == [1, 1, 2, 2, 6, 9, 10, 10]


def test_hours_take_the_type_of_their_day_in_paris() -> None:
    # From Friday 3 June 2022 at 21:00 UTC, 23:00 in Paris: Saturday in Paris starts at 22:00 UTC.
    hours = utc_hours("2022-06-03 21:00", 30)

    found = strata.classify_hours(pd.Series(50.0, index=hours), covariates(hours), EDGES)

    pd.testing.assert_series_equal(found["day_type"], days.day_types(hours))
    expected = ["working_day"] + ["saturday"] * 24 + ["sunday_or_holiday"] * 5
    assert found["day_type"].tolist() == expected


def test_hours_without_elevation_or_capacity_factor_have_no_band_or_decile() -> None:
    hours = utc_hours("2021-06-01 10:00", 3)
    potential_mwh = pd.Series([50.0, 50.0, np.nan], index=hours)

    found = strata.classify_hours(potential_mwh, covariates(hours, [35.0, np.nan, 35.0]), EDGES)

    pd.testing.assert_series_equal(
        found["elevation_band"],
        pd.Series(["30-40", None, "30-40"], index=hours, name="elevation_band", dtype="str"),
    )
    pd.testing.assert_series_equal(
        found["decile"], pd.Series([6, 6, None], index=hours, name="decile", dtype="Int64")
    )


@pytest.mark.parametrize(
    "edges",
    [EDGES[:-1], (*EDGES, 0.95), (0.2, 0.1, *EDGES[2:])],
    ids=["eight", "ten", "decreasing"],
)
def test_nine_increasing_decile_edges_are_required(edges: tuple[float, ...]) -> None:
    hours = utc_hours("2021-06-01 10:00", 2)

    with pytest.raises(ValueError, match="9 increasing decile edges"):
        strata.classify_hours(pd.Series(50.0, index=hours), covariates(hours), edges)


def test_the_potential_must_be_on_the_hours_of_the_covariates() -> None:
    hours = utc_hours("2021-06-01 10:00", 2)

    with pytest.raises(ValueError, match="not on the hours of the covariates"):
        strata.classify_hours(pd.Series(50.0, index=hours[1:]), covariates(hours), EDGES)


class Training(NamedTuple):
    published_mwh: pd.Series
    potential_mwh: pd.Series
    covariates: pd.DataFrame
    eligible: pd.Series


def two_strata() -> Training:
    """Eligible hours of two strata on working days, all with a potential of 20 MWh, so in the
    last decile: three hours at 35 degrees with a raw ratio of 1.5, on one UTC day but two days in
    Paris, and two hours at 15 degrees with a raw ratio of 0.5, on one day. The last hour is not
    eligible."""
    hours = utc_stamps(
        [
            # 23:00 on Wednesday 15 June 2022 in Paris, then 00:00 and 01:00 on the Thursday.
            "2022-06-15 21:00",
            "2022-06-15 22:00",
            "2022-06-15 23:00",
            "2022-06-20 10:00",
            "2022-06-20 11:00",
            "2022-06-20 12:00",
        ]
    )
    return Training(
        published_mwh=pd.Series([30.0, 30.0, 30.0, 10.0, 10.0, 500.0], index=hours),
        potential_mwh=pd.Series([20.0, 20.0, 20.0, 20.0, 20.0, 90.0], index=hours),
        covariates=covariates(hours, [35.0, 35.0, 35.0, 15.0, 15.0, 35.0]),
        eligible=pd.Series([True] * 5 + [False], index=hours),
    )


# Pooled ratio of `two_strata`: (90 + 20) / (60 + 40).
POOLED = 1.1


def row(ratios: strata.StratumRatios, band: str, decile: int, day_type: str) -> pd.Series:
    table = ratios.table
    (index,) = np.flatnonzero(
        (table["elevation_band"] == band)
        & (table["decile"] == decile)
        & (table["day_type"] == day_type)
    )
    return table.iloc[index]


def test_raw_ratios_are_shrunk_by_the_distinct_days_in_paris() -> None:
    ratios = strata.fit_ratios(*two_strata())

    assert ratios.pooled_ratio == pytest.approx(POOLED)
    two_days = row(ratios, "30-40", 10, "working_day")
    assert (two_days["training_hours"], two_days["distinct_days"]) == (3, 2)
    assert two_days["raw_ratio"] == pytest.approx(1.5)
    assert two_days["shrunk_ratio"] == pytest.approx(2 / 9 * 1.5 + 7 / 9 * POOLED)
    one_day = row(ratios, "10-20", 10, "working_day")
    assert (one_day["training_hours"], one_day["distinct_days"]) == (2, 1)
    assert one_day["raw_ratio"] == pytest.approx(0.5)
    assert one_day["shrunk_ratio"] == pytest.approx(1 / 8 * 0.5 + 7 / 8 * POOLED)


def test_the_table_holds_every_stratum_once() -> None:
    ratios = strata.fit_ratios(*two_strata())

    pd.testing.assert_frame_equal(ratios.table[strata.STRATUM], strata.STRATA.to_frame(index=False))
    empty = ratios.table[ratios.table["training_hours"] == 0]
    assert len(empty) == 7 * 10 * 3 - 2
    assert empty["distinct_days"].eq(0).all()
    assert empty["raw_ratio"].isna().all()
    assert empty["shrunk_ratio"].to_numpy() == pytest.approx(POOLED)
    with pytest.raises(SchemaError, match="every_stratum"):
        strata.StratumTable.validate(ratios.table.iloc[1:])


def test_a_stratum_without_training_hours_takes_the_pooled_ratio() -> None:
    ratios = strata.fit_ratios(*two_strata())
    # Saturday 2 July 2022 at 35 degrees, then Monday 4 July at 60 and 35 degrees, and without
    # a sun elevation.
    hours = utc_stamps(
        ["2022-07-02 10:00", "2022-07-04 10:00", "2022-07-04 11:00", "2022-07-04 12:00"]
    )
    potential_mwh = pd.Series([30.0, 20.0, 40.0, 10.0], index=hours)

    found = strata.apply_ratios(
        ratios, potential_mwh, covariates(hours, [35.0, 60.0, 35.0, np.nan])
    )

    trained = row(ratios, "30-40", 10, "working_day")["shrunk_ratio"]
    expected = [30.0 * POOLED, 20.0 * POOLED, 40.0 * trained, np.nan]
    np.testing.assert_allclose(found, expected)
    assert found.index.equals(hours)


def test_decile_edges_come_from_the_eligible_hours_only() -> None:
    rng = np.random.default_rng(0)
    hours = utc_hours("2021-03-01 00:00", 200)
    potential_mwh = pd.Series(rng.uniform(0.0, 60.0, hours.size), index=hours)
    # Higher capacity factors on the hours that are not eligible.
    potential_mwh.iloc[100:] += 30.0
    eligible = pd.Series(np.arange(hours.size) < 100, index=hours)

    ratios = strata.fit_ratios(1.1 * potential_mwh, potential_mwh, covariates(hours), eligible)

    factor = potential_mwh.iloc[:100] / CAPACITY_MW
    assert list(ratios.decile_edges) == np.quantile(factor, np.arange(1, 10) / 10).tolist()
    assert ratios.table["training_hours"].sum() == 100


def published_off_the_hours(given: Training) -> Training:
    return given._replace(published_mwh=given.published_mwh.iloc[1:])


def eligibility_as_numbers(given: Training) -> Training:
    return given._replace(eligible=given.eligible.astype(float))


def nothing_eligible(given: Training) -> Training:
    return given._replace(eligible=given.eligible & False)


SECOND_HOUR = pd.Timestamp("2022-06-15 22:00", tz="UTC")


def eligible_without_published_energy(given: Training) -> Training:
    given.published_mwh.iloc[1] = np.nan
    return given


def eligible_without_potential(given: Training) -> Training:
    given.potential_mwh.iloc[1] = np.nan
    return given


def eligible_without_capacity(given: Training) -> Training:
    given.covariates.loc[SECOND_HOUR, "capacity_mw"] = 0.0
    return given


def eligible_without_sun_elevation(given: Training) -> Training:
    given.covariates.loc[SECOND_HOUR, "sun_elevation_deg"] = np.nan
    return given


def an_hour_of_2026(given: Training) -> Training:
    hours = pd.DatetimeIndex(given.eligible.index)
    later = hours.where(hours != hours[-1], pd.Timestamp("2026-01-01", tz="UTC"))
    return Training(
        given.published_mwh.set_axis(later),
        given.potential_mwh.set_axis(later),
        given.covariates.set_axis(later),
        given.eligible.set_axis(later),
    )


@pytest.mark.parametrize(
    ("change", "error"),
    [
        (published_off_the_hours, "not on the same hours"),
        (eligibility_as_numbers, "must be boolean"),
        (nothing_eligible, "no eligible hours"),
        (eligible_without_published_energy, "15 22:00:00.* has no published energy"),
        (eligible_without_potential, "has no potential"),
        (eligible_without_capacity, "has no capacity"),
        (eligible_without_sun_elevation, "has no sun elevation"),
        # Refused even when the hour is not eligible.
        (an_hour_of_2026, r"2026-01-01 00:00:00\+00:00 lies outside the training years 2019-2025"),
    ],
)
def test_fit_inputs_are_checked(change: Callable[[Training], Training], error: str) -> None:
    given = change(two_strata())

    with pytest.raises(ValueError, match=error):
        strata.fit_ratios(*given)


def test_the_annual_level_is_a_ratio_of_sums_over_the_eligible_hours() -> None:
    hours = utc_stamps(
        [
            "2019-03-01 10:00",
            "2019-07-01 10:00",
            "2019-07-01 11:00",
            "2020-05-01 10:00",
            "2020-05-01 11:00",
            "2021-05-01 10:00",
        ]
    )
    published_mwh = pd.Series([10.0, 60.0, 500.0, 30.0, np.nan, 20.0], index=hours)
    potential_mwh = pd.Series([20.0, 40.0, 1.0, 20.0, 10.0, 10.0], index=hours)
    eligible = pd.Series([True, True, False, True, False, False], index=hours)

    levels = strata.annual_levels(published_mwh, potential_mwh, eligible)

    # 70 / 60 in 2019, where the mean of the hourly ratios would be 1; no eligible hour in 2021.
    np.testing.assert_allclose(levels, [70 / 60, 1.5, np.nan])
    assert levels.index.tolist() == [2019, 2020, 2021]
    assert levels.index.name == "year"


def test_each_week_takes_the_level_of_the_other_weeks_of_its_year() -> None:
    hours = utc_stamps(
        [
            # 2020 is a leap year: its last week holds 30 and 31 December.
            "2020-12-29 23:00",
            "2020-12-30 00:00",
            "2020-12-31 23:00",
            # The first week of 2021 ends on 8 January at 00:00 UTC, so it holds the hour before,
            # already 8 January in Paris. The last week holds 31 December alone.
            "2021-01-01 12:00",
            "2021-01-07 23:00",
            "2021-01-08 00:00",
            "2021-01-08 01:00",
            "2021-12-30 12:00",
            "2021-12-31 12:00",
            "2022-03-01 10:00",
        ]
    )
    published_mwh = pd.Series(
        [5.0, 20.0, 10.0, 10.0, 6.0, 30.0, 1000.0, 8.0, 40.0, 10.0], index=hours
    )
    potential_mwh = pd.Series(
        [10.0, 10.0, 10.0, 20.0, 4.0, 20.0, 1.0, 10.0, 40.0, 10.0], index=hours
    )
    eligible = pd.Series(True, index=hours)
    eligible.iloc[6] = False

    levels = strata.cross_fitted_levels(published_mwh, potential_mwh, eligible)

    # Published and potential energy of the eligible hours of each week: in 2020, 5 and 10, then 30
    # and 20; in 2021, 16 and 24, 30 and 20, 8 and 10, then 40 and 40, out of 94 and 94. 2022 has
    # no other week with eligible hours.
    in_2020 = [30 / 20, 5 / 10, 5 / 10]
    in_2021 = [78 / 70, 78 / 70, 64 / 74, 64 / 74, 86 / 84, 54 / 54]
    np.testing.assert_allclose(levels, [*in_2020, *in_2021, np.nan])
    assert levels.index.equals(hours)


def levels_off_the_hours(given: Training) -> Training:
    return given._replace(potential_mwh=given.potential_mwh.iloc[1:])


def levels_in_paris_time(given: Training) -> Training:
    return Training(
        given.published_mwh.tz_convert("Europe/Paris"),
        given.potential_mwh.tz_convert("Europe/Paris"),
        given.covariates,
        given.eligible.tz_convert("Europe/Paris"),
    )


@pytest.mark.parametrize(
    ("change", "error"),
    [
        (levels_off_the_hours, "not on the same hours"),
        (levels_in_paris_time, "DatetimeIndex in UTC"),
        (eligibility_as_numbers, "must be boolean"),
        (nothing_eligible, "no eligible hours"),
        (eligible_without_published_energy, "has no published energy"),
        (eligible_without_potential, "has no potential"),
    ],
)
def test_level_inputs_are_checked(change: Callable[[Training], Training], error: str) -> None:
    given = change(two_strata())

    with pytest.raises(ValueError, match=error):
        strata.annual_levels(given.published_mwh, given.potential_mwh, given.eligible)
    with pytest.raises(ValueError, match=error):
        strata.cross_fitted_levels(given.published_mwh, given.potential_mwh, given.eligible)
