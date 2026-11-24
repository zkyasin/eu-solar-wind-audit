import dataclasses

import numpy as np
import pandas as pd
import pytest
from numpy.typing import ArrayLike
from pandera.typing import DataFrame

from eu_solar_wind_audit import calibration, days, dc_ac, precompute, pv, regimes
from eu_solar_wind_audit.precompute import DC_AC_GRID, FloatArray
from eu_solar_wind_audit.pv import PVParameters, System

CAPACITY_MW = 100.0
CENTROID = (46.0, 2.0)
PV = PVParameters(fleet=(System(35.0, 180.0, 0.4), System(15.0, 110.0, 0.6)), dc_ac_ratio=1.2)
# Level, horizontal share and seasonal amplitude of the synthetic published output.
TRUTH = (0.85, 0.4, 0.04)
# DC output of the second configured system and of horizontal modules relative to the first
# system; horizontal modules gain on the others in summer, which tells their share apart from the
# level.
SECOND_SYSTEM = 0.85
HORIZONTAL = 0.75
HORIZONTAL_SUMMER_GAIN = 0.3

Case = tuple[list[precompute.SolarArrays], pd.Series, pd.Series]


def utc(stamp: str) -> pd.Timestamp:
    return pd.Timestamp(stamp, tz="UTC")


def paris_hours(first_day: str, end_day: str) -> pd.DatetimeIndex:
    """UTC hours of the Europe/Paris days from `first_day` to the day before `end_day`."""
    start, end = (pd.Timestamp(day, tz=days.TIMEZONE) for day in (first_day, end_day))
    hours = pd.date_range(start, end, freq="h", inclusive="left", unit="ns", name="time")
    return hours.tz_convert("UTC")


def sky_arrays(
    hours: pd.DatetimeIndex,
    clear_sky_index: ArrayLike,
    centroid: tuple[float, float] = CENTROID,
    air_temperature_c: ArrayLike = 20.0,
) -> precompute.SolarArrays:
    """Arrays on `hours` with the sun at 60 degrees at 12:00 UTC and 10 degrees lower for each
    hour away from it, and the given clear-sky index and air temperature in each hour. Their
    output is zero, which the sample and its subsets do not read."""
    covariates = pd.DataFrame(
        {
            "complete": True,
            "capacity_mw": CAPACITY_MW,
            "capacity_provisional": False,
            "era5_source": "ERA5",
            "sun_elevation_deg": 60.0 - 10.0 * np.abs(hours.hour.to_numpy() - 12),
            "ghi_w_m2": 100.0 * np.asarray(clear_sky_index, dtype=float),
            "clear_sky_ghi_w_m2": 100.0,
            "air_temperature_c": air_temperature_c,
            "day_type": days.day_types(hours, "FR").to_numpy(),
        },
        index=hours,
    )
    return precompute.SolarArrays(
        fleet_mwh=np.zeros((hours.size, 1, DC_AC_GRID.size)),
        horizontal_mwh=np.zeros((hours.size, DC_AC_GRID.size)),
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


def test_hours_are_near_transit_by_their_middle() -> None:
    # Transit comes at 12:20 UTC at 5 W: the hour from 09:00 UTC has its middle within 3 h of it
    # but not its start, and the hour from 15:00 UTC its start but not its middle.
    hours = paris_hours("2022-06-14", "2022-06-16")
    arrays = sky_arrays(hours, 0.95, centroid=(46.0, -5.0))
    prices = pd.Series(50.0, index=hours)
    prices.loc[[utc("2022-06-14 09:00"), utc("2022-06-15 15:00")]] = 2.0
    classes = regimes.classify_hours(prices)

    sample = dc_ac.sample_hours([arrays], classes["regime"] == "reference", classes)

    expected = utc_hours("2022-06-15", [8, 9, 10, 11, 12, 13, 14, 16])
    assert sample.index[sample.to_numpy()].equals(expected)


def test_the_sample_needs_inputs_on_the_hours_of_the_arrays() -> None:
    hours = paris_hours("2022-06-14", "2022-06-15")
    arrays = sky_arrays(hours, 0.95)
    classes = regimes.classify_hours(pd.Series(50.0, index=hours))
    eligible = classes["regime"] == "reference"

    with pytest.raises(ValueError, match="on the hours of the arrays"):
        dc_ac.sample_hours([arrays], eligible.iloc[1:], classes)
    with pytest.raises(ValueError, match="on the hours of the arrays"):
        dc_ac.sample_hours([arrays], eligible, classes.iloc[1:])


def noon_hours(days: list[str]) -> pd.DatetimeIndex:
    return pd.DatetimeIndex([utc(f"{day} 12:00") for day in days], name="time").as_unit("ns")


def selected_days(sample: pd.Series) -> list[str]:
    return list(pd.DatetimeIndex(sample.index)[sample.to_numpy()].strftime("%m-%d"))


def test_subsets_split_the_sample_by_air_temperature_and_season() -> None:
    hours = noon_hours(
        ["2022-02-15", "2022-03-15", "2022-04-15", "2022-07-15", "2022-07-16", "2022-10-15"]
    )
    arrays = sky_arrays(hours, 0.95, air_temperature_c=[5.0, 14.9, 15.0, 21.9, 22.0, np.nan])
    sample = pd.Series([True, True, True, True, False, True], index=hours)

    found = dc_ac.subsets([arrays], sample)

    assert {name: selected_days(hours) for name, hours in found.items()} == {
        "air_below_15_c": ["02-15", "03-15"],
        "air_15_to_22_c": ["04-15", "07-15"],
        "DJF": ["02-15"],
        "MAM": ["03-15", "04-15"],
        "JJA": ["07-15"],
        "SON": ["10-15"],
    }


def test_subsets_without_sample_hours_are_left_out() -> None:
    hours = noon_hours(["2022-01-15", "2022-06-15", "2022-06-16"])
    arrays = sky_arrays(hours, 0.95, air_temperature_c=[2.0, 25.0, 26.0])
    sample = pd.Series([False, True, True], index=hours)

    found = dc_ac.subsets([arrays], sample)

    assert list(found) == ["air_from_22_c", "JJA"]


def test_subsets_need_a_sample_on_the_hours_of_the_arrays() -> None:
    hours = noon_hours(["2022-06-15", "2022-06-16"])
    sample = pd.Series(True, index=hours[1:])

    with pytest.raises(ValueError, match="on the hours of the arrays"):
        dc_ac.subsets([sky_arrays(hours, 0.95)], sample)


def ac_output(dc: FloatArray, ratio: float) -> FloatArray:
    """AC output per unit of AC capacity of the DC output `dc` per unit of DC capacity."""
    return pv.ac_from_dc(dc, ratio, PV.dc_losses)


def covariates_on(hours: pd.DatetimeIndex) -> DataFrame[precompute.Covariates]:
    """Covariates of a clear and complete hour on each of `hours`."""
    covariates = pd.DataFrame(
        {
            "complete": True,
            "capacity_mw": CAPACITY_MW,
            "capacity_provisional": False,
            "era5_source": "ERA5",
            "sun_elevation_deg": 40.0,
            "ghi_w_m2": 600.0,
            "clear_sky_ghi_w_m2": 620.0,
            "air_temperature_c": 20.0,
            "day_type": days.day_types(hours, "FR").to_numpy(),
        },
        index=hours,
    )
    return precompute.Covariates.validate(covariates)


def clipping_case(
    true_ratio: float, dc_scale: float = 1.0, years: tuple[int, ...] = (2019, 2020)
) -> Case:
    """Arrays on the hours from 08:00 to 16:00 UTC of every seventh day of `years`, whose DC output
    follows the course of a clear day, and published output of the potential with TRUTH at
    `true_ratio`, with a noise of 0.002 in capacity factor; every hour is in the sample."""
    rng = np.random.default_rng(1)
    arrays, published = [], []
    for year in years:
        midnights = pd.date_range(f"{year}-01-03", f"{year}-12-31", freq="7D", tz="UTC")
        hours = pd.DatetimeIndex(
            [
                midnight + pd.Timedelta(hours=hour)
                for midnight in midnights
                for hour in range(8, 17)
            ],
            name="time",
        ).as_unit("ns")
        peak = np.repeat(rng.uniform(0.9, 1.05, midnights.size), 9)
        dc = dc_scale * peak * np.sin(np.pi * (hours.hour.to_numpy() + 0.5 - 5) / 14)
        summer = np.cos(2 * np.pi * (hours.day_of_year.to_numpy() - 172) / 365.25)
        horizontal = HORIZONTAL * (1 + HORIZONTAL_SUMMER_GAIN * summer) * dc
        systems = [dc, SECOND_SYSTEM * dc, horizontal]
        on_grid = [
            np.stack([ac_output(values, float(ratio)) for ratio in DC_AC_GRID], axis=-1)
            for values in systems
        ]
        shares = np.array([system.share for system in PV.fleet])
        arrays.append(
            precompute.SolarArrays(
                fleet_mwh=CAPACITY_MW * np.stack(on_grid[:-1], axis=1),
                horizontal_mwh=CAPACITY_MW * on_grid[-1],
                shares=shares,
                covariates=covariates_on(hours),
                centroid=CENTROID,
            )
        )
        fleet = sum(
            share * ac_output(values, true_ratio)
            for share, values in zip(shares, systems[:-1], strict=True)
        )
        potential = precompute.solar_potential(
            CAPACITY_MW * fleet,
            CAPACITY_MW * ac_output(systems[-1], true_ratio),
            precompute.seasonal_sine(hours),
            *TRUTH,
        )
        noise = rng.normal(0.0, 0.002 * CAPACITY_MW, hours.size)
        published.append(pd.Series(potential + noise, index=hours))
    published_mwh = pd.concat(published)
    return arrays, published_mwh, pd.Series(True, index=published_mwh.index)


def column(ratio: float) -> int:
    return int(np.flatnonzero(np.isclose(DC_AC_GRID, ratio))[0])


@pytest.mark.parametrize("true_ratio", [1.21, 1.37, 1.6])
def test_the_profile_finds_the_ratio_of_inverter_clipping(true_ratio: float) -> None:
    arrays, published_mwh, sample = clipping_case(true_ratio)

    found = dc_ac.profile(arrays, published_mwh, sample, PV)

    assert found.minimum == true_ratio
    assert not found.at_grid_end
    assert found.parameters[column(true_ratio)] == pytest.approx(TRUTH, abs=0.01)
    assert found.costs.shape == DC_AC_GRID.shape
    assert (found.hours, found.days) == (936, 104)


def test_ties_go_to_the_smallest_ratio() -> None:
    arrays, published_mwh, sample = clipping_case(1.37)
    # The ratios from 1.33 to 1.41 all get the output at 1.37.
    tied = slice(column(1.33), column(1.41) + 1)
    for part in arrays:
        part.fleet_mwh[:, :, tied] = part.fleet_mwh[:, :, [column(1.37)]]
        part.horizontal_mwh[:, tied] = part.horizontal_mwh[:, [column(1.37)]]

    found = dc_ac.profile(arrays, published_mwh, sample, PV)

    assert np.unique(found.costs[tied]).size == 1
    assert found.minimum == 1.33


@pytest.mark.parametrize(
    ("true_ratio", "dc_scale", "end"),
    [
        pytest.param(2.2, 1.0, 1.8, id="above-the-grid"),
        # A larger DC output lets inverters clip below the grid too.
        pytest.param(0.8, 1.6, 0.9, id="below-the-grid"),
    ],
)
def test_a_minimum_at_an_end_of_the_grid_is_flagged(
    true_ratio: float, dc_scale: float, end: float
) -> None:
    arrays, published_mwh, sample = clipping_case(true_ratio, dc_scale)

    found = dc_ac.profile(arrays, published_mwh, sample, PV)

    assert found.minimum == end
    assert found.at_grid_end


def test_the_scale_and_the_fit_at_the_configured_ratio_are_those_of_calibration() -> None:
    case, published_mwh, sample = clipping_case(1.37)
    # A capacity that changes from hour to hour, the energies unchanged, and a sample that leaves
    # out some hours of the arrays.
    rng = np.random.default_rng(2)
    arrays = [
        dataclasses.replace(
            part,
            covariates=precompute.Covariates.validate(
                part.covariates.assign(capacity_mw=rng.uniform(50.0, 150.0, len(part.covariates)))
            ),
        )
        for part in case
    ]
    sample[pd.DatetimeIndex(sample.index).hour == 12] = False

    found = dc_ac.profile(arrays, published_mwh, sample, PV)

    fitted = calibration.fit_solar(arrays, published_mwh, sample, PV)
    assert found.f_scale == fitted.fit.f_scale
    estimates = [fitted.level, fitted.horizontal_share, fitted.seasonal_amplitude]
    assert found.parameters[column(1.2)] == pytest.approx(estimates, abs=1e-7)
    assert found.costs[column(1.2)] == pytest.approx(fitted.fit.cost, rel=1e-9)


def test_only_sample_hours_are_profiled() -> None:
    arrays, published_mwh, sample = clipping_case(1.37)
    in_2019 = pd.DatetimeIndex(sample.index).year == 2019
    # Values of the other hours do not count.
    published_mwh[~in_2019] = 0.0

    found = dc_ac.profile(arrays, published_mwh, sample & in_2019, PV)

    # The same hours and values of 2019, without the other year.
    alone = dc_ac.profile(*clipping_case(1.37, years=(2019,)), PV)
    np.testing.assert_array_equal(found.costs, alone.costs)
    assert (found.minimum, found.hours, found.days) == (1.37, 468, 52)


def test_hours_of_2026_are_never_profiled() -> None:
    arrays, published_mwh, sample = clipping_case(1.37, years=(2025, 2026))

    with pytest.raises(ValueError, match="outside the training years"):
        dc_ac.profile(
            arrays, published_mwh, sample & (pd.DatetimeIndex(sample.index).year == 2025), PV
        )


def clear_day_course(hours: pd.DatetimeIndex) -> FloatArray:
    """DC output per unit of DC capacity over a clear day, from 05:00 to 19:00 UTC."""
    course: FloatArray = np.sin(np.pi * (hours.hour.to_numpy() + 0.5 - 5) / 14).clip(0.0)
    return course


def shared_course_arrays(
    covariates: DataFrame[precompute.Covariates], dc_scale: float = 1.0
) -> precompute.SolarArrays:
    """Arrays on the hours of `covariates` whose systems and horizontal modules all follow
    clear_day_course times `dc_scale`, so that the clipping of one day tells its ratio."""
    dc = dc_scale * clear_day_course(pd.DatetimeIndex(covariates.index))
    on_grid = np.stack([ac_output(dc, float(ratio)) for ratio in DC_AC_GRID], axis=-1)
    return precompute.SolarArrays(
        fleet_mwh=CAPACITY_MW * np.stack([on_grid, on_grid], axis=1),
        horizontal_mwh=CAPACITY_MW * on_grid,
        shares=np.array([system.share for system in PV.fleet]),
        covariates=covariates,
        centroid=CENTROID,
    )


def published_at(
    hours: pd.DatetimeIndex, ratio: float, seed: int, dc_scale: float = 1.0
) -> FloatArray:
    """Published output of shared_course_arrays with `dc_scale` at `ratio`, with the level of
    TRUTH and a noise of 0.002 in capacity factor."""
    output = TRUTH[0] * ac_output(dc_scale * clear_day_course(hours), ratio)
    noise = np.random.default_rng(seed).normal(0.0, 0.002, hours.size)
    published: FloatArray = CAPACITY_MW * (output + noise)
    return published


def two_clear_days(first_ratio: float, second_ratio: float) -> Case:
    """shared_course_arrays on the hours from 06:00 to 18:00 UTC of 14 and 15 June 2019, with
    output published at `first_ratio` on the first day and at `second_ratio` on the second."""
    hours = pd.DatetimeIndex(
        [utc(f"2019-06-{day} {hour:02d}:00") for day in (14, 15) for hour in range(6, 19)],
        name="time",
    ).as_unit("ns")
    first_day = hours.day.to_numpy() == 14
    published = np.where(
        first_day, published_at(hours, first_ratio, 3), published_at(hours, second_ratio, 3)
    )
    published_mwh = pd.Series(published, index=hours)
    return [shared_course_arrays(covariates_on(hours))], published_mwh, pd.Series(True, index=hours)


def test_the_bootstrap_draws_whole_days() -> None:
    arrays, published_mwh, sample = two_clear_days(1.3, 1.7)
    both = dc_ac.profile(arrays, published_mwh, sample, PV).minimum

    found = dc_ac.bootstrap(arrays, published_mwh, sample, PV, replicates=4, seed=1)

    # A replicate that draws one day twice finds the ratio of that day: drawn hours would mix the
    # two days.
    assert set(found.minima.tolist()) == {1.3, 1.7, both}
    ordered = np.sort(found.minima)
    # The 1st, 2nd and 4th smallest of 4 minima. The 2nd and 3rd differ, so that an interpolated
    # median would not be a minimum.
    assert ordered[1] != ordered[2]
    assert found.quantiles == (ordered[0], ordered[1], ordered[3])


def test_the_bootstrap_is_reproducible_with_its_seed() -> None:
    case = two_clear_days(1.3, 1.7)

    first = dc_ac.bootstrap(*case, PV, replicates=4)
    again = dc_ac.bootstrap(*case, PV, replicates=4)
    other = dc_ac.bootstrap(*case, PV, replicates=4, seed=1)

    assert first.seed == dc_ac.SEED
    np.testing.assert_array_equal(first.minima, again.minima)
    assert first.quantiles == again.quantiles
    assert not np.array_equal(first.minima, other.minima)


def test_the_report_states_each_minimum_at_0_05_with_the_sample_size() -> None:
    # Whole days in Paris, as sky classes need them, with air at 25 degC in 2019 and 18 degC in
    # 2020, and output published at a ratio of 1.33.
    arrays, published = [], []
    for year, temperature in [(2019, 25.0), (2020, 18.0)]:
        hours = paris_hours(f"{year}-06-14", f"{year}-06-15")
        covariates = sky_arrays(hours, 0.95, air_temperature_c=temperature).covariates
        arrays.append(shared_course_arrays(covariates))
        published.append(pd.Series(published_at(hours, 1.33, year), index=hours))
    published_mwh = pd.concat(published)
    classes = pd.DataFrame({"regime": "reference"}, index=published_mwh.index)
    eligible = pd.Series(True, index=published_mwh.index)

    found = dc_ac.report(arrays, published_mwh, eligible, classes, PV, replicates=3)

    # The sun is at 20 degrees or more from 08:00 to 16:00 UTC.
    years = pd.Index([2019, 2020], name="year")
    expected = pd.DataFrame({"hours": [9, 9], "days": [1, 1]}, index=years)
    pd.testing.assert_frame_equal(found.sample_size, expected)
    assert found.profile.minimum == 1.33
    assert list(found.subsets) == ["air_15_to_22_c", "air_from_22_c", "JJA"]
    assert found.bootstrap.minima.size == 3
    minima: pd.DataFrame = found.minima
    assert list(minima.index) == [
        "sample",
        *found.subsets,
        "bootstrap_5",
        "bootstrap_50",
        "bootstrap_95",
    ]
    # 1.33 (1 - 0.1408) is 1.143.
    assert minima.loc["sample"].to_dict() == {
        "hours": 18,
        "days": 2,
        "dc_ac_ratio": 1.33,
        "at_grid_end": False,
        "stated_ratio": 1.35,
        "stated_ratio_after_losses": 1.15,
    }
    assert minima.loc["bootstrap_50", ["hours", "days"]].to_dict() == {"hours": 18, "days": 2}
    for column, ratio in [
        ("stated_ratio", minima["dc_ac_ratio"]),
        ("stated_ratio_after_losses", minima["dc_ac_ratio"] * (1 - PV.dc_losses)),
    ]:
        steps = minima[column] / dc_ac.REPORT_STEP
        np.testing.assert_allclose(steps, steps.round(), atol=1e-9)
        assert ((minima[column] - ratio).abs() <= dc_ac.REPORT_STEP / 2).all()


def test_the_report_flags_minima_at_either_end_of_the_grid() -> None:
    # In 2019, air at 25 degC and output published at a ratio of 0.8, with a DC output 1.6 times
    # larger that clips below the grid; in 2020, air at 18 degC and output published at 2.2.
    arrays, published = [], []
    for year, temperature, ratio, dc_scale in [(2019, 25.0, 0.8, 1.6), (2020, 18.0, 2.2, 1.0)]:
        hours = paris_hours(f"{year}-06-14", f"{year}-06-15")
        covariates = sky_arrays(hours, 0.95, air_temperature_c=temperature).covariates
        arrays.append(shared_course_arrays(covariates, dc_scale))
        published.append(pd.Series(published_at(hours, ratio, year, dc_scale), index=hours))
    published_mwh = pd.concat(published)
    classes = pd.DataFrame({"regime": "reference"}, index=published_mwh.index)
    eligible = pd.Series(True, index=published_mwh.index)

    found = dc_ac.report(arrays, published_mwh, eligible, classes, PV, replicates=3)

    minima: pd.DataFrame = found.minima
    # 0.9 (1 - 0.1408) is 0.773 and 1.8 (1 - 0.1408) is 1.547.
    assert minima.loc["air_from_22_c"].to_dict() == {
        "hours": 9,
        "days": 1,
        "dc_ac_ratio": 0.9,
        "at_grid_end": True,
        "stated_ratio": 0.9,
        "stated_ratio_after_losses": 0.75,
    }
    assert minima.loc["air_15_to_22_c"].to_dict() == {
        "hours": 9,
        "days": 1,
        "dc_ac_ratio": 1.8,
        "at_grid_end": True,
        "stated_ratio": 1.8,
        "stated_ratio_after_losses": 1.55,
    }
