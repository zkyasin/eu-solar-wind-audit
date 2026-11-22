from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest
from pandera.errors import SchemaError, SchemaErrors
from pandera.typing import Series

from eu_solar_wind_audit.grid import (
    Hourly,
    check_period,
    hourly_from_instantaneous,
    hourly_from_period_end_means,
    hourly_from_period_start_energy,
    on_hours,
)

Conversion = Callable[[pd.Series], pd.DataFrame]


def series(start: str, values: list[float], freq: str = "15min", tz: str = "UTC") -> pd.Series:
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq=freq, tz=tz))


def hourly_frame(
    energy_mwh: list[float], complete: list[bool], start: str = "2025-06-01 00:00"
) -> pd.DataFrame:
    time = pd.date_range(start, periods=len(complete), freq="h", tz="UTC", unit="ns", name="time")
    return pd.DataFrame({"energy_mwh": energy_mwh, "complete": complete}, index=time)


def utc(stamp: str) -> pd.Timestamp:
    return pd.Timestamp(stamp, tz="UTC")


@pytest.mark.parametrize(("period", "per_hour"), [("30min", 2), ("15min", 4), ("1h", 1)])
def test_constant_power_gives_power_times_one_hour(period: str, per_hour: int) -> None:
    power = series("2025-06-01 00:00", [250.0] * (3 * per_hour + 1), freq=period)

    hourly = hourly_from_period_end_means(power.iloc[1:], period=period)

    assert hourly["energy_mwh"].tolist() == [250.0] * 3
    assert hourly["complete"].all()


def test_period_ending_on_the_hour_belongs_to_the_previous_hour() -> None:
    power = series("2025-06-01 00:30", [100.0, 300.0, 500.0, 700.0], freq="30min")

    hourly = hourly_from_period_end_means(power)

    assert hourly.index.equals(series("2025-06-01 00:00", [0.0, 0.0], freq="h").index)
    assert hourly["energy_mwh"].tolist() == [200.0, 600.0]


def test_half_hour_values_on_a_quarter_hour_export_are_read() -> None:
    # éCO2mix layout: a 15-minute grid with production filled every other row.
    power = series("2025-06-01 00:15", [np.nan, 100.0, np.nan, 300.0, np.nan])

    hourly = hourly_from_period_end_means(power)

    assert hourly["energy_mwh"].tolist() == [200.0]
    assert hourly["complete"].tolist() == [True]


def test_a_missing_half_hour_leaves_the_hour_incomplete() -> None:
    power = series("2025-06-01 00:30", [100.0, np.nan, 500.0, 700.0], freq="30min")

    hourly = hourly_from_period_end_means(power)

    assert np.isnan(hourly["energy_mwh"].iloc[0])
    assert hourly["complete"].tolist() == [False, True]


def test_repeated_end_stamps_are_dropped_without_choosing_one() -> None:
    power = series("2025-06-01 00:30", [100.0, 300.0, 500.0, 700.0], freq="30min")
    power = pd.concat([power, power.iloc[[0]]])

    hourly = hourly_from_period_end_means(power)

    assert hourly["complete"].tolist() == [False, True]
    assert np.isnan(hourly["energy_mwh"].iloc[0])


def test_a_value_between_half_hours_is_rejected() -> None:
    power = series("2025-06-01 00:15", [5.0, 100.0, np.nan, 300.0])

    with pytest.raises(ValueError, match="grid"):
        hourly_from_period_end_means(power)


def test_end_labelled_input_must_be_in_utc() -> None:
    power = series("2025-06-01 00:30", [1.0] * 4, freq="30min", tz="Europe/Paris")

    with pytest.raises(ValueError, match="UTC"):
        hourly_from_period_end_means(power)


def test_five_points_span_exactly_one_hour() -> None:
    power = series("2025-06-01 00:00", [80.0] * 5)

    hourly = hourly_from_instantaneous(power)

    assert hourly.index.equals(series("2025-06-01 00:00", [0.0], freq="h").index)
    assert hourly["energy_mwh"].tolist() == [80.0]
    assert hourly["complete"].tolist() == [True]


def test_trapezoid_is_exact_for_a_linear_ramp() -> None:
    hours = np.arange(9) * 0.25
    power = series("2025-06-01 00:00", list(100.0 + 400.0 * hours))

    hourly = hourly_from_instantaneous(power)

    # Integral of 100 + 400 t over [0, 1] and [1, 2].
    assert hourly["energy_mwh"].tolist() == pytest.approx([300.0, 700.0])


def test_boundary_points_count_half_in_each_hour() -> None:
    power = series("2025-06-01 00:00", [0.0, 0.0, 0.0, 0.0, 400.0, 0.0, 0.0, 0.0, 0.0])

    hourly = hourly_from_instantaneous(power)

    assert hourly["energy_mwh"].tolist() == [50.0, 50.0]


def test_inner_points_count_one_quarter_hour() -> None:
    power = series("2025-06-01 00:00", [0.0, 0.0, 400.0, 0.0, 0.0])

    hourly = hourly_from_instantaneous(power)

    assert hourly["energy_mwh"].tolist() == [100.0]


@pytest.mark.parametrize("missing", range(5))
def test_any_missing_point_leaves_the_hour_incomplete(missing: int) -> None:
    power = series("2025-06-01 00:00", [80.0] * 9)
    power.iloc[missing] = np.nan

    hourly = hourly_from_instantaneous(power)

    assert np.isnan(hourly["energy_mwh"].iloc[0])
    assert not hourly["complete"].iloc[0]
    # The point at 01:00 also bounds the second hour.
    assert hourly["complete"].iloc[1] == (missing != 4)


def test_a_dropped_stamp_leaves_the_hour_incomplete() -> None:
    power = series("2025-06-01 00:00", [80.0] * 5).drop(pd.Timestamp("2025-06-01 00:30", tz="UTC"))

    hourly = hourly_from_instantaneous(power)

    assert hourly["complete"].tolist() == [False]


def test_instantaneous_input_must_be_in_utc() -> None:
    with pytest.raises(ValueError, match="UTC"):
        hourly_from_instantaneous(series("2025-06-01", [1.0] * 5, tz="Europe/Paris"))


def test_start_labelled_energy_is_summed_per_hour() -> None:
    energy = series("2025-06-01 00:00", [1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 5.0, 5.0])

    hourly = hourly_from_period_start_energy(energy)

    assert hourly.index.equals(series("2025-06-01 00:00", [0.0, 0.0], freq="h").index)
    assert hourly["energy_mwh"].tolist() == [10.0, 20.0]
    assert hourly["complete"].tolist() == [True, True]


def test_period_at_the_top_of_the_hour_starts_that_hour() -> None:
    energy = series("2025-06-01 00:45", [1.0, 2.0, 2.0, 2.0, 2.0])

    hourly = hourly_from_period_start_energy(energy)

    assert hourly.loc["2025-06-01 01:00", "energy_mwh"] == 8.0
    assert not hourly.loc["2025-06-01 00:00", "complete"]


@pytest.mark.parametrize("missing", [0, 3])
def test_a_missing_period_leaves_the_hour_incomplete(missing: int) -> None:
    energy = series("2025-06-01 00:00", [1.0] * 8).drop(
        pd.Timestamp("2025-06-01 00:00", tz="UTC") + missing * pd.Timedelta("15min")
    )

    hourly = hourly_from_period_start_energy(energy)

    assert np.isnan(hourly["energy_mwh"].iloc[0])
    assert hourly["complete"].tolist() == [False, True]


def test_a_missing_value_leaves_the_hour_incomplete() -> None:
    energy = series("2025-06-01 00:00", [1.0, np.nan, 1.0, 1.0])

    hourly = hourly_from_period_start_energy(energy)

    assert np.isnan(hourly["energy_mwh"].iloc[0])
    assert not hourly["complete"].iloc[0]


def test_hours_without_any_value_inside_the_span_are_kept() -> None:
    energy = pd.concat(
        [series("2025-06-01 00:00", [1.0] * 4), series("2025-06-01 02:00", [1.0] * 4)]
    )

    hourly = hourly_from_period_start_energy(energy)

    assert len(hourly) == 3
    assert hourly["complete"].tolist() == [True, False, True]
    assert np.isnan(hourly["energy_mwh"].iloc[1])


def test_duplicated_stamps_are_all_dropped() -> None:
    energy = series("2025-06-01 00:00", [1.0] * 4)
    energy = pd.concat([energy, energy.iloc[[1]] * 2])

    hourly = hourly_from_period_start_energy(energy)

    assert not hourly["complete"].iloc[0]


def test_unsorted_input_gives_the_same_result() -> None:
    energy = series("2025-06-01 00:00", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])

    shuffled = hourly_from_period_start_energy(energy.sample(frac=1, random_state=0))

    pd.testing.assert_frame_equal(shuffled, hourly_from_period_start_energy(energy))


@pytest.mark.parametrize("tz", [None, "Europe/Berlin"])
def test_input_must_be_in_utc(tz: str | None) -> None:
    index = pd.date_range("2025-06-01", periods=4, freq="15min", tz=tz)

    with pytest.raises(ValueError, match="UTC"):
        hourly_from_period_start_energy(pd.Series(1.0, index=index))


def test_input_must_be_on_the_period_grid() -> None:
    with pytest.raises(ValueError, match="grid"):
        hourly_from_period_start_energy(series("2025-06-01 00:05", [1.0] * 4))


@pytest.mark.parametrize(
    "convert",
    [hourly_from_period_end_means, hourly_from_instantaneous, hourly_from_period_start_energy],
)
def test_a_series_without_values_gives_no_hours(convert: Conversion) -> None:
    empty = pd.Series([], index=pd.DatetimeIndex([], tz="UTC"), dtype=float)

    hourly = convert(empty)

    assert hourly.empty
    Hourly.validate(hourly)


@pytest.mark.parametrize(
    ("convert", "values"),
    [
        (
            hourly_from_period_end_means,
            series("2025-06-01 00:30", [1.0, np.nan, 1.0, 1.0], freq="30min"),
        ),
        (
            hourly_from_instantaneous,
            series("2025-06-01 00:00", [1.0, np.nan, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
        ),
        (
            hourly_from_period_start_energy,
            series("2025-06-01 00:00", [1.0, np.nan, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
        ),
    ],
)
def test_converted_series_are_hourly_series(convert: Conversion, values: pd.Series) -> None:
    hourly = convert(values)

    assert hourly["complete"].tolist() == [False, True]
    Hourly.validate(hourly)


class HourlyWithStatus(Hourly):
    status: Series[str]


INVALID = [
    pytest.param(
        hourly_frame([1.0, 2.0], [True, False]),
        "energy_exactly_when_complete",
        id="energy-without-complete",
    ),
    pytest.param(
        hourly_frame([1.0, np.nan], [True, True]),
        "energy_exactly_when_complete",
        id="complete-without-energy",
    ),
    pytest.param(hourly_frame([1.0], [True], "2025-06-01 00:30"), "on_the_hour", id="off-hour"),
    pytest.param(pd.concat([hourly_frame([1.0], [True])] * 2), "duplicate", id="duplicate"),
    pytest.param(hourly_frame([1.0], [True]).tz_localize(None), r"datetime64\[ns\]", id="naive"),
    pytest.param(hourly_frame([1.0], [True]).rename_axis(index=None), "name 'time'", id="unnamed"),
]


def test_hourly_model_accepts_a_valid_frame() -> None:
    Hourly.validate(hourly_frame([1.0, np.nan], [True, False]))


@pytest.mark.parametrize(("frame", "error"), INVALID)
def test_hourly_model_rejects(frame: pd.DataFrame, error: str) -> None:
    with pytest.raises(SchemaError, match=error):
        Hourly.validate(frame)


def test_a_subclass_adds_its_columns() -> None:
    frame = hourly_frame([1.0, np.nan], [True, False]).assign(status="realtime")

    HourlyWithStatus.validate(frame)
    # pandera reports an undeclared column as SchemaErrors, even without lazy validation.
    with pytest.raises(SchemaErrors, match="status"):
        Hourly.validate(frame)


@pytest.mark.parametrize(("frame", "error"), INVALID)
def test_a_subclass_keeps_the_hourly_checks(frame: pd.DataFrame, error: str) -> None:
    with pytest.raises(SchemaError, match=error):
        HourlyWithStatus.validate(frame.assign(status="realtime"))


def test_on_hours_keeps_adds_and_drops_hours() -> None:
    hourly = hourly_frame([1.0, np.nan, 2.0, 3.0, 4.0], [True, False, True, True, True])
    hourly = hourly.drop(utc("2025-06-01 02:00"))
    hourly["status"] = ["definitive", "consolidated", "realtime", "realtime"]

    framed = on_hours(hourly, utc("2025-06-01 01:00"), utc("2025-06-01 04:00"))

    expected = hourly_frame([np.nan, np.nan, 3.0], [False, False, True], "2025-06-01 01:00")
    expected["status"] = ["consolidated", None, "realtime"]
    pd.testing.assert_frame_equal(framed, expected)


@pytest.mark.parametrize(
    "stamp",
    [
        pd.Timestamp("2025-06-01 02:00"),
        pd.Timestamp("2025-06-01 02:00", tz="Europe/Paris"),
        utc("2025-06-01 02:30"),
    ],
    ids=["naive", "not-in-utc", "off-hour"],
)
def test_a_period_needs_hour_starts_in_utc(stamp: pd.Timestamp) -> None:
    with pytest.raises(ValueError, match="hour in UTC"):
        check_period(stamp, utc("2025-06-02 00:00"))
    with pytest.raises(ValueError, match="hour in UTC"):
        check_period(utc("2025-05-31 00:00"), stamp)


@pytest.mark.parametrize("end", ["2025-06-01 02:00", "2025-06-01 00:00"], ids=["empty", "reversed"])
def test_a_period_needs_end_after_start(end: str) -> None:
    with pytest.raises(ValueError, match="empty period"):
        check_period(utc("2025-06-01 02:00"), utc(end))


def test_on_hours_checks_its_period() -> None:
    with pytest.raises(ValueError, match="empty period"):
        on_hours(hourly_frame([1.0], [True]), utc("2025-06-01 02:00"), utc("2025-06-01 02:00"))


@pytest.mark.parametrize("tz", [None, "Europe/Paris"])
def test_on_hours_needs_an_index_in_utc(tz: str | None) -> None:
    hourly = hourly_frame([1.0], [True]).tz_convert(tz)

    with pytest.raises(ValueError, match="DatetimeIndex in UTC"):
        on_hours(hourly, utc("2025-06-01 00:00"), utc("2025-06-01 01:00"))
