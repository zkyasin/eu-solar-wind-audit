import numpy as np
import pandas as pd
import pytest

from eu_solar_wind_audit.grid import hourly_from_period_end_means, hourly_from_period_start_energy


def series(start: str, values: list[float], freq: str = "15min", tz: str = "UTC") -> pd.Series:
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq=freq, tz=tz))


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
