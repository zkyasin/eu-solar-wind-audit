import json

import numpy as np
import pandas as pd
import pytest
from pandera.errors import SchemaError

from eu_solar_wind_audit.anomalies.findings import Findings
from eu_solar_wind_audit.anomalies.outliers import (
    SOLAR_ZERO_FLOOR_HOURS,
    WIND_ZERO_FLOOR_FACTOR,
    Technology,
    above_capacity,
    frozen_values,
    negative_values,
)

SERIES = "test.FR.solar"
PROVENANCE = "Producer, test dataset"
START = "2022-06-01 00:00"


def utc(stamp: str) -> pd.Timestamp:
    return pd.Timestamp(stamp, tz="UTC")


def hourly(values: list[float], start: str = START) -> pd.DataFrame:
    """A grid.Hourly frame from one value per hour; NaN values are incomplete hours."""
    time = pd.date_range(utc(start), periods=len(values), freq="h", unit="ns", name="time")
    energy = pd.Series(values, index=time, dtype=float)
    return pd.DataFrame({"energy_mwh": energy, "complete": energy.notna()})


def with_status(frame: pd.DataFrame, realtime_from: str) -> pd.DataFrame:
    """An odre.Hourly frame: consolidated hours, then real-time hours from `realtime_from`."""
    status = np.where(frame.index < utc(realtime_from), "consolidated", "realtime")
    return frame.assign(status=status)


def spans(found: pd.DataFrame) -> list[tuple[str, pd.Timestamp, pd.Timestamp, float, bool]]:
    columns = ["kind", "start", "end", "score", "provisional"]
    return list(found[columns].itertuples(index=False, name=None))


def evidence(found: pd.DataFrame) -> list[dict[str, object]]:
    return [json.loads(text) for text in found["evidence"]]


def negatives(published: pd.DataFrame) -> pd.DataFrame:
    return negative_values(published, series=SERIES, provenance=PROVENANCE)


def test_a_run_of_negative_hours_is_one_finding() -> None:
    found = negatives(hourly([0.0, -1.0, -3.5, -2.0, 0.0, 5.0]))

    assert spans(found) == [
        ("negative", utc("2022-06-01 01:00"), utc("2022-06-01 04:00"), -3.5, False)
    ]
    assert evidence(found) == [
        {"hours": 3, "lowest_hour": "2022-06-01T02:00:00+00:00", "lowest_mwh": -3.5}
    ]
    assert found["series"].tolist() == [SERIES]
    assert found["provenance"].tolist() == [PROVENANCE]


def test_zero_and_incomplete_hours_are_not_negative() -> None:
    assert negatives(hourly([0.0, -0.0, np.nan, 2.0])).empty


def hours_apart(gap: int) -> pd.DataFrame:
    """One negative hour, `gap` hours without a negative value, then another negative hour."""
    return hourly([-1.0, *[1.0] * gap, -2.0])


def test_negative_runs_closer_than_seven_days_are_joined() -> None:
    found = negatives(hours_apart(7 * 24 - 1))

    assert spans(found) == [("negative", utc(START), utc("2022-06-08 01:00"), -2.0, False)]
    assert evidence(found)[0]["hours"] == 2


def test_negative_runs_seven_days_apart_are_separate() -> None:
    found = negatives(hours_apart(7 * 24))

    assert spans(found) == [
        ("negative", utc(START), utc("2022-06-01 01:00"), -1.0, False),
        ("negative", utc("2022-06-08 01:00"), utc("2022-06-08 02:00"), -2.0, False),
    ]


def test_negative_runs_are_never_joined_across_real_time_hours() -> None:
    published = with_status(hourly([-1.0, -1.0, 1.0, -3.0, -1.0]), "2022-06-01 03:00")

    found = negatives(published)

    assert spans(found) == [
        ("negative", utc(START), utc("2022-06-01 02:00"), -1.0, False),
        ("negative", utc("2022-06-01 03:00"), utc("2022-06-01 05:00"), -3.0, True),
    ]


def test_an_hour_without_rows_never_splits_real_time_runs() -> None:
    # Like the hour that éCO2mix leaves out at each autumn clock change.
    realtime = with_status(hourly([-1.0, np.nan, -2.0]), START)
    changing = with_status(hourly([-1.0, np.nan, -2.0]), "2022-06-01 01:00")
    for published in (realtime, changing):
        published.loc[utc("2022-06-01 01:00"), "status"] = None

    assert spans(negatives(realtime)) == [
        ("negative", utc(START), utc("2022-06-01 03:00"), -2.0, True)
    ]
    assert spans(negatives(changing)) == [
        ("negative", utc(START), utc("2022-06-01 01:00"), -1.0, False),
        ("negative", utc("2022-06-01 02:00"), utc("2022-06-01 03:00"), -2.0, True),
    ]


def test_a_series_without_status_gives_findings_that_are_not_provisional() -> None:
    found = negatives(hourly([-1.0]))

    assert not found["provisional"].any()


def test_hours_out_of_order_give_the_same_findings() -> None:
    published = hourly([-1.0, 2.0, -2.0, -1.0])

    pd.testing.assert_frame_equal(negatives(published.iloc[::-1]), negatives(published))


def test_hours_with_status_out_of_order_give_the_same_findings() -> None:
    published = with_status(hourly([-1.0, 2.0, -2.0, -1.0]), "2022-06-01 02:00")

    pd.testing.assert_frame_equal(negatives(published.iloc[::-1]), negatives(published))


def test_no_negative_value_gives_an_empty_valid_frame() -> None:
    found = negatives(hourly([1.0, 2.0]))

    assert found.empty
    assert list(found.columns) == list(Findings.to_schema().columns)


def test_a_series_off_the_hourly_grid_is_refused() -> None:
    published = hourly([-1.0, -2.0])
    published.index = published.index + pd.Timedelta(minutes=30)

    with pytest.raises(SchemaError, match="on_the_hour"):
        negatives(published)


def capacity(values: list[float], start: str = START) -> pd.Series:
    time = pd.date_range(utc(start), periods=len(values), freq="h", unit="ns", name="time")
    return pd.Series(values, index=time, name="capacity_mw")


def above(published: pd.DataFrame, capacity_mw: pd.Series) -> pd.DataFrame:
    return above_capacity(published, capacity_mw, series=SERIES, provenance=PROVENANCE)


def test_values_above_1_05_times_the_capacity_are_flagged() -> None:
    found = above(hourly([100.0, 104.0, 106.0, 105.0]), capacity([100.0] * 4))

    assert spans(found) == [
        ("above_capacity", utc("2022-06-01 02:00"), utc("2022-06-01 03:00"), 1.06, False)
    ]
    assert evidence(found) == [
        {
            "hours": 1,
            "highest_hour": "2022-06-01T02:00:00+00:00",
            "energy_mwh": 106.0,
            "capacity_mw": 100.0,
            "ratio": 1.06,
        }
    ]
    assert found["series"].tolist() == [SERIES]


def test_each_hour_is_compared_with_its_own_capacity() -> None:
    found = above(hourly([110.0, 110.0, 230.0]), capacity([100.0, 200.0, 200.0]))

    assert spans(found) == [("above_capacity", utc(START), utc("2022-06-01 03:00"), 1.15, False)]
    assert evidence(found)[0]["hours"] == 2
    assert evidence(found)[0]["highest_hour"] == "2022-06-01T02:00:00+00:00"


def test_the_evidence_is_read_at_the_hour_of_the_highest_ratio() -> None:
    # That hour has neither the highest energy nor the highest capacity of the run.
    found = above(hourly([120.0, 230.0]), capacity([100.0, 200.0]))

    assert evidence(found) == [
        {
            "hours": 2,
            "highest_hour": "2022-06-01T00:00:00+00:00",
            "energy_mwh": 120.0,
            "capacity_mw": 100.0,
            "ratio": 1.2,
        }
    ]


@pytest.mark.parametrize(("gap", "findings"), [(7 * 24 - 1, 1), (7 * 24, 2)])
def test_runs_above_the_capacity_are_joined_within_seven_days(gap: int, findings: int) -> None:
    published = hourly([2.0, *[1.0] * gap, 2.0])

    found = above(published, capacity([1.0] * len(published)))

    assert len(found) == findings


def test_runs_above_the_capacity_are_never_joined_across_real_time_hours() -> None:
    published = with_status(hourly([2.0, 1.0, 3.0]), "2022-06-01 02:00")

    found = above(published, capacity([1.0] * 3))

    assert spans(found) == [
        ("above_capacity", utc(START), utc("2022-06-01 01:00"), 2.0, False),
        ("above_capacity", utc("2022-06-01 02:00"), utc("2022-06-01 03:00"), 3.0, True),
    ]


def test_incomplete_hours_need_no_capacity() -> None:
    found = above(hourly([np.nan, 2.0]), capacity([np.nan, 1.0]))

    assert spans(found) == [
        ("above_capacity", utc("2022-06-01 01:00"), utc("2022-06-01 02:00"), 2.0, False)
    ]


@pytest.mark.parametrize(
    "capacity_mw",
    [capacity([1.0]), capacity([1.0, 0.0]), capacity([1.0, np.nan])],
    ids=["missing", "zero", "nan"],
)
def test_complete_hours_without_a_positive_capacity_are_refused(capacity_mw: pd.Series) -> None:
    with pytest.raises(ValueError, match="positive capacity"):
        above(hourly([1.0, 0.0]), capacity_mw)


def frozen(
    published: pd.DataFrame,
    technology: Technology = "wind",
    modelled: pd.DataFrame | None = None,
) -> pd.DataFrame:
    return frozen_values(
        published,
        series=SERIES,
        provenance=PROVENANCE,
        technology=technology,
        modelled=modelled,
    )


def modelled_hours(energy_mwh: list[float], capacity_mw: float = 100.0) -> pd.DataFrame:
    """A physics.Modelled frame from one modelled energy per hour."""
    return hourly(energy_mwh).assign(
        capacity_mw=capacity_mw, capacity_provisional=False, era5_source="ERA5"
    )


@pytest.mark.parametrize(("hours", "findings"), [(23, 0), (24, 1)])
def test_values_are_frozen_from_24_hours_with_one_value(hours: int, findings: int) -> None:
    found = frozen(hourly([1.0, *[5.0] * hours, 6.0]))

    assert len(found) == findings


def test_a_frozen_run_is_one_finding_with_its_value() -> None:
    found = frozen(hourly([1.0, *[5.5] * 30, 6.0]))

    assert spans(found) == [
        ("frozen", utc("2022-06-01 01:00"), utc("2022-06-02 07:00"), 30.0, False)
    ]
    assert evidence(found) == [{"hours": 30, "value_mwh": 5.5}]
    assert found["series"].tolist() == [SERIES]


def test_a_frozen_run_ends_at_an_incomplete_or_missing_hour() -> None:
    incomplete = hourly([*[5.0] * 20, np.nan, *[5.0] * 20])
    missing = hourly([5.0] * 41).drop(index=utc("2022-06-01 20:00"))

    assert frozen(incomplete).empty
    assert frozen(missing).empty


def test_frozen_runs_never_span_real_time_and_other_hours() -> None:
    published = with_status(hourly([5.0] * 48), "2022-06-02 00:00")
    short = with_status(hourly([5.0] * 30), "2022-06-01 20:00")

    found = frozen(published)

    assert spans(found) == [
        ("frozen", utc(START), utc("2022-06-02 00:00"), 24.0, False),
        ("frozen", utc("2022-06-02 00:00"), utc("2022-06-03 00:00"), 24.0, True),
    ]
    assert frozen(short).empty


def test_zeros_without_a_model_are_frozen() -> None:
    found = frozen(hourly([0.0] * 24))

    assert evidence(found) == [{"hours": 24, "value_mwh": 0.0}]


def calm(factor: float) -> pd.DataFrame:
    """24 hours of modelled wind with a capacity factor of `factor` on 100 MW."""
    return modelled_hours([100.0 * factor] * 24)


def test_calm_zeros_below_the_wind_floor_are_not_frozen() -> None:
    assert frozen(hourly([0.0] * 24), "wind", calm(0.9 * WIND_ZERO_FLOOR_FACTOR)).empty


def test_zeros_above_the_wind_floor_are_frozen() -> None:
    found = frozen(hourly([0.0] * 24), "wind", calm(1.1 * WIND_ZERO_FLOOR_FACTOR))

    assert evidence(found)[0]["modelled_capacity_factor"] == pytest.approx(
        1.1 * WIND_ZERO_FLOOR_FACTOR
    )


def day(full_load_hours: float) -> pd.DataFrame:
    """24 hours of modelled solar on 100 MW, all the energy around noon."""
    energy = np.zeros(24)
    energy[10:14] = 100.0 * full_load_hours / 4
    return modelled_hours(energy.tolist())


def test_dark_zeros_below_the_solar_floor_are_not_frozen() -> None:
    assert frozen(hourly([0.0] * 24), "solar", day(0.9 * SOLAR_ZERO_FLOOR_HOURS)).empty


def test_zeros_above_the_solar_floor_are_frozen() -> None:
    found = frozen(hourly([0.0] * 24), "solar", day(1.1 * SOLAR_ZERO_FLOOR_HOURS))

    assert evidence(found)[0]["modelled_full_load_hours"] == pytest.approx(
        1.1 * SOLAR_ZERO_FLOOR_HOURS
    )


def test_the_solar_floor_holds_for_the_daylight_energy_of_the_whole_run() -> None:
    # Each day alone stays below the floor, the two days together do not.
    two_days = pd.concat([day(0.6 * SOLAR_ZERO_FLOOR_HOURS)] * 2)
    two_days.index = pd.date_range(utc(START), periods=48, freq="h", unit="ns", name="time")

    found = frozen(hourly([0.0] * 48), "solar", two_days)

    assert evidence(found)[0]["modelled_full_load_hours"] == pytest.approx(
        1.2 * SOLAR_ZERO_FLOOR_HOURS
    )


def test_zeros_are_frozen_when_the_model_lacks_some_of_their_hours() -> None:
    model = calm(0.0)
    model.loc[utc("2022-06-01 12:00"), ["energy_mwh", "complete"]] = [np.nan, False]

    found = frozen(hourly([0.0] * 24), "wind", model)
    shorter = frozen(hourly([0.0] * 25), "wind", calm(0.0))

    assert evidence(found) == [{"hours": 24, "value_mwh": 0.0}]
    assert evidence(shorter) == [{"hours": 25, "value_mwh": 0.0}]


def test_a_model_out_of_order_judges_zeros_the_same() -> None:
    assert frozen(hourly([0.0] * 24), "wind", calm(0.0).iloc[::-1]).empty


@pytest.mark.parametrize("technology", ["solar", "wind"])
def test_frozen_values_other_than_zero_need_no_model(technology: Technology) -> None:
    found = frozen(hourly([3.0] * 24), technology, calm(0.0))

    assert evidence(found) == [{"hours": 24, "value_mwh": 3.0}]
