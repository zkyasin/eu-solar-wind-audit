import json

import numpy as np
import pandas as pd
import pytest
from pandera.errors import SchemaError

from eu_solar_wind_audit.anomalies.findings import Findings
from eu_solar_wind_audit.anomalies.outliers import above_capacity, negative_values

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


def test_a_series_without_status_gives_findings_that_are_not_provisional() -> None:
    found = negatives(hourly([-1.0]))

    assert not found["provisional"].any()


def test_hours_out_of_order_give_the_same_findings() -> None:
    published = hourly([-1.0, 2.0, -2.0, -1.0])

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
