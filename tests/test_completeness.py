import json

import numpy as np
import pandas as pd
import pytest

from eu_solar_wind_audit import grid
from eu_solar_wind_audit.anomalies.completeness import completeness

DATASET = "test.dataset"
PROVENANCE = "Producer, test dataset"
DAY_START, DAY_END = "2022-06-01 00:00", "2022-06-02 00:00"


def utc(stamp: str) -> pd.Timestamp:
    return pd.Timestamp(stamp, tz="UTC")


def rows(first: str = DAY_START, last: str = DAY_END, step: str = "15min") -> pd.Series:
    """A value at every stamp from `first` to `last`, both included."""
    stamps = pd.date_range(utc(first), utc(last), freq=step)
    return pd.Series(np.arange(len(stamps), dtype=float), index=stamps)


def check(
    published: pd.Series | dict[str, pd.Series],
    start: str = DAY_START,
    end: str = DAY_END,
    *,
    step: str = "15min",
    label: grid.Label = "start",
) -> pd.DataFrame:
    if isinstance(published, pd.Series):
        published = {"test.DE.solar": published}
    return completeness(
        published,
        utc(start),
        utc(end),
        step=pd.Timedelta(step),
        label=label,
        dataset=DATASET,
        provenance=PROVENANCE,
    )


def spans(found: pd.DataFrame) -> list[tuple[str, pd.Timestamp, pd.Timestamp, float]]:
    return list(zip(found["kind"], found["start"], found["end"], found["score"], strict=True))


def evidence(found: pd.DataFrame) -> list[dict[str, object]]:
    return [json.loads(text) for text in found["evidence"]]


@pytest.mark.parametrize("label", ["start", "end", "instant"])
def test_complete_rows_give_no_finding(label: grid.Label) -> None:
    # The rows run from the first stamp that any label needs to the last one.
    assert check(rows(), label=label).empty


@pytest.mark.parametrize(
    ("label", "stamp", "first_hour", "end"),
    [
        ("start", "10:00", "10:00", "11:00"),
        # A quarter-hour mean stamped at its end, 10:00, covers 09:45 to 10:00.
        ("end", "10:00", "09:00", "10:00"),
        # An instantaneous value on the hour closes one hour and opens the next.
        ("instant", "10:00", "09:00", "11:00"),
        ("instant", "10:15", "10:00", "11:00"),
    ],
)
def test_a_missing_stamp_is_a_gap_in_the_hours_its_value_covers(
    label: grid.Label, stamp: str, first_hour: str, end: str
) -> None:
    missing = utc(f"2022-06-01 {stamp}")

    found = check(rows().drop(missing), label=label)

    start, stop = utc(f"2022-06-01 {first_hour}"), utc(f"2022-06-01 {end}")
    assert spans(found) == [("gap", start, stop, (stop - start) / grid.HOUR)]
    assert evidence(found) == [
        {
            "first_stamp": missing.isoformat(),
            "last_stamp": missing.isoformat(),
            "absent_stamps": 1,
            "empty_stamps": 0,
        }
    ]


def test_a_missing_value_is_a_gap_with_an_empty_stamp() -> None:
    published = rows()
    published.loc[published.index == utc("2022-06-01 10:15")] = np.nan

    found = check(published)

    assert spans(found) == [("gap", utc("2022-06-01 10:00"), utc("2022-06-01 11:00"), 1.0)]
    assert evidence(found)[0]["absent_stamps"] == 0
    assert evidence(found)[0]["empty_stamps"] == 1


def test_missing_stamps_give_one_gap_per_run_of_hours() -> None:
    three_hours = pd.date_range(utc("2022-06-01 10:00"), utc("2022-06-01 12:45"), freq="15min")

    found = check(rows().drop(three_hours).drop(utc("2022-06-01 15:30")))

    assert spans(found) == [
        ("gap", utc("2022-06-01 10:00"), utc("2022-06-01 13:00"), 3.0),
        ("gap", utc("2022-06-01 15:00"), utc("2022-06-01 16:00"), 1.0),
    ]
    assert [found_evidence["absent_stamps"] for found_evidence in evidence(found)] == [12, 1]


def test_hours_without_rows_at_the_ends_of_the_period_are_gaps() -> None:
    found = check(rows("2022-06-01 02:00", "2022-06-01 21:45"))

    assert spans(found) == [
        ("gap", utc("2022-06-01 00:00"), utc("2022-06-01 02:00"), 2.0),
        ("gap", utc("2022-06-01 22:00"), utc("2022-06-02 00:00"), 2.0),
    ]


def test_a_series_without_rows_is_one_gap_over_the_period() -> None:
    no_rows = pd.Series([], index=pd.DatetimeIndex([], tz="UTC"), dtype=float)

    found = check(no_rows)

    assert spans(found) == [("gap", utc(DAY_START), utc(DAY_END), 24.0)]
    assert evidence(found)[0]["absent_stamps"] == 96


def test_a_repeated_stamp_is_a_duplicate() -> None:
    published = rows()
    repeated = utc("2022-06-01 10:15")

    found = check(pd.concat([published, published[[repeated]] + 1.0]))

    assert spans(found) == [("duplicate", utc("2022-06-01 10:00"), utc("2022-06-01 11:00"), 1.0)]
    assert evidence(found) == [
        {"first_stamp": repeated.isoformat(), "last_stamp": repeated.isoformat(), "rows": 2}
    ]


def test_rows_outside_the_period_are_not_checked() -> None:
    published = rows()
    published = pd.concat([published, published[[utc("2022-06-01 05:45")]]])

    found = check(published.drop(utc("2022-06-01 18:15")), "2022-06-01 06:00", "2022-06-01 18:00")

    assert found.empty


def test_each_series_is_checked_on_its_own() -> None:
    complete = rows()

    found = check({"test.DE.solar": complete, "test.DE.wind": complete.drop(utc(DAY_START))})

    assert found["series"].tolist() == ["test.DE.wind"]
    assert found["provenance"].tolist() == [PROVENANCE]


@pytest.mark.parametrize("tz", [None, "Europe/Berlin"])
def test_stamps_must_be_in_utc(tz: str | None) -> None:
    with pytest.raises(ValueError, match="UTC"):
        check(rows().tz_convert(tz))


def test_the_step_must_divide_an_hour() -> None:
    with pytest.raises(ValueError, match="divide an hour"):
        check(rows(step="7min"), step="7min")


def test_the_period_must_be_hour_starts_in_utc() -> None:
    with pytest.raises(ValueError, match="hour in UTC"):
        check(rows(), "2022-06-01 00:30")
