import json

import numpy as np
import pandas as pd
import pytest

from eu_solar_wind_audit import grid
from eu_solar_wind_audit.anomalies.completeness import completeness, on_grid

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
    local_days: str | None = None,
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
        local_days=local_days,
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


# Half-hour means stamped at the end of their period over spring to autumn 2022, as éCO2mix
# publishes them: the stamps of the hour after the spring clock change come twice, and those of
# the hour before the autumn change are missing.
SPRING_REPEATED = [utc("2022-03-27 01:00"), utc("2022-03-27 01:30")]
AUTUMN_MISSING = [utc("2022-10-30 00:00"), utc("2022-10-30 00:30")]
SEASON_START, SEASON_END = "2022-03-26 00:00", "2022-11-01 00:00"


def local_day_rows() -> pd.Series:
    published = rows("2022-03-26 00:30", SEASON_END, step="30min")
    with_repeats: pd.Series = pd.concat([published, published.loc[SPRING_REPEATED] + 1.0])
    return with_repeats.drop(AUTUMN_MISSING)


def check_season(
    published: pd.Series | dict[str, pd.Series], local_days: str | None = None
) -> pd.DataFrame:
    return check(
        published, SEASON_START, SEASON_END, step="30min", label="end", local_days=local_days
    )


def test_clock_change_rows_of_all_the_series_give_one_finding_for_the_dataset() -> None:
    published = {"test.FR.solar": local_day_rows(), "test.FR.wind": local_day_rows()}

    found = check_season(published, local_days="Europe/Paris")

    # The repeated half-hours end at 01:00 and 01:30, the missing ones at 00:00 and 00:30.
    assert spans(found) == [
        ("clock_change_convention", utc("2022-03-27 00:00"), utc("2022-10-30 01:00"), 2.0)
    ]
    assert found["series"].tolist() == [DATASET]
    assert evidence(found) == [
        {
            "time_zone": "Europe/Paris",
            "repeated_stamps": ["2022-03-27T01:00:00+00:00/2022-03-27T02:00:00+00:00"],
            "missing_stamps": ["2022-10-30T00:00:00+00:00/2022-10-30T01:00:00+00:00"],
            "series_count": 2,
        }
    ]


def test_without_local_days_clock_change_rows_are_gaps_and_duplicates() -> None:
    found = check_season(local_day_rows())

    assert spans(found) == [
        ("gap", utc("2022-10-29 23:00"), utc("2022-10-30 01:00"), 2.0),
        ("duplicate", utc("2022-03-27 00:00"), utc("2022-03-27 02:00"), 2.0),
    ]


def test_clock_change_rows_beyond_the_convention_are_still_reported() -> None:
    published = local_day_rows()
    third_row = published[[utc("2022-03-27 01:00")]].iloc[:1]
    empty_row = pd.Series([np.nan], index=[utc("2022-10-30 00:30")])

    found = check_season(pd.concat([published, third_row, empty_row]), local_days="Europe/Paris")

    assert spans(found) == [
        ("gap", utc("2022-10-30 00:00"), utc("2022-10-30 01:00"), 1.0),
        ("duplicate", utc("2022-03-27 00:00"), utc("2022-03-27 01:00"), 1.0),
        # Only the half-hour ending 01:30 and the one ending 00:00 follow the convention.
        ("clock_change_convention", utc("2022-03-27 01:00"), utc("2022-10-30 00:00"), 2.0),
    ]
    assert evidence(found)[1]["rows"] == 3


def test_rows_stamped_in_utc_across_clock_changes_give_no_finding() -> None:
    published = rows("2022-03-26 00:30", SEASON_END, step="30min")

    assert check_season(published, local_days="Europe/Paris").empty


def test_values_off_the_grid_are_reported_in_the_hours_they_cover() -> None:
    off = pd.Series([5.0, 6.0], index=[utc("2022-06-01 10:05"), utc("2022-06-01 10:20")])

    found = check(pd.concat([rows(), off]))

    assert spans(found) == [("off_grid", utc("2022-06-01 10:00"), utc("2022-06-01 11:00"), 2.0)]
    assert evidence(found) == [
        {
            "first_stamp": "2022-06-01T10:05:00+00:00",
            "last_stamp": "2022-06-01T10:20:00+00:00",
            "minutes_past_hour": [5, 20],
        }
    ]


def test_empty_rows_off_the_grid_are_not_reported() -> None:
    # The éCO2mix national layout: a row every quarter-hour, a value every half-hour.
    published = rows("2022-06-01 00:15", DAY_END)
    published.loc[pd.DatetimeIndex(published.index).minute % 30 == 15] = np.nan

    assert check(published, step="30min", label="end").empty


def test_the_rows_on_the_grid_convert_to_hours() -> None:
    published = pd.concat(
        [rows(DAY_START, "2022-06-01 01:45"), pd.Series([5.0], index=[utc("2022-06-01 00:05")])]
    )
    with pytest.raises(ValueError, match="off the"):
        grid.hourly_from_period_start_energy(published)

    hourly = grid.hourly_from_period_start_energy(on_grid(published, pd.Timedelta("15min")))

    assert hourly["energy_mwh"].tolist() == [0.0 + 1 + 2 + 3, 4.0 + 5 + 6 + 7]
    assert hourly["complete"].all()


def test_a_day_of_finer_values_is_a_change_of_resolution() -> None:
    published = pd.concat(
        [
            rows(DAY_START, "2022-06-01 23:30", step="30min"),
            rows("2022-06-02 00:00", "2022-06-02 23:45", step="15min"),
        ]
    )

    found = check(published, DAY_START, "2022-06-03 00:00", step="30min")

    changes = found[found["kind"] == "resolution_change"]
    assert spans(changes) == [
        ("resolution_change", utc("2022-06-02 00:00"), utc("2022-06-03 00:00"), 15.0)
    ]
    assert evidence(changes) == [{"step_minutes": 30, "days": 1}]
    # The values between the half-hours are off the grid.
    assert found.loc[found["kind"] == "off_grid", "score"].tolist() == [48.0]


def test_days_of_hourly_values_are_one_change_of_resolution_and_gaps() -> None:
    published = pd.concat(
        [
            rows(DAY_START, "2022-06-01 23:45"),
            rows("2022-06-02 00:00", "2022-06-03 23:00", step="1h"),
            rows("2022-06-04 00:00", "2022-06-04 23:45"),
        ]
    )

    found = check(published, DAY_START, "2022-06-05 00:00")

    assert spans(found) == [
        ("gap", utc("2022-06-02 00:00"), utc("2022-06-04 00:00"), 48.0),
        ("resolution_change", utc("2022-06-02 00:00"), utc("2022-06-04 00:00"), 60.0),
    ]
    assert evidence(found)[0]["absent_stamps"] == 2 * 72
    assert evidence(found)[1] == {"step_minutes": 15, "days": 2}
