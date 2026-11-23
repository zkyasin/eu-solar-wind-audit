import json
from datetime import date
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import pytest

from eu_solar_wind_audit import grid
from eu_solar_wind_audit.anomalies.completeness import completeness, on_grid
from eu_solar_wind_audit.sources import energy_charts, odre, smard

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
        provisional=False,
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


def test_instantaneous_points_at_the_ends_of_the_period_are_needed() -> None:
    # The first hour opens with the point at the start, the last closes with the point at the end.
    found = check(rows().drop([utc(DAY_START), utc(DAY_END)]), label="instant")

    assert spans(found) == [
        ("gap", utc("2022-06-01 00:00"), utc("2022-06-01 01:00"), 1.0),
        ("gap", utc("2022-06-01 23:00"), utc("2022-06-02 00:00"), 1.0),
    ]
    assert [(gap["first_stamp"], gap["absent_stamps"]) for gap in evidence(found)] == [
        (utc(DAY_START).isoformat(), 1),
        (utc(DAY_END).isoformat(), 1),
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


def test_empty_rows_at_a_repeated_spring_stamp_are_a_gap() -> None:
    published = local_day_rows()
    published.loc[published.index == utc("2022-03-27 01:00")] = np.nan

    found = check_season(published, local_days="Europe/Paris")

    # Both rows ending at 01:00 are empty: their repetition follows the convention, not the gap.
    assert spans(found) == [
        ("gap", utc("2022-03-27 00:00"), utc("2022-03-27 01:00"), 1.0),
        ("clock_change_convention", utc("2022-03-27 00:00"), utc("2022-10-30 01:00"), 2.0),
    ]
    assert evidence(found)[0]["empty_stamps"] == 1


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


def test_separate_days_of_one_spacing_are_separate_changes_of_resolution() -> None:
    published = pd.concat(
        [
            rows(DAY_START, "2022-06-01 23:00", step="1h"),
            rows("2022-06-02 00:00", "2022-06-02 23:45"),
            rows("2022-06-03 00:00", "2022-06-03 23:00", step="1h"),
        ]
    )

    found = check(published, DAY_START, "2022-06-04 00:00")

    changes = found[found["kind"] == "resolution_change"]
    assert spans(changes) == [
        ("resolution_change", utc("2022-06-01 00:00"), utc("2022-06-02 00:00"), 60.0),
        ("resolution_change", utc("2022-06-03 00:00"), utc("2022-06-04 00:00"), 60.0),
    ]
    assert evidence(changes) == [{"step_minutes": 15, "days": 1}] * 2


def test_values_left_around_a_long_gap_are_not_a_change_of_resolution() -> None:
    # Three values two hours apart, and nothing else on the day.
    found = check(rows("2022-06-01 10:00", "2022-06-01 14:00", step="2h"))

    assert spans(found) == [("gap", utc(DAY_START), utc(DAY_END), 24.0)]


# Exports from https://odre.opendatasoft.com/api/explore/v2.1/catalog/datasets/<dataset>/exports/parquet
# with the select of each fetch function and where=odre.utc_days(first day, last day), retrieved
# 2026-11-22.
ODRE = Path(__file__).parent / "fixtures" / "odre"
NATIONAL_SPRING = ODRE / "eco2mix-national-cons-def_2025-03-30.parquet"
REALTIME_AUTUMN = ODRE / "eco2mix-national-tr_2026-10-24_2026-10-25.parquet"
REGIONAL_AUTUMN = ODRE / "eco2mix-regional-cons-def_2022-10-29_2022-10-30.parquet"
REGIONAL_YEAR_END = ODRE / "eco2mix-regional-cons-def_2025-12-31.parquet"
# SMARD German solar feed-in in the weeks of the 2025 clock changes, and Energy-Charts German
# production on 2025 days, retrieved 2026-11-22 (see tests/test_smard.py and
# tests/test_energy_charts.py).
SMARD = Path(__file__).parent / "fixtures" / "smard"
ENERGY_CHARTS = Path(__file__).parent / "fixtures" / "energy_charts"


def serving(path: Path) -> httpx.Client:
    content = path.read_bytes()
    return httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, content=content))
    )


def eco2mix(dataset: str) -> dict[str, str]:
    return {"dataset": f"odre.{dataset}", "provenance": f"RTE, éCO2mix, ODRÉ dataset {dataset}"}


def regional_series(frame: pd.DataFrame, *names: str) -> dict[str, pd.Series]:
    return {
        f"odre.FR-{code}.{name}": frame.loc[frame["region_code"] == code, f"{name}_mw"]
        for code in odre.REGIONS
        for name in names
    }


def test_national_spring_rows_follow_the_clock_change_convention(tmp_path: Path) -> None:
    with serving(NATIONAL_SPRING) as client:
        published = odre.fetch_national_consolidated(
            client, date(2025, 3, 30), date(2025, 3, 30), cache_dir=tmp_path
        )

    found = completeness(
        {f"odre.FR.{name}": published[f"{name}_mw"] for name in ("load", "solar", "wind")},
        utc("2025-03-30"),
        utc("2025-03-31"),
        step=odre.CONSOLIDATED_STEP,
        label=odre.CONSOLIDATED_LABEL,
        provisional=False,
        local_days=odre.TIMEZONE,
        **eco2mix(odre.NATIONAL_CONSOLIDATED),
    )

    # The half-hours ending 01:00 and 01:30 come twice; the empty quarter-hour rows between the
    # half-hours are not findings.
    assert spans(found) == [
        ("clock_change_convention", utc("2025-03-30 00:00"), utc("2025-03-30 02:00"), 1.0)
    ]
    assert found["series"].tolist() == ["odre.eco2mix-national-cons-def"]
    assert evidence(found)[0]["repeated_stamps"] == [
        "2025-03-30T01:00:00+00:00/2025-03-30T02:00:00+00:00"
    ]
    assert evidence(found)[0]["series_count"] == 3
    assert not found["provisional"].any()


def test_realtime_autumn_points_follow_the_clock_change_convention(tmp_path: Path) -> None:
    with serving(REALTIME_AUTUMN) as client:
        published = odre.fetch_national_realtime(
            client, date(2026, 10, 24), date(2026, 10, 25), cache_dir=tmp_path
        )
    names = ("load", "solar", "wind", "wind_onshore", "wind_offshore")

    found = completeness(
        {f"odre.FR.{name}": published[f"{name}_mw"] for name in names},
        utc("2026-10-24"),
        utc("2026-10-26"),
        step=odre.REALTIME_STEP,
        label=odre.REALTIME_LABEL,
        provisional=True,
        local_days=odre.TIMEZONE,
        **eco2mix(odre.NATIONAL_REALTIME),
    )

    # No point from 00:00 to 00:45 UTC on 25 October; the point at 00:00 would close the hour
    # from 23:00.
    assert spans(found) == [
        ("clock_change_convention", utc("2026-10-24 23:00"), utc("2026-10-25 01:00"), 1.0)
    ]
    assert evidence(found)[0]["missing_stamps"] == [
        "2026-10-25T00:00:00+00:00/2026-10-25T01:00:00+00:00"
    ]
    assert found["provisional"].all()
    assert evidence(found)[0]["series_count"] == len(names)


def test_regional_autumn_rows_follow_the_clock_change_convention(tmp_path: Path) -> None:
    with serving(REGIONAL_AUTUMN) as client:
        published = odre.fetch_regional_consolidated(
            client, date(2022, 10, 29), date(2022, 10, 30), cache_dir=tmp_path
        )
        hourly = odre.regional_hourly(
            client, "solar", utc("2022-10-29"), utc("2022-10-31"), cache_dir=tmp_path
        )

    found = completeness(
        regional_series(published, "solar", "wind"),
        utc("2022-10-29"),
        utc("2022-10-31"),
        step=odre.CONSOLIDATED_STEP,
        label=odre.CONSOLIDATED_LABEL,
        provisional=False,
        local_days=odre.TIMEZONE,
        **eco2mix(odre.REGIONAL_CONSOLIDATED),
    )

    # One finding for the 24 series, over the hours that the hourly conversion leaves incomplete.
    assert spans(found) == [
        ("clock_change_convention", utc("2022-10-29 23:00"), utc("2022-10-30 01:00"), 1.0)
    ]
    assert evidence(found)[0]["series_count"] == 24
    incomplete = pd.date_range(utc("2022-10-29 23:00"), periods=2, freq="h", name="time")
    for frame in hourly.values():
        assert frame.index[~frame["complete"]].equals(incomplete)


def test_the_empty_row_that_opens_2026_is_a_gap_in_every_regional_series(tmp_path: Path) -> None:
    with serving(REGIONAL_YEAR_END) as client:
        published = odre.fetch_regional_consolidated(
            client, date(2025, 12, 31), date(2025, 12, 31), cache_dir=tmp_path
        )
    series = regional_series(published, "solar", "wind")

    found = completeness(
        series,
        utc("2025-12-31"),
        utc("2026-01-01"),
        step=odre.CONSOLIDATED_STEP,
        label=odre.CONSOLIDATED_LABEL,
        provisional=False,
        local_days=odre.TIMEZONE,
        **eco2mix(odre.REGIONAL_CONSOLIDATED),
    )

    # The row stamped 23:00 UTC, midnight in Paris and the first consolidated row, has no values.
    assert found["series"].tolist() == list(series)
    gap = ("gap", utc("2025-12-31 22:00"), utc("2025-12-31 23:00"), 1.0)
    assert spans(found) == [gap] * len(series)
    stamp = "2025-12-31T23:00:00+00:00"
    empty_row = {"first_stamp": stamp, "last_stamp": stamp, "absent_stamps": 0, "empty_stamps": 1}
    assert evidence(found) == [empty_row] * len(series)


@pytest.mark.parametrize(
    ("week_ms", "hours"), [(1742770800000, 167), (1760911200000, 169)], ids=["spring", "autumn"]
)
def test_smard_weeks_of_the_clock_changes_are_complete(
    week_ms: int, hours: int, tmp_path: Path
) -> None:
    files = {
        "index_quarterhour.json": SMARD / "4068_DE_index_quarterhour.json",
        f"4068_DE_quarterhour_{week_ms}.json": SMARD / f"4068_DE_quarterhour_{week_ms}.json",
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=files[request.url.path.rsplit("/", 1)[-1]].read_bytes())

    start = pd.Timestamp(week_ms, unit="ms", tz="UTC")
    end = start + pd.Timedelta(hours=hours)
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        solar = smard.fetch_series(
            smard.SOLAR, "DE", "quarterhour", start, end, client=client, cache_dir=tmp_path
        )

    found = completeness(
        {"smard.DE.solar": solar},
        start,
        end,
        step=smard.FEED_IN_STEP,
        label=smard.FEED_IN_LABEL,
        dataset="smard.4068.DE",
        provenance="Bundesnetzagentur | SMARD.de, filter 4068, region DE",
        provisional=False,
    )

    assert found.empty


@pytest.mark.parametrize("endpoint", ["public_power", "total_power"])
@pytest.mark.parametrize("day", ["2025-03-30", "2025-06-21", "2025-10-26"])
def test_energy_charts_days_are_complete(endpoint: str, day: str) -> None:
    payload = json.loads((ENERGY_CHARTS / f"{endpoint}_de_{day}.json").read_bytes())
    power = energy_charts.parse_power(payload)
    local = pd.Timestamp(day, tz=energy_charts.TIMEZONE)

    found = completeness(
        {f"energy_charts.DE.{endpoint}.{name}": power[name] for name in power.columns},
        local.tz_convert("UTC"),
        (local + pd.DateOffset(days=1)).tz_convert("UTC"),
        step=energy_charts.QUARTER_HOUR,
        label=energy_charts.POWER_LABEL,
        dataset=f"energy_charts.{endpoint}",
        provenance=f"Energy-Charts.info, Fraunhofer ISE, {endpoint}",
        provisional=False,
    )

    assert found.empty
