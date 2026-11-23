import json
from collections.abc import Mapping

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
from pandera.errors import SchemaError

from eu_solar_wind_audit import grid
from eu_solar_wind_audit.anomalies.cross_source import proportional, sum_of_parts
from eu_solar_wind_audit.anomalies.findings import Findings
from eu_solar_wind_audit.sources import odre

SERIES = "test.DE.solar"
PROVENANCE = "Producer, test datasets"
FIRST_HOUR = pd.Timestamp("2022-06-01 00:00", tz="UTC")
HOURS = 72

FloatArray = npt.NDArray[np.float64]


def hour(number: int, first_hour: pd.Timestamp = FIRST_HOUR) -> pd.Timestamp:
    return first_hour + number * grid.HOUR


def hourly(energy_mwh: npt.ArrayLike, first_hour: pd.Timestamp = FIRST_HOUR) -> pd.DataFrame:
    """A frame of grid.Hourly from `first_hour` on; NaN hours are incomplete."""
    energy = np.asarray(energy_mwh, dtype=float)
    time = pd.date_range(first_hour, periods=len(energy), freq="h", unit="ns", name="time")
    frame = pd.DataFrame({"energy_mwh": energy, "complete": ~np.isnan(energy)}, index=time)
    return grid.Hourly.validate(frame)


def with_status(frame: pd.DataFrame, status: npt.ArrayLike) -> pd.DataFrame:
    """`frame` as odre.Hourly gives it, with `status` in its complete hours and none in hours
    without rows, which are incomplete."""
    by_hour = pd.Series(status, index=frame.index).where(frame["complete"])
    return odre.Hourly.validate(frame.assign(status=by_hour))


def split(total_mwh: npt.ArrayLike, gaps: Mapping[int, float]) -> list[FloatArray]:
    """A total and two parts that add up to it, except that they fall short of it by `gaps[h]`
    at hour h."""
    total = np.asarray(total_mwh, dtype=float)
    first, second = total / 2, total / 2
    for number, gap in gaps.items():
        second[number] -= gap
    return [total, first, second]


def check_parts(
    total: pd.DataFrame, *parts: pd.DataFrame, provisional: bool = False
) -> pd.DataFrame:
    return sum_of_parts(
        total,
        {f"part{number}": part for number, part in enumerate(parts)},
        series=SERIES,
        provenance=PROVENANCE,
        provisional=provisional,
    )


def check_split(total_mwh: npt.ArrayLike, gaps: Mapping[int, float]) -> pd.DataFrame:
    return check_parts(*[hourly(energy) for energy in split(total_mwh, gaps)])


def spans(found: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    return list(zip(found["start"], found["end"], strict=True))


def evidence(found: pd.DataFrame) -> list[dict[str, object]]:
    return [json.loads(text) for text in found["evidence"]]


def test_parts_that_add_up_give_no_finding() -> None:
    found = check_split(np.full(HOURS, 1000.0), {})

    assert found.empty
    Findings.validate(found)


def test_a_gap_of_5_mwh_is_rounding_and_a_gap_of_2_percent_is_a_finding() -> None:
    # 5 MWh is 5 % of a total of 100 MWh, but within the 10 MWh floor; 40 MWh is 2 % of 2000 MWh.
    total = np.repeat([100.0, 2000.0], HOURS // 2)

    found = check_split(total, {10: 5.0, 50: 40.0})

    assert spans(found) == [(hour(50), hour(51))]
    assert found["kind"].tolist() == ["sum_of_parts"]
    assert found["score"].tolist() == [40.0]
    assert evidence(found) == [
        {"parts": 2, "hours": 1, "largest_gap_mwh": 40.0, "largest_relative_gap": 0.02}
    ]


@pytest.mark.parametrize(
    ("total_mwh", "gap_mwh", "flagged"),
    [
        (100.0, 10.0, False),
        (100.0, 10.5, True),
        (100.0, -10.5, True),
        (3000.0, 25.0, False),
        (3000.0, 30.0, False),
        (3000.0, 30.5, True),
        (3000.0, -30.5, True),
    ],
)
def test_a_gap_counts_beyond_10_mwh_and_beyond_1_percent_of_the_total(
    total_mwh: float, gap_mwh: float, flagged: bool
) -> None:
    found = check_split(np.full(HOURS, total_mwh), {10: gap_mwh})

    assert spans(found) == ([(hour(10), hour(11))] if flagged else [])


@pytest.mark.parametrize("missing", ["total", "part", "part row"])
def test_an_hour_that_the_total_or_a_part_lacks_is_not_compared(missing: str) -> None:
    total, first, second = split(np.full(HOURS, 1000.0), {})
    if missing == "total":
        total[10] = np.nan
    if missing == "part":
        second[10] = np.nan
    parts = [hourly(first), hourly(second)]
    if missing == "part row":
        parts[1] = parts[1].drop(hour(10))

    assert check_parts(hourly(total), *parts).empty


def test_only_the_hours_of_the_total_are_compared() -> None:
    total, first, second = split(np.full(HOURS, 1000.0), {60: 50.0})

    assert check_parts(hourly(total[:48]), hourly(first), hourly(second)).empty


@pytest.mark.parametrize(
    ("later_hour", "expected"),
    [
        # 23 hours without a gap between the two: one finding over the hours between them.
        (34, [(10, 35, 2)]),
        (35, [(10, 11, 1), (35, 36, 1)]),
    ],
)
def test_gaps_less_than_a_day_apart_are_one_finding(
    later_hour: int, expected: list[tuple[int, int, int]]
) -> None:
    found = check_split(np.full(HOURS, 1000.0), {10: 50.0, later_hour: 50.0})

    assert spans(found) == [(hour(start), hour(end)) for start, end, _ in expected]
    assert [gap["hours"] for gap in evidence(found)] == [hours for _, _, hours in expected]


def test_evidence_gives_the_largest_gap_and_the_largest_share_of_the_total() -> None:
    total = np.full(HOURS, 3000.0)
    total[10] = 100.0

    found = check_split(total, {10: 20.0, 12: 60.0})

    assert found["score"].tolist() == [60.0]
    assert evidence(found) == [
        {"parts": 2, "hours": 2, "largest_gap_mwh": 60.0, "largest_relative_gap": 0.2}
    ]


@pytest.mark.parametrize(("total_mwh", "relative_gap"), [(0.0, None), (-12.0, 11 / 12)])
def test_the_relative_gap_is_a_share_of_the_size_of_the_total(
    total_mwh: float, relative_gap: float | None
) -> None:
    total = np.full(HOURS, 1000.0)
    total[10] = total_mwh

    found = check_split(total, {10: 11.0})

    assert evidence(found)[0]["largest_relative_gap"] == relative_gap


def test_findings_name_the_total_with_the_provenance_given() -> None:
    total, first, second = split(np.full(HOURS, 1000.0), {10: 50.0})

    found = check_parts(hourly(total), hourly(first), hourly(second), provisional=True)

    assert found["series"].tolist() == [SERIES]
    assert found["provenance"].tolist() == [PROVENANCE]
    assert found["provisional"].tolist() == [True]


@pytest.mark.parametrize(("status", "provisional"), [("definitive", False), ("realtime", True)])
def test_frames_with_the_status_of_each_hour_are_compared(status: str, provisional: bool) -> None:
    total, first, second = split(np.full(HOURS, 1000.0), {10: 50.0})
    total[20] = np.nan

    found = check_parts(
        *[with_status(hourly(energy), status) for energy in (total, first, second)],
        provisional=provisional,
    )

    assert spans(found) == [(hour(10), hour(11))]
    assert found["provisional"].tolist() == [provisional]


@pytest.mark.parametrize("provisional", [False, True])
def test_a_frame_that_mixes_real_time_and_other_hours_is_refused(provisional: bool) -> None:
    # A national total whose last day is real-time, as odre.national_hourly gives it.
    total, first, second = split(np.full(HOURS, 1000.0), {})
    status = np.where(np.arange(HOURS) < 48, "consolidated", "realtime")

    with pytest.raises(ValueError, match="real-time hours"):
        check_parts(
            with_status(hourly(total), status),
            hourly(first),
            hourly(second),
            provisional=provisional,
        )


def test_every_part_is_added() -> None:
    # A national total and its 12 regions; the last region falls 50 MWh short in one hour.
    regions = np.full((12, HOURS), 100.0)
    regions[11, 10] -= 50.0

    found = check_parts(hourly(np.full(HOURS, 1200.0)), *[hourly(region) for region in regions])

    assert spans(found) == [(hour(10), hour(11))]
    assert evidence(found)[0]["parts"] == 12


def test_series_must_be_on_utc_hours() -> None:
    total, first, second = (hourly(energy) for energy in split(np.full(HOURS, 1000.0), {}))

    with pytest.raises(SchemaError, match="UTC"):
        check_parts(total.tz_localize(None), first, second)


# Hours from 2022-01-30 00:00 UTC: hour 47, 2022-01-31 23:00 UTC, is the first of February in
# Berlin.
MONTH_CHANGE = pd.Timestamp("2022-01-30 00:00", tz="UTC")
FEBRUARY = 47
TIME_ZONE = "Europe/Berlin"
ORIGINAL = np.linspace(1000.0, 1950.0, 96)


def rescale(original_mwh: npt.ArrayLike) -> FloatArray:
    """`original_mwh` times 1.10 in January and 0.95 in February, Berlin time."""
    original = np.asarray(original_mwh, dtype=float)
    return original * np.where(np.arange(len(original)) < FEBRUARY, 1.10, 0.95)


def check_ratio(
    rescaled_mwh: npt.ArrayLike, original_mwh: npt.ArrayLike, provisional: bool = False
) -> pd.DataFrame:
    return proportional(
        hourly(rescaled_mwh, MONTH_CHANGE),
        hourly(original_mwh, MONTH_CHANGE),
        time_zone=TIME_ZONE,
        series=SERIES,
        provenance=PROVENANCE,
        provisional=provisional,
    )


def month_hour(number: int) -> pd.Timestamp:
    return hour(number, MONTH_CHANGE)


def test_a_new_factor_from_the_first_hour_of_a_berlin_month_gives_no_finding() -> None:
    found = check_ratio(rescale(ORIGINAL), ORIGINAL)

    assert found.empty
    Findings.validate(found)


def test_a_one_hour_departure_from_the_factor_of_the_month_is_a_finding() -> None:
    rescaled = rescale(ORIGINAL)
    rescaled[60] *= 1.02

    found = check_ratio(rescaled, ORIGINAL, provisional=True)

    assert spans(found) == [(month_hour(60), month_hour(61))]
    assert found["kind"].tolist() == ["cross_source_gap"]
    assert found["score"].tolist() == [pytest.approx(0.02)]
    assert evidence(found) == [
        {
            "time_zone": TIME_ZONE,
            "monthly_factors": {"2022-02": pytest.approx(0.95)},
            "hours": 1,
            "largest_departure": pytest.approx(0.02),
        }
    ]
    assert found["series"].tolist() == [SERIES]
    assert found["provenance"].tolist() == [PROVENANCE]
    assert found["provisional"].tolist() == [True]


@pytest.mark.parametrize(
    ("departure", "flagged"), [(1.009, False), (1.011, True), (0.991, False), (0.989, True)]
)
def test_a_departure_counts_beyond_1_percent_of_the_factor(departure: float, flagged: bool) -> None:
    rescaled = rescale(ORIGINAL)
    rescaled[60] *= departure

    found = check_ratio(rescaled, ORIGINAL)

    assert spans(found) == ([(month_hour(60), month_hour(61))] if flagged else [])


@pytest.mark.parametrize(("original_mwh", "flagged"), [(49.75, False), (50.0, True)])
def test_hours_under_50_mwh_in_the_original_are_not_compared(
    original_mwh: float, flagged: bool
) -> None:
    original = ORIGINAL.copy()
    original[60] = original_mwh
    rescaled = rescale(original)
    rescaled[60] *= 1.10

    found = check_ratio(rescaled, original)

    assert spans(found) == ([(month_hour(60), month_hour(61))] if flagged else [])


def test_hours_under_50_mwh_in_the_original_do_not_set_the_factor() -> None:
    # Most February hours are under 50 MWh, where the rescaled series is 20 % above the factor.
    original = ORIGINAL.copy()
    original[FEBRUARY:80] = 20.0
    rescaled = rescale(original)
    rescaled[FEBRUARY:80] *= 1.20

    assert check_ratio(rescaled, original).empty


@pytest.mark.parametrize("missing", ["rescaled", "original"])
def test_an_hour_that_either_series_lacks_is_not_compared(missing: str) -> None:
    original, rescaled = ORIGINAL.copy(), rescale(ORIGINAL)
    rescaled[60] *= 1.10
    if missing == "rescaled":
        rescaled[60] = np.nan
    else:
        original[60] = np.nan

    assert check_ratio(rescaled, original).empty


@pytest.mark.parametrize(
    ("later_hour", "expected"),
    [
        # 23 hours without a departure between the two: one finding over the hours between them.
        (74, [(50, 75, 2)]),
        (75, [(50, 51, 1), (75, 76, 1)]),
    ],
)
def test_departures_less_than_a_day_apart_are_one_finding(
    later_hour: int, expected: list[tuple[int, int, int]]
) -> None:
    rescaled = rescale(ORIGINAL)
    rescaled[[50, later_hour]] *= 1.05

    found = check_ratio(rescaled, ORIGINAL)

    assert spans(found) == [(month_hour(start), month_hour(end)) for start, end, _ in expected]
    assert [departure["hours"] for departure in evidence(found)] == [
        hours for _, _, hours in expected
    ]


def test_a_finding_over_two_months_gives_the_factor_of_each() -> None:
    rescaled = rescale(ORIGINAL)
    rescaled[FEBRUARY - 3] *= 1.05
    rescaled[FEBRUARY + 3] *= 1.02

    found = check_ratio(rescaled, ORIGINAL)

    assert spans(found) == [(month_hour(FEBRUARY - 3), month_hour(FEBRUARY + 4))]
    assert found["score"].tolist() == [pytest.approx(0.05)]
    assert evidence(found)[0]["monthly_factors"] == {
        "2022-01": pytest.approx(1.10),
        "2022-02": pytest.approx(0.95),
    }


def test_a_month_where_the_rescaled_series_is_mostly_zero_has_no_factor() -> None:
    rescaled = rescale(ORIGINAL)
    rescaled[FEBRUARY:80] = 0.0

    assert check_ratio(rescaled, ORIGINAL).empty


def test_both_series_must_be_on_utc_hours() -> None:
    rescaled, original = hourly(rescale(ORIGINAL)), hourly(ORIGINAL)

    with pytest.raises(SchemaError, match="UTC"):
        proportional(
            rescaled,
            original.tz_localize(None),
            time_zone=TIME_ZONE,
            series=SERIES,
            provenance=PROVENANCE,
            provisional=False,
        )
