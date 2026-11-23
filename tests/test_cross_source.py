import json
from collections.abc import Mapping

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
from pandera.errors import SchemaError

from eu_solar_wind_audit import grid
from eu_solar_wind_audit.anomalies.cross_source import sum_of_parts
from eu_solar_wind_audit.anomalies.findings import Findings

SERIES = "test.DE.solar"
PROVENANCE = "Producer, test datasets"
FIRST_HOUR = pd.Timestamp("2022-06-01 00:00", tz="UTC")
HOURS = 72

FloatArray = npt.NDArray[np.float64]


def hour(number: int) -> pd.Timestamp:
    return FIRST_HOUR + number * grid.HOUR


def hourly(energy_mwh: npt.ArrayLike) -> pd.DataFrame:
    """A frame of grid.Hourly from FIRST_HOUR on; NaN hours are incomplete."""
    energy = np.asarray(energy_mwh, dtype=float)
    time = pd.date_range(FIRST_HOUR, periods=len(energy), freq="h", unit="ns", name="time")
    frame = pd.DataFrame({"energy_mwh": energy, "complete": ~np.isnan(energy)}, index=time)
    return grid.Hourly.validate(frame)


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
