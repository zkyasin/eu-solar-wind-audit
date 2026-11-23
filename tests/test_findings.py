import json

import numpy as np
import pandas as pd
import pytest
from pandera.errors import SchemaError, SchemaErrors

from eu_solar_wind_audit.anomalies.findings import Findings, from_records, runs


def utc(stamp: str) -> pd.Timestamp:
    return pd.Timestamp(stamp, tz="UTC")


def hours(*stamps: str) -> pd.DatetimeIndex:
    """Stamps of 27 March 2022 in UTC, in nanoseconds like grid.UtcTime."""
    return pd.DatetimeIndex([f"2022-03-27 {stamp}" for stamp in stamps], tz="UTC").as_unit("ns")


def record(**changes: object) -> dict[str, object]:
    return {
        "series": "odre.FR.solar",
        "provenance": "RTE, éCO2mix, ODRÉ dataset eco2mix-national-cons-def",
        "start": utc("2022-03-27 00:00"),
        "end": utc("2022-03-27 02:00"),
        "kind": "gap",
        "score": 2.0,
        "evidence": {"absent_stamps": 4},
        **changes,
    }


def frame(**changes: object) -> pd.DataFrame:
    row = record(**changes)
    row["evidence"] = json.dumps(row["evidence"])
    found = pd.DataFrame([row])
    for column in ("start", "end"):
        found[column] = found[column].astype("datetime64[ns, UTC]")
    return found


def test_findings_model_accepts_a_valid_frame() -> None:
    Findings.validate(frame())


@pytest.mark.parametrize(
    ("found", "error"),
    [
        pytest.param(frame(kind="spike"), "isin", id="unknown-kind"),
        pytest.param(frame(end=utc("2022-03-27 00:00")), "ends_after_start", id="empty-span"),
        pytest.param(frame(end=utc("2022-03-26 23:00")), "ends_after_start", id="reversed-span"),
        pytest.param(frame(start=utc("2022-03-27 00:30")), "on_the_hour", id="start-off-hour"),
        pytest.param(frame(end=utc("2022-03-27 01:45")), "on_the_hour", id="end-off-hour"),
        pytest.param(frame(evidence=[1, 2]), "json_object", id="evidence-not-an-object"),
        pytest.param(frame().assign(evidence="{absent: 4}"), "json_object", id="evidence-not-json"),
        pytest.param(
            frame().assign(evidence='{"ratio": NaN}'), "json_object", id="evidence-with-nan"
        ),
        pytest.param(frame(score=np.nan), "non-nullable", id="no-score"),
        pytest.param(frame(series=""), "str_length", id="no-series"),
        pytest.param(frame(provenance=""), "str_length", id="no-provenance"),
        pytest.param(
            frame().assign(start=pd.Timestamp("2022-03-27 00:00")),
            r"datetime64\[ns, UTC\]",
            id="naive-start",
        ),
    ],
)
def test_findings_model_rejects(found: pd.DataFrame, error: str) -> None:
    with pytest.raises(SchemaError, match=error):
        Findings.validate(found)


def test_findings_model_rejects_an_undeclared_column() -> None:
    # pandera reports an undeclared column as SchemaErrors, even without lazy validation.
    with pytest.raises(SchemaErrors, match="price"):
        Findings.validate(frame().assign(price=42.0))


def test_records_without_findings_give_an_empty_valid_frame() -> None:
    found = from_records([])

    assert found.empty
    assert list(found.columns) == list(Findings.to_schema().columns)
    Findings.validate(found)


def test_records_are_stored_in_nanoseconds_with_float_scores() -> None:
    microseconds = utc("2022-03-27 00:00").as_unit("us")

    found = from_records([record(start=microseconds, score=2)])

    assert str(found["start"].dtype) == "datetime64[ns, UTC]"
    assert found["start"].item() == utc("2022-03-27 00:00")
    assert found["score"].item() == 2.0


def test_evidence_mapping_is_stored_as_a_json_object() -> None:
    found = from_records([record(evidence={"time_zone": "Europe/Paris", "series": 3})])

    # Keys are sorted, so that the same finding always gives the same text.
    assert found["evidence"].item() == '{"series": 3, "time_zone": "Europe/Paris"}'


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_evidence_outside_json_is_refused(value: float) -> None:
    with pytest.raises(ValueError, match="not JSON compliant"):
        from_records([record(evidence={"ratio": value})])


def test_records_with_naive_times_are_refused() -> None:
    with pytest.raises(TypeError, match="timezone-naive"):
        from_records([record(start=pd.Timestamp("2022-03-27 00:00"))])


def test_records_are_validated() -> None:
    with pytest.raises(SchemaError, match="ends_after_start"):
        from_records([record(end=utc("2022-03-26 00:00"))])


def expected_runs(*spans: tuple[str, str, int]) -> pd.DataFrame:
    """Runs from (start, end, flagged hours) on 27 March 2022."""
    return pd.DataFrame(
        {
            "start": hours(*[start for start, _, _ in spans]),
            "end": hours(*[end for _, end, _ in spans]),
            "flagged_hours": pd.Series([flagged for _, _, flagged in spans], dtype="int64"),
        }
    )


def test_consecutive_hours_form_one_run() -> None:
    found = runs(hours("01:00", "02:00", "03:00"), merge_gap=pd.Timedelta(0))

    pd.testing.assert_frame_equal(found, expected_runs(("01:00", "04:00", 3)))


def test_an_unflagged_hour_splits_runs_without_a_merge_gap() -> None:
    found = runs(hours("01:00", "03:00", "04:00"), merge_gap=pd.Timedelta(0))

    expected = expected_runs(("01:00", "02:00", 1), ("03:00", "05:00", 2))
    pd.testing.assert_frame_equal(found, expected)


def test_runs_closer_than_the_merge_gap_are_joined() -> None:
    # One unflagged hour between the first two runs, two between the last two.
    found = runs(hours("01:00", "03:00", "06:00"), merge_gap=pd.Timedelta(hours=2))

    expected = expected_runs(("01:00", "04:00", 2), ("06:00", "07:00", 1))
    pd.testing.assert_frame_equal(found, expected)


def test_order_and_repeats_do_not_change_the_runs() -> None:
    found = runs(hours("03:00", "01:00", "02:00", "03:00"), merge_gap=pd.Timedelta(0))

    pd.testing.assert_frame_equal(found, expected_runs(("01:00", "04:00", 3)))


def test_no_flagged_hour_gives_no_run() -> None:
    found = runs(hours(), merge_gap=pd.Timedelta(hours=24))

    pd.testing.assert_frame_equal(found, expected_runs())


@pytest.mark.parametrize(
    "flagged",
    [
        pd.DatetimeIndex(["2022-03-27 01:00"]),
        pd.DatetimeIndex(["2022-03-27 01:00"], tz="Europe/Paris"),
        hours("01:30"),
    ],
    ids=["naive", "not-in-utc", "off-hour"],
)
def test_runs_need_hour_starts_in_utc(flagged: pd.DatetimeIndex) -> None:
    with pytest.raises(ValueError, match="hour starts in UTC"):
        runs(flagged, merge_gap=pd.Timedelta(0))
