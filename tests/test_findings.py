import json

import numpy as np
import pandas as pd
import pytest
from pandera.errors import SchemaError, SchemaErrors

from eu_solar_wind_audit.anomalies.findings import Findings, from_records


def utc(stamp: str) -> pd.Timestamp:
    return pd.Timestamp(stamp, tz="UTC")


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


def test_records_with_naive_times_are_refused() -> None:
    with pytest.raises(TypeError, match="timezone-naive"):
        from_records([record(start=pd.Timestamp("2022-03-27 00:00"))])


def test_records_are_validated() -> None:
    with pytest.raises(SchemaError, match="ends_after_start"):
        from_records([record(end=utc("2022-03-26 00:00"))])
