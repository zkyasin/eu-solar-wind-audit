"""Findings of the anomaly detectors of the evaluation plan, axis 2.

A finding describes the published rows of one series, or of every series of a dataset, over the
hours [start, end) in UTC. It is a data report: `provenance` cites the producer and the dataset of
the rows as their source. `evidence` is a JSON object, without NaN or Infinity, with the facts a
reader can check against the published data. It never holds hourly ENTSO-E values, whose terms
allow only derived values, and it gives prices only as price classes (docs/data-licences.md).
"""

import json
from collections.abc import Iterable, Mapping

import pandas as pd
import pandera.pandas as pa
from pandera.typing import DataFrame, Series

from eu_solar_wind_audit import grid

# The detectors of the plan, axis 2, in the order of its table.
KINDS = [
    # Completeness
    "gap",
    "duplicate",
    "off_grid",
    "resolution_change",
    "clock_change_convention",
    # Timestamp
    "timestamp_shift",
    "clock_change_jump",
    # Outliers
    "residual_outlier",
    "above_capacity",
    "negative",
    "frozen",
    # Level breaks
    "level_break",
    # Cross-source gaps: a total that differs from the sum of its parts, and two sources of one
    # quantity that differ
    "parts_gap",
    "cross_source_gap",
]


class Findings(pa.DataFrameModel):
    """One finding per row."""

    # `<source>.<area>.<quantity>` for a series, such as odre.FR-84.wind, or `<source>.<dataset>`
    # for a finding on every series of a dataset.
    series: Series[str] = pa.Field(str_length={"min_value": 1})
    # Producer and dataset of the rows, such as "RTE, éCO2mix, ODRÉ dataset eco2mix-national-tr".
    provenance: Series[str] = pa.Field(str_length={"min_value": 1})
    # Whether the rows are real-time values, which their producer replaces later.
    provisional: Series[bool]
    start: Series[grid.UtcTime]
    end: Series[grid.UtcTime]
    kind: Series[str] = pa.Field(isin=KINDS)
    # Size of the finding, in the unit its detector gives for the kind.
    score: Series[float]
    evidence: Series[str]

    @pa.check("start", "end")
    @classmethod
    def on_the_hour(cls, time: "pd.Series[pd.Timestamp]") -> "pd.Series[bool]":
        return time == time.dt.floor("h")

    @pa.check("evidence", element_wise=True)
    @classmethod
    def json_object(cls, evidence: str) -> bool:
        try:
            return isinstance(json.loads(evidence, parse_constant=_not_json), dict)
        except ValueError:
            return False

    @pa.dataframe_check
    @classmethod
    def ends_after_start(cls, frame: pd.DataFrame) -> "pd.Series[bool]":
        return frame["end"] > frame["start"]

    class Config:
        strict = True


def from_records(records: Iterable[Mapping[str, object]]) -> DataFrame[Findings]:
    """Validated findings from one mapping per finding, with the evidence as a mapping.

    Times are stored in nanoseconds, the unit of grid.UtcTime, whatever unit they come in.
    """
    rows = [{**record, "evidence": _json(record["evidence"])} for record in records]
    if not rows:
        return Findings.empty()
    frame = pd.DataFrame(rows)
    for column in ("start", "end"):
        frame[column] = frame[column].astype("datetime64[ns, UTC]")
    frame["score"] = frame["score"].astype(float)
    return Findings.validate(frame)


def _json(evidence: object) -> str:
    # Sorted keys, so that the same finding always gives the same text.
    return json.dumps(evidence, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _not_json(constant: str) -> object:
    # Python reads NaN and Infinity, which are not JSON and which strict parsers refuse.
    raise ValueError(f"{constant} is not JSON")


def runs(hours: pd.DatetimeIndex, merge_gap: pd.Timedelta) -> pd.DataFrame:
    """Runs of consecutive flagged hours: the start, the exclusive end and the number of flagged
    hours of each, in time order.

    `hours` are the starts of the flagged hours in UTC, in any order and possibly repeated. Runs
    separated by less than `merge_gap` are joined into one; a zero gap joins nothing.
    """
    if str(hours.tz) != "UTC" or (hours != hours.floor("h")).any():
        raise ValueError("expected hour starts in UTC")
    flagged = pd.Series(hours.unique().sort_values().as_unit("ns"))
    unflagged_before = flagged.diff() - grid.HOUR
    new_run = (unflagged_before > pd.Timedelta(0)) & (unflagged_before >= merge_gap)
    by_run = flagged.groupby(new_run.cumsum())
    return pd.DataFrame(
        {"start": by_run.min(), "end": by_run.max() + grid.HOUR, "flagged_hours": by_run.size()}
    ).reset_index(drop=True)
