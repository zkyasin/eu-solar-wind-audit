"""Completeness of the raw rows of a dataset against the layout its source documents (evaluation
plan, axis 2): gaps and repeated stamps.

The detector reads the rows as fetched, before the hourly conversion of `grid`, which drops
repeated stamps, so that their hours only look incomplete. Its findings are facts and use no
threshold. Scores: `gap`, hours that lack a value; `duplicate`, stamps that come more than once.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import numpy.typing as npt
import pandas as pd
from pandera.typing import DataFrame

from eu_solar_wind_audit import grid
from eu_solar_wind_audit.anomalies import findings
from eu_solar_wind_audit.anomalies.findings import Findings

NO_MERGING = pd.Timedelta(0)

# The periods a value covers, as the offsets in steps from its stamp to the starts of the first
# and the last of them. An instantaneous value bounds the period before it and the one after it.
COVERED: dict[grid.Label, tuple[int, int]] = {"start": (0, 0), "end": (-1, -1), "instant": (-1, 0)}


class Run(NamedTuple):
    """Consecutive hours [start, end) whose values need some of `stamps`."""

    start: pd.Timestamp
    end: pd.Timestamp
    hours: int
    stamps: pd.DatetimeIndex


@dataclass(frozen=True)
class Layout:
    """Rows every `step`, stamped as `label` says, checked over the hours [start, end)."""

    start: pd.Timestamp
    end: pd.Timestamp
    step: pd.Timedelta
    label: grid.Label

    def hours(self, stamps: pd.DatetimeIndex) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
        """First and last hour that the value of each stamp covers."""
        first, last = COVERED[self.label]
        return (stamps + first * self.step).floor("h"), (stamps + last * self.step).floor("h")

    def expected(self) -> pd.DatetimeIndex:
        """Stamps on the grid whose values cover the hours."""
        first, last = COVERED[self.label]
        return pd.date_range(
            self.start - last * self.step,
            self.end - first * self.step,
            freq=self.step,
            inclusive="left",
            unit="ns",
        )

    def covering(self, values: pd.Series) -> pd.Series:
        """The rows of `values` that cover some of the hours, stamped in nanoseconds."""
        stamps = pd.DatetimeIndex(values.index)
        if str(stamps.tz) != "UTC":
            raise ValueError("expected a DatetimeIndex in UTC")
        first, last = self.hours(stamps)
        rows = values[(last >= self.start) & (first < self.end)]
        return rows.set_axis(pd.DatetimeIndex(rows.index).as_unit("ns"))

    def runs(self, stamps: pd.DatetimeIndex) -> list[Run]:
        """Runs of the hours that `stamps` cover, each with its stamps in time order."""
        stamps = stamps.sort_values()
        first, last = self.hours(stamps)
        hours = pd.DatetimeIndex(first.append(last))
        spans = findings.runs(hours[(hours >= self.start) & (hours < self.end)], NO_MERGING)
        return [
            Run(start, end, int(flagged), stamps[(last >= start) & (first < end)])
            for start, end, flagged in spans.itertuples(index=False)
        ]


def completeness(
    published: Mapping[str, pd.Series],
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    step: pd.Timedelta,
    label: grid.Label,
    provenance: str,
) -> DataFrame[Findings]:
    """Findings on the raw rows of the series of one dataset that cover the hours [start, end),
    both hour starts in UTC.

    `published` maps each series id to its values as fetched: indexed by UTC stamps, with repeated
    stamps kept and missing values as NaN. `step` and `label` give the layout of the rows.
    """
    grid.check_period(start, end)
    if grid.HOUR % step:
        raise ValueError(f"{step} does not divide an hour")
    layout = Layout(start, end, step, label)
    expected = layout.expected()

    records: list[dict[str, object]] = []
    for series, values in published.items():
        rows = layout.covering(values)
        on_step = _on_step(pd.DatetimeIndex(rows.index), step)
        stamps = pd.DatetimeIndex(rows.index)[on_step]
        counts = stamps.value_counts()
        repeated = pd.DatetimeIndex(counts.index[counts > 1])
        missing = expected.difference(stamps[rows[on_step].notna().to_numpy()])
        absent = missing.difference(stamps)

        for run in layout.runs(missing):
            absent_stamps = len(run.stamps.intersection(absent))
            empty_stamps = len(run.stamps) - absent_stamps
            records.append(
                _record(
                    series,
                    provenance,
                    run,
                    "gap",
                    run.hours,
                    absent_stamps=absent_stamps,
                    empty_stamps=empty_stamps,
                )
            )
        for run in layout.runs(repeated):
            rows_at = int(counts[run.stamps].sum())
            records.append(
                _record(series, provenance, run, "duplicate", len(run.stamps), rows=rows_at)
            )

    return findings.from_records(records)


def _on_step(stamps: pd.DatetimeIndex, step: pd.Timedelta) -> npt.NDArray[np.bool_]:
    """Whether each stamp lies on the `step` grid."""
    return np.asarray(stamps == stamps.floor(step))


def _record(
    series: str, provenance: str, run: Run, kind: str, score: float, **evidence: object
) -> dict[str, object]:
    return {
        "series": series,
        "provenance": provenance,
        "start": run.start,
        "end": run.end,
        "kind": kind,
        "score": score,
        "evidence": {
            "first_stamp": run.stamps[0].isoformat(),
            "last_stamp": run.stamps[-1].isoformat(),
            **evidence,
        },
    }
