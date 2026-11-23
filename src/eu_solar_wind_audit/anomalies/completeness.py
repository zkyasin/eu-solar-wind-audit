"""Completeness of the raw rows of a dataset against the layout its source documents (evaluation
plan, axis 2): gaps, repeated stamps, values off the grid and changes of resolution.

The detector reads the rows as fetched, before the hourly conversion of `grid`, which drops
repeated stamps, so that their hours only look incomplete, and refuses values off the grid. Its
findings are facts and use no threshold. Scores: `gap`, hours that lack a value; `duplicate`,
stamps that come more than once; `off_grid`, values off the grid; `resolution_change`, the spacing
in minutes that the values of a UTC day mostly have; `clock_change_convention`, clock changes
whose rows follow the convention of the dataset.

Values off the grid are reported, then left out by `on_grid`: a detection run converts the rows
that `on_grid` keeps, so that it goes on.
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
MINUTE = pd.Timedelta(minutes=1)
DAY = pd.Timedelta(days=1)

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
    dataset: str,
    provenance: str,
    local_days: str | None = None,
) -> DataFrame[Findings]:
    """Findings on the raw rows of the series of one dataset that cover the hours [start, end),
    both hour starts in UTC.

    `published` maps each series id to its values as fetched: indexed by UTC stamps, with repeated
    stamps kept and missing values as NaN. `step` and `label` give the layout of the rows.

    `local_days` is the time zone of a dataset that publishes the same rows for every local day,
    as éCO2mix does. In UTC, the stamps of the hour after a spring clock change then come twice,
    and those of the hour before an autumn change are missing. These rows give one
    `clock_change_convention` finding named `dataset` for all the series, instead of a duplicate
    and a gap in each of them.
    """
    grid.check_period(start, end)
    if grid.HOUR % step:
        raise ValueError(f"{step} does not divide an hour")
    layout = Layout(start, end, step, label)
    expected = layout.expected()
    after_spring, before_autumn = _clock_change_hours(local_days, start, end)

    records: list[dict[str, object]] = []
    conventional: list[pd.DatetimeIndex] = []
    for series, values in published.items():
        rows = layout.covering(values)
        on_step = _on_step(pd.DatetimeIndex(rows.index), step)
        stamps = pd.DatetimeIndex(rows.index)[on_step]
        counts = stamps.value_counts()
        repeated = pd.DatetimeIndex(counts.index[counts > 1])
        missing = expected.difference(stamps[rows[on_step].notna().to_numpy()])
        absent = missing.difference(stamps)
        # Stamps published twice in the hour after a spring change are not duplicates, and stamps
        # without a row in the hour before an autumn change are not gaps: they follow the
        # convention of the dataset. A repeated stamp whose rows are all empty is still a gap.
        twice = repeated[(counts[repeated] == 2).to_numpy()]
        repeated_in_spring = twice[twice.floor("h").isin(after_spring)]
        absent_in_autumn = absent[absent.floor("h").isin(before_autumn)]
        explained = repeated_in_spring.union(absent_in_autumn)
        if not explained.empty:
            conventional.append(explained)

        for run in layout.runs(missing.difference(absent_in_autumn)):
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
        for run in layout.runs(repeated.difference(repeated_in_spring)):
            rows_at = int(counts[run.stamps].sum())
            records.append(
                _record(series, provenance, run, "duplicate", len(run.stamps), rows=rows_at)
            )
        off_grid = pd.DatetimeIndex(rows.index)[~on_step & rows.notna().to_numpy()]
        for run in layout.runs(off_grid.unique()):
            values_off = int(off_grid.isin(run.stamps).sum())
            minutes = sorted({int(minute) for minute in run.stamps.minute})
            records.append(
                _record(series, provenance, run, "off_grid", values_off, minutes_past_hour=minutes)
            )
        records += _resolution_changes(series, provenance, rows, layout)

    if conventional:
        records.append(
            _convention(dataset, provenance, local_days, conventional, layout, after_spring)
        )
    return findings.from_records(records)


def on_grid(values: pd.Series, step: pd.Timedelta) -> pd.Series:
    """The rows of `values` stamped on the `step` grid: the hourly conversions of `grid` refuse
    the others, which `completeness` reports as `off_grid`."""
    return values[_on_step(pd.DatetimeIndex(values.index), step)]


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


def _resolution_changes(
    series: str, provenance: str, rows: pd.Series, layout: Layout
) -> list[dict[str, object]]:
    """One finding per run of UTC days whose values are mostly spaced otherwise than the step."""
    stamps = pd.DatetimeIndex(rows.index[rows.notna().to_numpy()]).unique().sort_values()
    # A spacing belongs to the day of the first hour that the later value covers.
    first, _ = layout.hours(stamps[1:])
    spacing = pd.Series(stamps[1:] - stamps[:-1], index=first.floor("D"))
    by_day = spacing.groupby(level=0)
    modal = by_day.agg(lambda day: day.mode().min())
    # A spacing is a resolution only on a day that its values fill: the few values left around a
    # long gap are not one.
    filled = by_day.size() + 1 >= DAY / modal
    changed = modal[(modal != layout.step) & filled]
    hours = pd.date_range(layout.start, layout.end, freq="h", inclusive="left", unit="ns")
    records: list[dict[str, object]] = []
    for observed, days in changed.groupby(changed):
        spans = findings.runs(hours[hours.floor("D").isin(days.index)], NO_MERGING)
        for first_hour, end in zip(spans["start"], spans["end"], strict=True):
            in_span = (days.index >= pd.Timestamp(first_hour).floor("D")) & (days.index < end)
            records.append(
                {
                    "series": series,
                    "provenance": provenance,
                    "start": first_hour,
                    "end": end,
                    "kind": "resolution_change",
                    "score": observed / MINUTE,
                    "evidence": {
                        "step_minutes": int(layout.step / MINUTE),
                        "days": int(in_span.sum()),
                    },
                }
            )
    return records


def _clock_change_hours(
    time_zone: str | None, start: pd.Timestamp, end: pd.Timestamp
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """The UTC hours after each spring clock change and before each autumn change of
    `time_zone`, within a day of [start, end); none without a time zone."""
    hours = pd.date_range(start - DAY, end + DAY, freq="h", unit="ns")
    if time_zone is None:
        return hours[:0], hours[:0]
    utc_offset = hours.tz_convert(time_zone).tz_localize(None) - hours.tz_localize(None)
    change = utc_offset[1:] - utc_offset[:-1]
    changed_at = hours[1:]
    return changed_at[change > pd.Timedelta(0)], changed_at[change < pd.Timedelta(0)] - grid.HOUR


def _convention(
    dataset: str,
    provenance: str,
    time_zone: str | None,
    conventional: list[pd.DatetimeIndex],
    layout: Layout,
    after_spring: pd.DatetimeIndex,
) -> dict[str, object]:
    """One finding for the rows of all the series that follow the clock-change convention."""
    stamps = pd.DatetimeIndex(conventional[0].append(conventional[1:])).unique()
    runs = layout.runs(stamps)
    stamp_hours = stamps.floor("h").unique().sort_values()
    in_spring = stamp_hours.isin(after_spring)
    return {
        "series": dataset,
        "provenance": provenance,
        "start": runs[0].start,
        "end": runs[-1].end,
        "kind": "clock_change_convention",
        "score": len(stamp_hours),
        "evidence": {
            "time_zone": time_zone,
            "repeated_stamps": [_interval(hour) for hour in stamp_hours[in_spring]],
            "missing_stamps": [_interval(hour) for hour in stamp_hours[~in_spring]],
            "series_count": len(conventional),
        },
    }


def _interval(hour: pd.Timestamp) -> str:
    """The hour from `hour` as an ISO 8601 interval."""
    return f"{hour.isoformat()}/{(hour + grid.HOUR).isoformat()}"
