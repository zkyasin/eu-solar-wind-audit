"""Outliers of a published hourly series (evaluation plan, axis 2): negative values.

These rules state facts about the published values, so they use no thresholds file and no
false-alarm budget (evaluation plan, Deviations). The detectors read one series on the grid of
`grid`, as a `grid.Hourly` frame or, with the publication status of each hour, an `odre.Hourly`
frame, and look at its complete hours only. Findings on real-time hours are provisional, and no
finding spans real-time and other hours. Scores: `negative`, the lowest value in MWh.
"""

from typing import NamedTuple

import pandas as pd
from pandera.typing import DataFrame

from eu_solar_wind_audit import grid
from eu_solar_wind_audit.anomalies import findings
from eu_solar_wind_audit.anomalies.findings import Findings
from eu_solar_wind_audit.sources import odre

# Runs of flagged hours closer than a week form one finding, since negative values recur every
# night. A choice of the detector design: the plan sets no merge gap.
MERGE_GAP = pd.Timedelta(days=7)


class Run(NamedTuple):
    """Hours [start, end) of one finding, `flagged_hours` of them flagged, all real-time or none
    (`provisional`)."""

    start: pd.Timestamp
    end: pd.Timestamp
    flagged_hours: int
    provisional: bool


def negative_values(
    published: pd.DataFrame, *, series: str, provenance: str
) -> DataFrame[Findings]:
    """Runs of complete hours below zero in `published`, the hourly series `series` whose rows come
    from `provenance`; runs closer than MERGE_GAP are joined. The evidence gives the number of
    negative hours and the lowest value with its hour."""
    published, realtime = _hours(published)
    energy = published["energy_mwh"]
    flagged = published["complete"] & (energy < 0)
    records = []
    for run in _runs(flagged, realtime):
        values = energy[run.start : run.end - grid.HOUR].dropna()
        lowest = int(values.argmin())
        records.append(
            _record(
                series,
                provenance,
                run,
                "negative",
                values.iloc[lowest],
                hours=run.flagged_hours,
                lowest_hour=pd.DatetimeIndex(values.index)[lowest].isoformat(),
                lowest_mwh=values.iloc[lowest],
            )
        )
    return findings.from_records(records)


def _hours(published: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """`published`, validated and in time order, and whether each of its hours is real-time."""
    if "status" in published.columns:
        with_status = odre.Hourly.validate(published).sort_index()
        return with_status, with_status["status"].eq("realtime")
    hours = grid.Hourly.validate(published).sort_index()
    return hours, pd.Series(False, index=hours.index)


def _runs(flagged: pd.Series, realtime: pd.Series) -> list[Run]:
    """Runs of the flagged hours, as findings.runs gives them with MERGE_GAP, never joined across a
    change between real-time and other hours."""
    # Blocks of consecutive hours that are all real-time, or all not.
    block = realtime.ne(realtime.shift()).cumsum()
    runs = []
    for _, in_block in flagged.groupby(block):
        hours = pd.DatetimeIndex(in_block.index[in_block.to_numpy()])
        provisional = bool(realtime[in_block.index[0]])
        spans = findings.runs(hours, MERGE_GAP)
        runs += [
            Run(start, end, int(count), provisional)
            for start, end, count in spans.itertuples(index=False)
        ]
    return runs


def _record(
    series: str, provenance: str, run: Run, kind: str, score: float, **evidence: object
) -> dict[str, object]:
    return {
        "series": series,
        "provenance": provenance,
        "provisional": run.provisional,
        "start": run.start,
        "end": run.end,
        "kind": kind,
        "score": score,
        "evidence": evidence,
    }
