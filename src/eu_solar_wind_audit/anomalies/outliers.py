"""Outliers of a published hourly series (evaluation plan, axis 2): negative values, values above
1.05 times the capacity and values frozen for 24 hours or more.

These rules state facts about the published values, so they use no thresholds file and no
false-alarm budget (evaluation plan, Deviations). The detectors read one series on the grid of
`grid`, as a `grid.Hourly` frame or, with the publication status of each hour, an `odre.Hourly`
frame, and look at its complete hours only. Findings on real-time hours are provisional, and no
finding spans real-time hours and consolidated or definitive ones. Scores: `negative`, the lowest
value in MWh; `above_capacity`, the highest ratio of a value to the capacity; `frozen`, the length
of the run in hours.
"""

from typing import Literal, NamedTuple

import pandas as pd
from pandera.typing import DataFrame

from eu_solar_wind_audit import grid, physics
from eu_solar_wind_audit.anomalies import findings
from eu_solar_wind_audit.anomalies.findings import Findings
from eu_solar_wind_audit.sources import odre

# Values above 1.05 times the AC capacity, and values frozen for 24 hours or more (evaluation
# plan, axis 2).
CAPACITY_LIMIT = 1.05
FROZEN_HOURS = 24
# Runs of negative values, or of values above the capacity, closer than a week form one finding,
# since negative values recur every night. A choice of the detector design: the plan sets no merge
# gap.
MERGE_GAP = pd.Timedelta(days=7)
# A run of zeros is frozen only when the model says that the source should have produced. The
# floors were measured on 2019-2022 with physics.model, over the 13 éCO2mix series of mainland
# France and its regions.
# Solar: energy of the run in hours at full capacity. The 18,245 published runs of zero solar
# output, nights that reach at most into the first or last hours of daylight and last at most 18
# hours, held up to 0.128 (Île-de-France, 22 to 23 January 2019). A floor of 0.2 leaves out at
# most 0.8 % of the 24-hour windows of a region and none of mainland France.
SOLAR_ZERO_FLOOR_HOURS = 0.2
# Wind: capacity factor of the run, its energy over its capacity. The 10 runs of zero wind output
# of 24 hours or more (Île-de-France 2, Provence-Alpes-Côte d'Azur 8) reached 0.041, and the 120
# runs of 12 hours or more 0.050. A floor of 0.06 leaves out 2.2 % of the 24-hour windows of
# mainland France and 5 to 22 % of a region.
WIND_ZERO_FLOOR_FACTOR = 0.06

# Wind covers onshore, offshore and published wind series.
Technology = Literal["solar", "wind"]


class Run(NamedTuple):
    """Hours [start, end) of one finding, `flagged_hours` of them flagged, whose published rows are
    all real-time or none (`provisional`)."""

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


def above_capacity(
    published: pd.DataFrame, capacity_mw: pd.Series, *, series: str, provenance: str
) -> DataFrame[Findings]:
    """Runs of complete hours of `published` whose energy exceeds CAPACITY_LIMIT times the capacity
    at the start of the hour; runs closer than MERGE_GAP are joined. `published` is the hourly
    series `series`, whose rows come from `provenance`.

    `capacity_mw` is the AC capacity in MW of the published perimeter at each hour start, as
    sdes.hourly_capacity and energy_charts.hourly_capacity give it; it must be positive at every
    complete hour. The evidence gives the number of hours above the limit and, at the hour with
    the highest ratio, the energy, the capacity and their ratio.
    """
    published, realtime = _hours(published)
    energy = published["energy_mwh"]
    complete = published["complete"]
    capacity = capacity_mw.reindex(published.index)
    if not (capacity[complete] > 0).all():
        raise ValueError("expected a positive capacity at every complete hour")
    flagged = complete & (energy > CAPACITY_LIMIT * capacity)
    ratio = energy / capacity
    records = []
    for run in _runs(flagged, realtime):
        ratios = ratio[run.start : run.end - grid.HOUR].dropna()
        highest = int(ratios.argmax())
        hour = pd.DatetimeIndex(ratios.index)[highest]
        records.append(
            _record(
                series,
                provenance,
                run,
                "above_capacity",
                ratios.iloc[highest],
                hours=run.flagged_hours,
                highest_hour=hour.isoformat(),
                energy_mwh=energy[hour],
                capacity_mw=capacity[hour],
                ratio=ratios.iloc[highest],
            )
        )
    return findings.from_records(records)


def frozen_values(
    published: pd.DataFrame,
    *,
    series: str,
    provenance: str,
    technology: Technology,
    modelled: pd.DataFrame | None = None,
) -> DataFrame[Findings]:
    """Runs of FROZEN_HOURS or more consecutive complete hours of `published` with one value, one
    finding each. `published` is the hourly series `series` of `technology`, whose rows come from
    `provenance`.

    A run of zeros is left out when `modelled`, a physics.Modelled frame of the same perimeter, has
    every hour of the run and stays below the zero floor of the technology over it: the source
    could then have produced nothing. Without `modelled`, every run of zeros is reported. The
    evidence gives the repeated value and, for a run of zeros judged by the model, the modelled
    output.
    """
    published, realtime = _hours(published)
    if modelled is not None:
        modelled = physics.Modelled.validate(modelled).sort_index()
    hours = published.index[published["complete"].to_numpy()]
    energy = published.loc[hours, "energy_mwh"]
    provisional = realtime[hours]
    # A run ends at a change of value, at a missing or incomplete hour, and between real-time and
    # other hours.
    new_run = (
        (energy != energy.shift())
        | (hours.to_series().diff() != grid.HOUR)
        | (provisional != provisional.shift())
    )
    records = []
    for _, values in energy.groupby(new_run.cumsum()):
        if len(values) < FROZEN_HOURS:
            continue
        start, end = values.index[0], values.index[-1] + grid.HOUR
        evidence = {"hours": len(values), "value_mwh": values.iloc[0]}
        if values.iloc[0] == 0 and modelled is not None:
            output = _modelled_output(modelled, technology, start, end)
            if output is not None:
                name, amount, floor = output
                if amount < floor:
                    continue
                evidence[name] = amount
        run = Run(start, end, len(values), bool(provisional[start]))
        records.append(_record(series, provenance, run, "frozen", len(values), **evidence))
    return findings.from_records(records)


def _modelled_output(
    modelled: pd.DataFrame, technology: Technology, start: pd.Timestamp, end: pd.Timestamp
) -> tuple[str, float, float] | None:
    """The modelled output over [start, end) that a run of zeros is judged on, with its name in the
    evidence and the zero floor of `technology`; None unless the model has every hour."""
    hours = modelled.loc[start : end - grid.HOUR]
    if len(hours) < (end - start) / grid.HOUR or not hours["complete"].all():
        return None
    capacity = hours["capacity_mw"].mean()
    full_load_hours = float(hours["energy_mwh"].sum() / capacity) if capacity > 0 else 0.0
    if technology == "solar":
        return "modelled_full_load_hours", full_load_hours, SOLAR_ZERO_FLOOR_HOURS
    return "modelled_capacity_factor", full_load_hours / len(hours), WIND_ZERO_FLOOR_FACTOR


def _hours(published: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """`published`, validated and in time order, and whether each of its hours is real-time."""
    if "status" in published.columns:
        with_status = odre.Hourly.validate(published).sort_index()
        # An hour without rows, such as the hour that éCO2mix leaves out at each autumn clock
        # change, has no status. It takes the status of the hour before it, so that it never
        # splits a run of real-time hours.
        return with_status, with_status["status"].ffill().eq("realtime")
    hours = grid.Hourly.validate(published).sort_index()
    return hours, pd.Series(False, index=hours.index)


def _runs(flagged: pd.Series, realtime: pd.Series) -> list[Run]:
    """Runs of the flagged hours, as findings.runs gives them with MERGE_GAP, never joined across a
    change between real-time hours and consolidated or definitive ones."""
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
