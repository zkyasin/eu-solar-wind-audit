"""Cross-source checks of the evaluation plan, axis 2: published series against other published
series of the same quantity.

Until ENTSO-E access is granted, only national sources are compared (evaluation plan,
Deviations). `sum_of_parts` compares a total with the sum of its parts, such as éCO2mix national
solar and wind with the 12 regions, and SMARD Germany with its four control areas (offshore wind:
50Hertz and TenneT).

The rules compare published values only and are fixed in advance: their findings are facts,
outside the false-alarm budget. Only the hours complete in every series are compared, and the
completeness detector reports the others. Runs of hours that break a rule are one finding when
less than a day separates them. Score of `sum_of_parts`: the largest gap in MWh.
"""

from collections.abc import Mapping

import numpy as np
import pandas as pd
from pandera.typing import DataFrame

from eu_solar_wind_audit import grid
from eu_solar_wind_audit.anomalies import findings
from eu_solar_wind_audit.anomalies.findings import Findings

# A gap counts beyond 10 MWh and beyond 1 % of the total. 10 MWh is above the rounding of the
# published values in an hour: 13 x 0.5 MWh for éCO2mix, whose national and 12 regional values
# are whole MW over half-hours, and 5 x 0.5 MWh for SMARD, whose quarter-hours of Germany and its
# four control areas came in steps of 0.25 MWh over 2019-2022.
GAP_FLOOR_MWH = 10.0
GAP_SHARE = 0.01

# Runs less than a day apart are one finding, so that a gap over the daylight hours of
# consecutive days is reported once.
MERGE_GAP = pd.Timedelta(days=1)


def sum_of_parts(
    total: pd.DataFrame,
    parts: Mapping[str, pd.DataFrame],
    *,
    series: str,
    provenance: str,
    provisional: bool,
) -> DataFrame[Findings]:
    """Findings named `series` over the hours where a published total and the sum of its
    published parts differ by more than max(10 MWh, 1 % of the total). Only the hours complete in
    the total and in every part are compared.

    `total` and the values of `parts` are hourly frames of `grid.Hourly` or of a model derived from
    it, such as `odre.Hourly`. `provenance` cites the producer and the datasets of the series, and
    `provisional` tells whether their rows are real-time values. The evidence gives the number of
    `parts`, the `hours` of the finding beyond the tolerance, the `largest_gap_mwh`, and the
    `largest_relative_gap`, a share of the absolute total, null when a total of zero leaves it
    unbounded.
    """
    total_mwh = _energy(total)
    parts_mwh = pd.concat(
        {name: _energy(part) for name, part in parts.items()}, axis="columns", sort=False
    )
    # NaN in every hour that some part lacks, so that the hour is not compared.
    sum_mwh = parts_mwh.reindex(total_mwh.index).sum(axis="columns", min_count=len(parts))
    gap_mwh = (total_mwh - sum_mwh).abs()
    beyond = gap_mwh[gap_mwh > (GAP_SHARE * total_mwh).clip(lower=GAP_FLOOR_MWH)]

    records: list[dict[str, object]] = []
    for start, end, run in _runs(beyond):
        relative = (run / total_mwh[run.index].abs()).max()
        records.append(
            {
                "series": series,
                "provenance": provenance,
                "provisional": provisional,
                "start": start,
                "end": end,
                "kind": "sum_of_parts",
                "score": run.max(),
                "evidence": {
                    "parts": len(parts),
                    "hours": len(run),
                    "largest_gap_mwh": run.max(),
                    "largest_relative_gap": relative if np.isfinite(relative) else None,
                },
            }
        )
    return findings.from_records(records)


def _energy(hourly: pd.DataFrame) -> pd.Series:
    """Energy in MWh of a frame of `grid.Hourly` or of a model derived from it, checked against
    `grid.Hourly`: NaN exactly in the incomplete hours."""
    return grid.Hourly.validate(hourly[["energy_mwh", "complete"]])["energy_mwh"]


def _runs(beyond: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Series]]:
    """Start, exclusive end and values of each run of the hours of `beyond`, a series indexed by
    hour starts in UTC; runs less than MERGE_GAP apart are joined."""
    hours = pd.DatetimeIndex(beyond.index)
    spans = findings.runs(hours, MERGE_GAP)
    return [
        (start, end, beyond[(hours >= start) & (hours < end)])
        for start, end in zip(spans["start"], spans["end"], strict=True)
    ]
