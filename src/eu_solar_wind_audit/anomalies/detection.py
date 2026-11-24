"""A detection run of the evaluation plan, axis 2: every detector on the inputs of one country,
with the thresholds fixed on 2019-2022, keeping the findings that reach the hours [start, end) it
judges.

Rule-based detectors run on every series: completeness on the raw rows of each dataset; negative
values, values above capacity and frozen values on each part of a series; the national totals
against the sum of their regions or control areas, and Energy-Charts public power against SMARD.
An offshore wind series with an offshore mask is checked on the hours it flags only, since éCO2mix
publishes zeros for the regions and the months without offshore farms. Values above capacity are
checked where the capacity is positive: the SDES capacity of a region grows from zero over the
quarter of its first farm.

Statistical detectors run on the series with a model, the solar and onshore wind of the areas of
France and Germany: timestamp shifts and clock-change jumps of the solar series that the thresholds
make eligible, residual outliers, and the level breaks of the solar and the wind series of each
area. They read the wind of an area as `onshore_wind` gives it, with the thresholds and the profile
of its wind series (detector design), so that no hour is judged twice.
"""

import hashlib
import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import pandas as pd
from pandera.typing import DataFrame

from eu_solar_wind_audit import grid, physics
from eu_solar_wind_audit.anomalies import (
    breaks,
    completeness,
    cross_source,
    outliers,
    residuals,
    timestamp,
)
from eu_solar_wind_audit.anomalies.findings import Findings
from eu_solar_wind_audit.anomalies.inputs import (
    SMARD_AREAS,
    DetectionInputs,
    Published,
    SeriesInputs,
)
from eu_solar_wind_audit.anomalies.thresholds import Thresholds
from eu_solar_wind_audit.sources import energy_charts, odre, smard

# The local time of the clock changes of each country.
TIME_ZONES = {"FR": odre.TIMEZONE, "DE": energy_charts.TIMEZONE}


@dataclass(frozen=True)
class Detection:
    """Findings of a run that reach its hours [start, end), in order of start, series and kind,
    and the level breaks of these hours left unattributed."""

    findings: DataFrame[Findings]
    unattributed: DataFrame[breaks.Unattributed]


class Period(NamedTuple):
    """Hours [start, end) whose findings are named after `series`, from rows of `provenance`."""

    series: str
    provenance: str
    start: pd.Timestamp
    end: pd.Timestamp


class OnshoreWind(NamedTuple):
    """The wind of an area as the statistical detectors read it: its hours, its physics.Modelled
    output, the hours that contain offshore output, and the periods of its findings."""

    published: pd.DataFrame
    modelled: DataFrame[physics.Modelled]
    offshore: pd.Series
    periods: tuple[Period, ...]


def detect(
    found: DetectionInputs, fixed: Thresholds | None, start: pd.Timestamp, end: pd.Timestamp
) -> Detection:
    """Run every detector on `found` and keep the findings that reach the hours [start, end), both
    hour starts in UTC, which lie within [found.start, found.end).

    `fixed` holds the thresholds of the statistical detectors and the profile of each modelled
    series; it may be None only when no series has a model. The level breaks need the daily ratios
    from breaks.FIRST_DAY on, so French inputs start there.
    """
    grid.check_period(start, end)
    if start < found.start or end > found.end:
        raise ValueError(f"the inputs judge {found.start} to {found.end}, not {start} to {end}")
    frames = [*_completeness(found, start, end), *_rule_based(found), *_cross_source(found)]
    unattributed: list[pd.DataFrame] = []
    modelled_solar = [
        series
        for series, item in found.series.items()
        if item.technology == "solar" and item.modelled is not None
    ]
    if modelled_solar:
        if fixed is None:
            raise ValueError("the statistical detectors need the thresholds of the country")
        for series in modelled_solar:
            area_findings, area_unattributed = _statistical(
                found, fixed, series.removesuffix(".solar"), start, end
            )
            frames += area_findings
            unattributed += area_unattributed
    return Detection(_reaching(frames, start, end), _unattributed(unattributed))


def onshore_wind(found: DetectionInputs, area: str) -> OnshoreWind:
    """The wind of `area`, such as odre.FR, as the statistical detectors read it: its wind series
    up to breaks.WIND_SWITCH and, when the inputs have it, its onshore wind series from it. The
    hours of the wind series that contain offshore output are flagged. Findings before the switch
    are named after the wind series, those from it after the onshore wind series. An area without
    a wind series, such as smard.DE, reads its onshore wind series throughout.
    """
    wind_id, onshore_id = f"{area}.wind", f"{area}.wind_onshore"
    if wind_id not in found.series:
        wind_id = onshore_id
    wind = found.series[wind_id]
    published = _hours(wind)
    modelled = _modelled(wind, wind_id)
    if wind.offshore is None:
        raise ValueError(f"{wind_id} has no offshore mask")
    offshore = wind.offshore
    first, last = found.hours[0], found.hours[-1] + grid.HOUR
    onshore = found.series.get(onshore_id)
    if wind_id == onshore_id or onshore is None or last <= breaks.WIND_SWITCH:
        whole = Period(wind_id, wind.published[0].provenance, first, last)
        return OnshoreWind(published, modelled, offshore, (whole,))
    switch = breaks.WIND_SWITCH
    from_switch = _hours(onshore)
    published = pd.concat(
        [published[published.index < switch], from_switch[from_switch.index >= switch]]
    )
    periods = (
        Period(wind_id, wind.published[0].provenance, first, switch),
        Period(onshore_id, onshore.published[0].provenance, switch, last),
    )
    return OnshoreWind(published, modelled, offshore & (offshore.index < switch), periods)


def describe(
    found: DetectionInputs, detected: Detection, start: pd.Timestamp, end: pd.Timestamp
) -> dict[str, Any]:
    """What a run judged, read and found: its hours; each part of each series, with its first and
    last hour, its complete hours, its statuses and the SHA-256 of its hourly frame (of
    pandas.util.hash_pandas_object); the ERA5 releases and the provisional capacity hours of each
    model; the price regimes; and the number of findings by kind and of unattributed breaks by
    reason."""
    return {
        "country": found.country,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "first_hour": found.hours[0].isoformat(),
        "last_hour": found.hours[-1].isoformat(),
        "price_regimes": _counts(found.price_classes["regime"]),
        "series": {series: _series(item) for series, item in sorted(found.series.items())},
        "findings": _counts(detected.findings["kind"]),
        "unattributed": _counts(detected.unattributed["reason"]),
    }


def write(
    directory: Path,
    found: DetectionInputs,
    detected: Detection,
    start: pd.Timestamp,
    end: pd.Timestamp,
    run: dict[str, Any],
) -> list[Path]:
    """Write to `directory` the findings, the unattributed breaks and a JSON file with `describe`
    and `run`, what the caller adds about the thresholds and the code, named after the country."""
    directory.mkdir(parents=True, exist_ok=True)
    paths = [
        directory / f"{found.country}-findings.parquet",
        directory / f"{found.country}-unattributed.parquet",
        directory / f"{found.country}-detection.json",
    ]
    detected.findings.to_parquet(paths[0])
    detected.unattributed.to_parquet(paths[1])
    content = describe(found, detected, start, end) | run
    text = json.dumps(content, indent=1, ensure_ascii=False, allow_nan=False)
    paths[2].write_text(text + "\n", encoding="utf-8")
    return paths


def _completeness(
    found: DetectionInputs, start: pd.Timestamp, end: pd.Timestamp
) -> Iterator[pd.DataFrame]:
    """Completeness findings of the rows of each dataset over the hours of [start, end) that the
    dataset publishes."""
    for rows in found.rows:
        first, last = max(rows.start, start), min(rows.end, end)
        if first < last:
            yield completeness.completeness(
                rows.values,
                first,
                last,
                step=rows.step,
                label=rows.label,
                dataset=rows.dataset,
                provenance=rows.provenance,
                provisional=rows.provisional,
                local_days=rows.local_days,
            )


def _rule_based(found: DetectionInputs) -> Iterator[pd.DataFrame]:
    """Findings of the outlier rules on each part of each series."""
    for series, item in found.series.items():
        technology: outliers.Technology = "solar" if item.technology == "solar" else "wind"
        for part in item.published:
            hourly = _checked_hours(part.hourly, item)
            if hourly.empty:
                continue
            yield outliers.negative_values(hourly, series=series, provenance=part.provenance)
            if item.capacity_mw is not None:
                with_capacity = (item.capacity_mw.reindex(hourly.index) > 0).to_numpy()
                yield outliers.above_capacity(
                    hourly[with_capacity],
                    item.capacity_mw,
                    series=series,
                    provenance=part.provenance,
                )
            yield outliers.frozen_values(
                hourly,
                series=series,
                provenance=part.provenance,
                technology=technology,
                modelled=item.modelled,
            )


def _checked_hours(hourly: pd.DataFrame, item: SeriesInputs) -> pd.DataFrame:
    """The hours of `hourly`, a part of `item`, that the outlier rules check: all of them but, for
    an offshore wind series with an offshore mask, the hours before the first offshore output of
    its area."""
    if item.technology != "wind_offshore" or item.offshore is None:
        return hourly
    return hourly[item.offshore.reindex(hourly.index, fill_value=False).to_numpy()]


def _cross_source(found: DetectionInputs) -> Iterator[pd.DataFrame]:
    """Findings of the national totals against their parts, and of Energy-Charts public power
    against SMARD."""
    if found.country == "FR":
        for technology in ("solar", "wind"):
            regions = [f"odre.FR-{code}.{technology}" for code in odre.REGIONS]
            yield _sum_of_parts(found, f"odre.FR.{technology}", regions)
    elif found.country == "DE":
        for technology in smard.FEED_IN:
            areas = smard.OFFSHORE_REGIONS if technology == "wind_offshore" else SMARD_AREAS
            parts = [f"smard.{area}.{technology}" for area in areas if area != "DE"]
            yield _sum_of_parts(found, f"smard.DE.{technology}", parts)
            public = found.series[f"energy_charts.DE.public_power.{technology}"].published[0]
            original = found.series[f"smard.DE.{technology}"].published[0]
            yield cross_source.proportional(
                public.hourly,
                original.hourly,
                time_zone=energy_charts.TIMEZONE,
                series=public.series,
                provenance=_joined([public, original]),
                provisional=False,
            )


def _sum_of_parts(found: DetectionInputs, total: str, parts: list[str]) -> pd.DataFrame:
    """The first part of the series `total`, its consolidated rows for éCO2mix, against the sum
    of the series `parts`, of one part each."""
    whole = found.series[total].published[0]
    pieces = [found.series[name].published[0] for name in parts]
    return cross_source.sum_of_parts(
        whole.hourly,
        {piece.series: piece.hourly for piece in pieces},
        series=total,
        provenance=_joined([whole, *pieces]),
        provisional=whole.provisional,
    )


def _statistical(
    found: DetectionInputs,
    fixed: Thresholds,
    area: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    """Findings of the statistical detectors on the solar and the wind of `area`, and its level
    breaks of [start, end) left unattributed."""
    solar_id = f"{area}.solar"
    solar = found.series[solar_id]
    solar_hours, solar_model = _hours(solar), _modelled(solar, solar_id)
    solar_provenance = solar.published[0].provenance
    wind = onshore_wind(found, area)
    classes = found.price_classes
    frames: list[pd.DataFrame] = []

    if solar_id in fixed.shift_series:
        centroids = timestamp.daily_centroids(
            solar_hours, solar_model, classes, bias_minutes=fixed.centroid_bias_minutes
        )
        frames.append(timestamp.shifts(centroids, series=solar_id, provenance=solar_provenance))
        frames.append(
            timestamp.clock_change_jumps(
                centroids,
                time_zone=TIME_ZONES[found.country],
                series=solar_id,
                provenance=solar_provenance,
            )
        )

    solar_scores = residuals.residual_scores(solar_hours, solar_model, classes, technology="solar")
    frames.append(
        residuals.residual_outliers(
            solar_scores,
            series=solar_id,
            provenance=solar_provenance,
            threshold=fixed.residual_thresholds["solar"],
        )
    )
    wind_scores = residuals.residual_scores(
        wind.published, wind.modelled, classes, technology="wind", offshore=wind.offshore
    )
    score_hours = pd.DatetimeIndex(wind_scores.index)
    for period in wind.periods:
        frames.append(
            residuals.residual_outliers(
                wind_scores[(score_hours >= period.start) & (score_hours < period.end)],
                series=period.series,
                provenance=period.provenance,
                threshold=fixed.residual_thresholds["wind"],
            )
        )

    ratios: dict[breaks.Technology, breaks.Ratios] = {
        "solar": breaks.Ratios(
            breaks.daily_log_ratios(solar_hours, solar_model, classes, technology="solar"),
            _profile(fixed, solar_id),
        ),
        "wind": breaks.Ratios(
            breaks.daily_log_ratios(
                wind.published, wind.modelled, classes, technology="wind", offshore=wind.offshore
            ),
            _profile(fixed, wind.periods[0].series),
        ),
    }
    junctions = pd.concat(
        [
            breaks.junctions(solar_hours, solar_model),
            breaks.junctions(wind.published, wind.modelled),
        ],
        ignore_index=True,
    )
    unattributed: list[pd.DataFrame] = []
    for period in wind.periods:
        first, last = max(start, period.start), min(end, period.end)
        if first >= last:
            continue
        level = breaks.level_breaks(
            ratios,
            junctions,
            series={"solar": solar_id, "wind": period.series},
            provenance={"solar": solar_provenance, "wind": period.provenance},
            penalties=fixed.penalties,
            start=first,
            end=last,
        )
        frames.append(level.findings)
        unattributed.append(level.unattributed)
    return frames, unattributed


def _reaching(
    frames: Iterable[pd.DataFrame], start: pd.Timestamp, end: pd.Timestamp
) -> DataFrame[Findings]:
    """The findings of `frames` that reach the hours [start, end), in order."""
    filled = [frame for frame in frames if not frame.empty]
    if not filled:
        return Findings.empty()
    table = pd.concat(filled, ignore_index=True)
    table = table[(table["end"] > start) & (table["start"] < end)]
    return Findings.validate(table.sort_values(["start", "series", "kind"], ignore_index=True))


def _unattributed(frames: Iterable[pd.DataFrame]) -> DataFrame[breaks.Unattributed]:
    filled = [frame for frame in frames if not frame.empty]
    if not filled:
        return breaks.Unattributed.empty()
    table = pd.concat(filled, ignore_index=True)
    return breaks.Unattributed.validate(table.sort_values(["start", "series"], ignore_index=True))


def _hours(item: SeriesInputs) -> pd.DataFrame:
    """The hours of all the parts of a series, in time order."""
    return pd.concat([part.hourly for part in item.published]).sort_index()


def _modelled(item: SeriesInputs, series: str) -> DataFrame[physics.Modelled]:
    if item.modelled is None:
        raise ValueError(f"{series} has no model")
    return item.modelled


def _profile(fixed: Thresholds, series: str) -> breaks.SeasonalProfile:
    if series not in fixed.profiles:
        raise ValueError(f"the thresholds have no profile for {series}")
    return fixed.profiles[series]


def _joined(parts: Iterable[Published]) -> str:
    """The provenances of `parts`, each once, in order."""
    return "; ".join(dict.fromkeys(part.provenance for part in parts))


def _series(item: SeriesInputs) -> dict[str, Any]:
    parts = [
        {
            "provenance": part.provenance,
            "provisional": part.provisional,
            "first_hour": part.hourly.index[0].isoformat(),
            "last_hour": part.hourly.index[-1].isoformat(),
            "complete_hours": int(part.hourly["complete"].sum()),
            "statuses": _counts(part.hourly["status"]) if "status" in part.hourly else None,
            "sha256": _sha256(part.hourly),
        }
        for part in item.published
        if not part.hourly.empty
    ]
    model = None
    if item.modelled is not None:
        model = {
            "era5_sources": _counts(item.modelled["era5_source"]),
            "provisional_capacity_hours": int(item.modelled["capacity_provisional"].sum()),
        }
    return {"technology": item.technology, "parts": parts, "model": model}


def _counts(values: pd.Series) -> dict[str, int]:
    return {str(value): int(count) for value, count in values.value_counts().sort_index().items()}


def _sha256(frame: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(frame).to_numpy().tobytes()).hexdigest()
