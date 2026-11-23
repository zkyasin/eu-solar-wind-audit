"""Timestamp checks of a published solar series against its model (evaluation plan, axis 2).

The checks compare, day by day, the centroid in time of the published and of the modelled output
over the usable hours of each UTC day: hours complete in both curves, from consolidated or
definitive rows, with the reference price class (evaluation plan, Deviations). The residual of a
day is the published minus the modelled centroid, minus the centroid bias of the model. Only valid
days, sunny enough and mostly usable, are judged.

`shifts` claims shifts of the timestamps by one hour or more, and `clock_change_jumps` tests the
residual for jumps at the clock changes of the time zone of the source. Both are statistical
detectors: their thresholds rest on 2019-2022, and their findings count in the false-alarm budget
(evaluation plan, Deviations). Real-time hours are left out, so no finding is provisional. Scores:
`timestamp_shift`, the median residual in hours of the days that size the shift, which can be
fewer than the days of the finding; `clock_change_jump`, the jump of the residual in hours.
"""

import numpy as np
import numpy.typing as npt
import pandas as pd
import pandera.pandas as pa
from pandera.typing import DataFrame, Index, Series

from eu_solar_wind_audit import grid
from eu_solar_wind_audit.anomalies import findings
from eu_solar_wind_audit.anomalies.findings import Findings
from eu_solar_wind_audit.sources import odre

DAY = pd.Timedelta(days=1)
HOURS_PER_DAY = 24

# Centroid bias of the uncalibrated physics model: the median residual of éCO2mix national solar
# over the valid days of 2019-2022 (detector design). It belongs to the model version, not to a
# series, so that a series one hour off from its first day is still caught.
BIAS_MINUTES = -8.4
# A day is valid when its usable hours hold at least 80 % of the modelled daily energy, and that
# energy is at least half of its highest value over the 31 days centred on the day, which leaves
# out cloudy days (detector design).
USABLE_SHARE_FROM = 0.8
PEAK_SHARE_FROM = 0.5
PEAK_WINDOW_DAYS = 31
# A shift is decided on two consecutive valid days off by 30 minutes or more with the same sign
# (detector design), and claimed only when these days are off by 45 minutes or more on average
# (evaluation plan, Deviations).
DECIDE_FROM_MINUTES = 30.0
CLAIM_FROM_MINUTES = 45.0
# Runs of days off with one sign and at most 3 valid days between them are one finding, whose size
# is the median residual of its first 7 days off from the deciding days (detector design).
MERGE_GAP_DAYS = 3
SIZE_DAYS = 7
# A jump compares the median residuals of the valid days in the 14 days before and after a clock
# change, with at least 3 on each side, and counts from 30 minutes (detector design).
JUMP_WINDOW_DAYS = 14
JUMP_MIN_VALID_DAYS = 3
JUMP_FROM_MINUTES = 30.0


class Centroids(pa.DataFrameModel):
    """Timing of the published and the modelled output over the UTC day starting at `day`.

    A centroid is in hours after midnight UTC: sum((h + 0.5) E_h) / sum(E_h) over the usable hours
    h of the day, null when the usable energy of the curve is not positive. `usable_share` is the
    share of the modelled daily energy in the usable hours, and `peak_share` the modelled daily
    energy over its highest value in the PEAK_WINDOW_DAYS days centred on the day; both are null
    when the model lacks an hour of the day. `residual_minutes` is the published minus the
    modelled centroid, minus the centroid bias of the model. Only `valid` days are judged.
    """

    day: Index[grid.UtcTime] = pa.Field(unique=True, check_name=True)
    published_centroid_hours: Series[float] = pa.Field(nullable=True)
    modelled_centroid_hours: Series[float] = pa.Field(nullable=True)
    usable_share: Series[float] = pa.Field(nullable=True)
    peak_share: Series[float] = pa.Field(nullable=True)
    residual_minutes: Series[float] = pa.Field(nullable=True)
    valid: Series[bool]

    @pa.check("day")
    @classmethod
    def at_midnight(cls, day: "pd.Series[pd.Timestamp]") -> "pd.Series[bool]":
        return day == day.dt.floor("D")

    @pa.dataframe_check
    @classmethod
    def valid_days_have_a_residual(cls, frame: pd.DataFrame) -> "pd.Series[bool]":
        return ~frame["valid"] | frame["residual_minutes"].notna()

    class Config:
        strict = True


def daily_centroids(
    published: pd.DataFrame,
    modelled: pd.DataFrame,
    price_classes: pd.DataFrame,
    *,
    bias_minutes: float = BIAS_MINUTES,
    usable_share_from: float = USABLE_SHARE_FROM,
    peak_share_from: float = PEAK_SHARE_FROM,
) -> DataFrame[Centroids]:
    """`Centroids` of each UTC day with an hour in `published`, a solar series of grid.Hourly or,
    with the status of each hour, of odre.Hourly.

    `modelled` is the modelled output of the same perimeter, a frame of grid.Hourly such as
    physics.Modelled, and `price_classes` gives the `regime` of each UTC hour, as
    regimes.classify_hours does. An hour is usable when it is complete in `published` and in
    `modelled`, its status is not real-time, and its price class is reference. A day is valid when
    its usable share reaches `usable_share_from`, its peak share reaches `peak_share_from`, and both
    centroids exist. `bias_minutes` is the centroid bias of the model.
    """
    published, realtime = _published(published)
    hours = pd.DatetimeIndex(published.index)
    modelled_mwh = grid.Hourly.validate(modelled[["energy_mwh", "complete"]])["energy_mwh"]
    regime = price_classes["regime"]
    # Classes on other hours would leave every hour out without an error.
    if not isinstance(regime.index, pd.DatetimeIndex) or str(regime.index.tz) != "UTC":
        raise ValueError("expected price classes on a DatetimeIndex in UTC")
    on_hours = modelled_mwh.reindex(hours)
    usable = (
        published["complete"] & ~realtime & on_hours.notna() & regime.reindex(hours).eq("reference")
    )

    day = hours.floor("D").rename("day")
    position = pd.Series(hours.hour + 0.5, index=hours)
    published_centroid = _centroid(published["energy_mwh"].where(usable), position, day)
    modelled_centroid = _centroid(on_hours.where(usable), position, day)

    model_days = pd.DatetimeIndex(modelled_mwh.index).floor("D")
    daily_mwh = modelled_mwh.groupby(model_days).sum(min_count=HOURS_PER_DAY)
    # The window counts calendar days, whether the model covers them or not.
    peak_mwh = daily_mwh.asfreq("D").rolling(PEAK_WINDOW_DAYS, center=True, min_periods=1).max()
    days = published_centroid.index
    daily = daily_mwh.reindex(days)
    usable_share = on_hours.where(usable).groupby(day).sum() / daily.where(daily > 0)
    peak = peak_mwh.reindex(days)
    peak_share = daily / peak.where(peak > 0)

    table = pd.DataFrame(
        {
            "published_centroid_hours": published_centroid,
            "modelled_centroid_hours": modelled_centroid,
            "usable_share": usable_share,
            "peak_share": peak_share,
            "residual_minutes": (published_centroid - modelled_centroid) * 60 - bias_minutes,
            "valid": (
                (usable_share >= usable_share_from)
                & (peak_share >= peak_share_from)
                & published_centroid.notna()
                & modelled_centroid.notna()
            ),
        }
    )
    table.index = pd.DatetimeIndex(days, name="day").as_unit("ns")
    return Centroids.validate(table)


def shifts(
    centroids: pd.DataFrame,
    *,
    series: str,
    provenance: str,
    decide_from_minutes: float = DECIDE_FROM_MINUTES,
    claim_from_minutes: float = CLAIM_FROM_MINUTES,
) -> DataFrame[Findings]:
    """Timestamp shifts of one hour or more in the solar series `series`, whose rows come from
    `provenance`, from its `Centroids`.

    The valid days are read in time order. A day is off when its residual reaches
    `decide_from_minutes` in absolute value. Consecutive days off with one sign form a run, and
    runs of one sign with at most MERGE_GAP_DAYS valid days between them form one finding, from its
    first to its last day off. The finding is decided by its first two consecutive days off that
    average `claim_from_minutes` or more in absolute value, and claimed when the median residual of
    the SIZE_DAYS days off from these two does too: that median, rounded to whole hours, is the
    size of the shift. Smaller offsets stay in the centroid table only. The evidence gives the size
    in hours, the median residual, the two deciding days with their mean residual, and the number
    of days off.
    """
    table = Centroids.validate(centroids).sort_index()
    valid = table[table["valid"]]
    residual = valid["residual_minutes"].to_numpy()
    days = pd.DatetimeIndex(valid.index)
    groups = [
        (sign, positions)
        for sign in (1, -1)
        for positions in _groups(np.flatnonzero(sign * residual >= decide_from_minutes))
    ]
    records: list[dict[str, object]] = []
    for sign, positions in sorted(groups, key=lambda group: int(group[1][0])):
        consecutive = np.diff(positions) == 1
        pair_means = (residual[positions[:-1]] + residual[positions[1:]]) / 2
        deciding = np.flatnonzero(consecutive & (sign * pair_means >= claim_from_minutes))
        if deciding.size == 0:
            continue
        first = deciding[0]
        # Days off before the deciding days are often a smaller offset, or noise.
        median = float(np.median(residual[positions[first : first + SIZE_DAYS]]))
        if sign * median < claim_from_minutes:
            continue
        records.append(
            {
                "series": series,
                "provenance": provenance,
                "provisional": False,
                "start": days[positions[0]],
                "end": days[positions[-1]] + DAY,
                "kind": "timestamp_shift",
                "score": median / 60,
                "evidence": {
                    "shift_hours": round(median / 60),
                    "median_residual_minutes": median,
                    "deciding_days": [
                        days[positions[first]].date().isoformat(),
                        days[positions[first + 1]].date().isoformat(),
                    ],
                    "deciding_mean_minutes": float(pair_means[first]),
                    "days_off": int(positions.size),
                },
            }
        )
    return findings.from_records(records)


def clock_change_jumps(
    centroids: pd.DataFrame,
    *,
    time_zone: str,
    series: str,
    provenance: str,
    jump_from_minutes: float = JUMP_FROM_MINUTES,
) -> DataFrame[Findings]:
    """Jumps of the residual at the clock changes of `time_zone`, the time zone of the source, in
    the solar series `series`, whose rows come from `provenance`, from its `Centroids`.

    For a change on the UTC day D, the jump is the median residual of the valid days in
    (D, D + JUMP_WINDOW_DAYS] minus that of the valid days in [D - JUMP_WINDOW_DAYS, D). It is
    tested when each side has at least JUMP_MIN_VALID_DAYS valid days, and gives a finding over
    these days when it reaches `jump_from_minutes` in absolute value. A source that labels local
    time as UTC gives a jump of one hour at each change. The evidence gives the time zone, the
    instant of the change, the jump, and the median residual and the number of valid days on each
    side.
    """
    table = Centroids.validate(centroids).sort_index()
    residual = table.loc[table["valid"], "residual_minutes"]
    if residual.empty:
        return findings.from_records([])
    days = pd.DatetimeIndex(residual.index)
    window = JUMP_WINDOW_DAYS * DAY
    records: list[dict[str, object]] = []
    for change in _clock_changes(time_zone, days[0], days[-1] + DAY):
        day = change.floor("D")
        before = residual[(days >= day - window) & (days < day)]
        after = residual[(days > day) & (days <= day + window)]
        if min(len(before), len(after)) < JUMP_MIN_VALID_DAYS:
            continue
        jump = after.median() - before.median()
        if abs(jump) < jump_from_minutes:
            continue
        records.append(
            {
                "series": series,
                "provenance": provenance,
                "provisional": False,
                "start": before.index[0],
                "end": after.index[-1] + DAY,
                "kind": "clock_change_jump",
                "score": jump / 60,
                "evidence": {
                    "time_zone": time_zone,
                    "clock_change": change.isoformat(),
                    "jump_minutes": jump,
                    "median_before_minutes": before.median(),
                    "median_after_minutes": after.median(),
                    "valid_days_before": len(before),
                    "valid_days_after": len(after),
                },
            }
        )
    return findings.from_records(records)


def _published(published: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """`published` validated, and whether each of its hours is real-time."""
    if "status" in published.columns:
        with_status = odre.Hourly.validate(published)
        return with_status, with_status["status"].eq("realtime")
    hours = grid.Hourly.validate(published)
    return hours, pd.Series(False, index=hours.index)


def _groups(off: npt.NDArray[np.intp]) -> list[npt.NDArray[np.intp]]:
    """The positions `off` of the valid days off with one sign, in groups that more than
    MERGE_GAP_DAYS valid days separate; groups of a single day are left out, since a claim needs
    two."""
    valid_days_between = np.diff(off) - 1
    groups = np.split(off, np.flatnonzero(valid_days_between > MERGE_GAP_DAYS) + 1)
    return [positions for positions in groups if positions.size > 1]


def _clock_changes(time_zone: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    """The UTC hours of [start, end) from which the UTC offset of `time_zone` changes."""
    hours = pd.date_range(start, end, freq="h", inclusive="left", unit="ns")
    utc_offset = hours.tz_convert(time_zone).tz_localize(None) - hours.tz_localize(None)
    return hours[1:][utc_offset[1:] != utc_offset[:-1]]


def _centroid(energy: pd.Series, position: pd.Series, day: pd.DatetimeIndex) -> pd.Series:
    """Centroid of `energy` over each day, at the `position` of each hour, null when the energy of
    the day is not positive."""
    total = energy.groupby(day).sum()
    moment = (energy * position).groupby(day).sum()
    return moment / total.where(total > 0)
