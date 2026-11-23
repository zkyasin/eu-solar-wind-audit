"""Timestamp checks of a published solar series against its model (evaluation plan, axis 2).

The checks compare, day by day, the centroid in time of the published and of the modelled output
over the usable hours of each UTC day: hours complete in both curves, from consolidated or
definitive rows, with the reference price class (evaluation plan, Deviations). The residual of a
day is the published minus the modelled centroid, minus the centroid bias of the model. Only valid
days, sunny enough and mostly usable, are judged.
"""

import pandas as pd
import pandera.pandas as pa
from pandera.typing import DataFrame, Index, Series

from eu_solar_wind_audit import grid
from eu_solar_wind_audit.sources import odre

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


def _published(published: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """`published` validated, and whether each of its hours is real-time."""
    if "status" in published.columns:
        with_status = odre.Hourly.validate(published)
        return with_status, with_status["status"].eq("realtime")
    hours = grid.Hourly.validate(published)
    return hours, pd.Series(False, index=hours.index)


def _centroid(energy: pd.Series, position: pd.Series, day: pd.DatetimeIndex) -> pd.Series:
    """Centroid of `energy` over each day, at the `position` of each hour, null when the energy of
    the day is not positive."""
    total = energy.groupby(day).sum()
    moment = (energy * position).groupby(day).sum()
    return moment / total.where(total > 0)
