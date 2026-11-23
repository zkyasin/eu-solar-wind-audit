"""Residual scores of a published hourly series against its model (evaluation plan, axis 2): how
far the published output of each hour is from the modelled output at the local level of the
series.

For an hour of the UTC day d, with y the published and m the modelled energy in MWh and C the
modelled capacity in MW at the start of the hour, the residual and the score are

    r = ln((y + 0.01 C) / (L_d m + 0.01 C)),    z = r / s_d,

where the level L_d is the exponential of the median of the daily log ratios ln(sum y / sum m),
each over the usable hours of one day, of the 31 days centred on d, and the scale s_d is 1.4826
times the median of abs(r) over the usable hours of the same 31 days, each residual at the level
of its own day (detector design). Adding 1 % of the capacity keeps the residual of zero output
finite; an hour where y + 0.01 C is not positive has no residual.

Usable hours are complete in the published and the modelled series, from consolidated or
definitive rows, with the reference price class (evaluation plan, Deviations) and a positive
modelled capacity; solar hours also need a modelled capacity factor of SOLAR_FACTOR_FROM or more
(detector design).

The level of a day is final once the inputs cover the 31 days centred on it. Its scale is final
once every residual of its window has a final level, that is once the inputs cover the 61 days
centred on the day; until then, the score of the day is deferred. The inputs cover the UTC days
whose hours all lie between the first and the last hour with both a published value from
consolidated or definitive rows and a modelled value, so a detection run that loads 30 days before
and after its period scores all of it.
"""

import numpy as np
import pandas as pd
import pandera.pandas as pa
from pandera.typing import DataFrame, Index, Series

from eu_solar_wind_audit import grid, physics
from eu_solar_wind_audit.anomalies.outliers import Technology
from eu_solar_wind_audit.sources import odre

DAY = pd.Timedelta(days=1)
HOURS_PER_DAY = 24

# Share of the capacity added to both energies, set beforehand and not tuned (detector design).
CAPACITY_SHARE = 0.01
# The level and the scale of a day come from the 31 days centred on it (detector design).
WINDOW_DAYS = 31
HALF_WINDOW_DAYS = WINDOW_DAYS // 2
# Solar hours count from this modelled capacity factor (detector design), which leaves out the
# first and last hours of daylight.
SOLAR_FACTOR_FROM = 0.05
# 1.4826 times the median of abs(r) estimates the standard deviation of a normal r centred on 0.
NORMAL_SCALE = 1.4826
# A level needs the daily ratios of 7 days of its window, and a scale the residuals of 168 hours: a
# week of each, as in the measurements of the detector design.
LEVEL_MIN_DAYS = 7
SCALE_MIN_HOURS = 7 * HOURS_PER_DAY


class ResidualScores(pa.DataFrameModel):
    """Residual and score of the hour starting at `time`, as the module docstring defines them.

    `usable` tells whether the hour counts, and `deferred` whether the score of its day waits for
    inputs that cover the 61 days centred on the day. `published_mwh` is the published energy,
    `expected_mwh` the modelled energy at the level of the day and `scale` the scale of the day.
    Each value is null until it is final, and where it cannot be computed.
    """

    time: Index[grid.UtcTime] = pa.Field(unique=True, check_name=True)
    usable: Series[bool]
    deferred: Series[bool]
    published_mwh: Series[float] = pa.Field(nullable=True)
    expected_mwh: Series[float] = pa.Field(nullable=True)
    residual: Series[float] = pa.Field(nullable=True)
    scale: Series[float] = pa.Field(gt=0, nullable=True)
    score: Series[float] = pa.Field(nullable=True)

    @pa.dataframe_check
    @classmethod
    def scores_of_usable_hours_of_final_days(cls, frame: pd.DataFrame) -> "pd.Series[bool]":
        return frame["score"].isna() | (frame["usable"] & ~frame["deferred"])

    class Config:
        strict = True


def residual_scores(
    published: pd.DataFrame,
    modelled: pd.DataFrame,
    price_classes: pd.DataFrame,
    *,
    technology: Technology,
) -> DataFrame[ResidualScores]:
    """`ResidualScores` of each hour of `published`, a series of `technology` on the grid of
    grid.Hourly or, with the status of each hour, of odre.Hourly.

    `modelled` is the physics.Modelled output of the same perimeter, from the uncalibrated model,
    and `price_classes` gives the `regime` of each UTC hour, as regimes.classify_hours does.
    """
    published, realtime = _published(published)
    model = physics.Modelled.validate(modelled)
    regime = price_classes["regime"]
    # Classes on other hours would leave every hour out without an error.
    if not isinstance(regime.index, pd.DatetimeIndex) or str(regime.index.tz) != "UTC":
        raise ValueError("expected price classes on a DatetimeIndex in UTC")
    hours = pd.DatetimeIndex(published.index).sort_values()
    if hours.empty:
        return ResidualScores.empty()

    # Every hour of the UTC days of `published`, so that each day has its 24 hours.
    days = pd.date_range(hours[0].floor("D"), hours[-1].floor("D"), freq="D", unit="ns")
    on_days = pd.date_range(
        days[0], days[-1] + DAY, freq="h", inclusive="left", unit="ns", name="time"
    )
    published_mwh = published["energy_mwh"].reindex(on_days)
    modelled_mwh = model["energy_mwh"].reindex(on_days)
    capacity_mw = model["capacity_mw"].reindex(on_days)
    final = (published["complete"] & ~realtime).reindex(on_days, fill_value=False)
    with_values = final & modelled_mwh.notna()
    usable = with_values & (capacity_mw > 0) & regime.reindex(on_days).eq("reference")
    if technology == "solar":
        usable &= modelled_mwh >= SOLAR_FACTOR_FROM * capacity_mw

    covered = on_days[with_values.to_numpy()]
    # Both are NaT, which covers no day, when no hour has both values.
    first, last = covered.min().ceil("D"), (covered.max() + grid.HOUR).floor("D") - DAY
    half = HALF_WINDOW_DAYS * DAY
    level_final = (days - half >= first) & (days + half <= last)
    scale_final = (days - 2 * half >= first) & (days + 2 * half <= last)

    output = published_mwh.where(usable)
    day = on_days.floor("D")
    daily_output = output.groupby(day).sum()
    daily_model = modelled_mwh.where(usable).groupby(day).sum()
    ratio = (daily_output / daily_model).where((daily_output > 0) & (daily_model > 0))
    log_ratio = pd.Series(np.log(ratio.to_numpy()), index=days)
    median = log_ratio.rolling(WINDOW_DAYS, center=True, min_periods=LEVEL_MIN_DAYS).median()
    level = np.exp(median.where(level_final).to_numpy())

    shift = CAPACITY_SHARE * capacity_mw.to_numpy()
    expected_mwh = modelled_mwh.to_numpy() * np.repeat(level, HOURS_PER_DAY)
    shifted_output = output.to_numpy() + shift
    # A shifted output that is not positive has no logarithm.
    shifted_output[~(shifted_output > 0)] = np.nan
    residual = pd.Series(np.log(shifted_output / (expected_mwh + shift)), index=on_days)
    # The window of a day is the 744 hours that end with the last hour of the 15th day after it.
    window_median = (
        residual.abs()
        .rolling(WINDOW_DAYS * HOURS_PER_DAY, min_periods=SCALE_MIN_HOURS)
        .median()
        .iloc[HOURS_PER_DAY - 1 :: HOURS_PER_DAY]
        .set_axis(days)
        .shift(-HALF_WINDOW_DAYS)
    )
    scale = (NORMAL_SCALE * window_median).where((window_median > 0) & scale_final)
    scale_of_hour = np.repeat(scale.to_numpy(), HOURS_PER_DAY)

    table = pd.DataFrame(
        {
            "usable": usable,
            "deferred": np.repeat(~scale_final, HOURS_PER_DAY),
            "published_mwh": published_mwh,
            "expected_mwh": expected_mwh,
            "residual": residual,
            "scale": scale_of_hour,
            "score": residual / scale_of_hour,
        },
        index=on_days,
    )
    return ResidualScores.validate(table.reindex(hours))


def _published(published: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """`published` validated, and whether each of its hours is real-time."""
    if "status" in published.columns:
        with_status = odre.Hourly.validate(published)
        return with_status, with_status["status"].eq("realtime")
    hours = grid.Hourly.validate(published)
    return hours, pd.Series(False, index=hours.index)
