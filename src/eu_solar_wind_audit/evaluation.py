"""Axis-1 model quality: how closely each model follows the published hourly output of a year it
was not fitted on (evaluation plan, "Axis 1: model quality").

Models are compared on a `Predictions` frame, which leave-one-year-out folds produce: hours of
held-out years with their published energy, AC capacity, the energy each model gives and the groups
that metrics are given for.
"""

from collections.abc import Hashable
from typing import ClassVar

import numpy as np
import pandas as pd
import pandera.pandas as pa
from pandera.typing import DataFrame, Index, Series

from eu_solar_wind_audit import grid

# DJF holds January, February and December of the same year.
SEASONS = {"DJF": (12, 1, 2), "MAM": (3, 4, 5), "JJA": (6, 7, 8), "SON": (9, 10, 11)}
SEASON_OF_MONTH = {month: season for season, months in SEASONS.items() for month in months}
SKY_CLASSES = ["clear", "mixed", "overcast"]
# Every energy column other than the published one holds the prediction of a model.
MODEL_COLUMNS = r"^(?!published_mwh$).+_mwh$"
METRICS = ["correlation", "rmse_cf", "bias", "bias_relative_to_year"]
# Models and baselines learn from these years only; 2026 is a prospective test (evaluation plan).
TRAINING_YEARS = range(2019, 2026)
# The 2019-2022 folds are the years the model forms were chosen on, so their errors are optimistic;
# the minimal detectable effect uses the 2023-2025 folds (evaluation plan, Deviations).
PERIODS = {"2019-2022": range(2019, 2023), "2023-2025": range(2023, 2026)}


class Predictions(pa.DataFrameModel):
    """Hours of held-out years and the energy in MWh that each model gives for them.

    A model column is the model name followed by `_mwh`; there is at least one. `held_out_year` is
    the UTC year of the hour, the fold that predicted it. `capacity_mw` is the AC capacity at the
    start of the hour, `season` that of the UTC month and `sky_class` that of the hour's day. Only
    eligible hours are evaluated: they have their published energy, every model energy and a sky
    class.
    """

    time: Index[grid.UtcTime] = pa.Field(unique=True, check_name=True)
    held_out_year: Series[int]
    published_mwh: Series[float] = pa.Field(nullable=True)
    capacity_mw: Series[float] = pa.Field(gt=0)
    model_mwh: Series[float] = pa.Field(alias=MODEL_COLUMNS, regex=True, nullable=True)
    eligible: Series[bool]
    season: Series[str] = pa.Field(isin=list(SEASONS))
    sky_class: Series[str] = pa.Field(isin=SKY_CLASSES, nullable=True)

    @pa.check("time")
    @classmethod
    def on_the_hour(cls, time: "pd.Series[pd.Timestamp]") -> "pd.Series[bool]":
        return time == time.dt.floor("h")

    @pa.dataframe_check
    @classmethod
    def held_out_year_is_the_utc_year(cls, frame: pd.DataFrame) -> "pd.Series[bool]":
        return frame["held_out_year"] == pd.DatetimeIndex(frame.index).year

    @pa.dataframe_check
    @classmethod
    def season_of_the_utc_month(cls, frame: pd.DataFrame) -> "pd.Series[bool]":
        return frame["season"] == pd.DatetimeIndex(frame.index).month.map(SEASON_OF_MONTH)

    @pa.dataframe_check
    @classmethod
    def eligible_hours_have_every_value(cls, frame: pd.DataFrame) -> "pd.Series[bool]":
        values = frame.filter(regex="_mwh$").assign(sky_class=frame["sky_class"])
        return ~frame["eligible"] | values.notna().all(axis="columns")

    class Config:
        strict = True


def climatology(hours: pd.DataFrame, held_out_year: int) -> pd.Series:
    """Monthly-hourly climatology of the hours of `held_out_year`, in MWh: the mean capacity factor
    of the eligible hours of the other years of `TRAINING_YEARS` in `hours` with the same calendar
    month and UTC hour, times the capacity of the hour. NaN where no such hour has that month and
    hour.

    `hours` has a UTC time index and the columns `published_mwh`, `capacity_mw` and `eligible` of
    `Predictions`.
    """
    time = hours.index
    if not isinstance(time, pd.DatetimeIndex) or str(time.tz) != "UTC":
        raise ValueError("expected a DatetimeIndex in UTC")
    training = (
        time.year.isin(TRAINING_YEARS) & (time.year != held_out_year) & hours["eligible"].to_numpy()
    )
    held_out = time.year == held_out_year
    factor = hours["published_mwh"] / hours["capacity_mw"]
    mean_factor = factor[training].groupby([time.month[training], time.hour[training]]).mean()
    cells = pd.MultiIndex.from_arrays([time.month[held_out], time.hour[held_out]])
    energy = mean_factor.reindex(cells).to_numpy() * hours["capacity_mw"].to_numpy()[held_out]
    return pd.Series(energy, index=time[held_out], name="climatology_mwh")


class Metrics(pa.DataFrameModel):
    """Value of a metric for one model over the eligible hours of a held-out year, all of them
    (group "all") or those of one group; `hours` counts them."""

    technology: Series[str]
    held_out_year: Series[int]
    model: Series[str]
    metric: Series[str] = pa.Field(isin=METRICS)
    group: Series[str] = pa.Field(isin=["all", *SEASONS, *SKY_CLASSES])
    # NaN where undefined, such as a correlation with a constant series.
    value: Series[float] = pa.Field(nullable=True)
    hours: Series[int] = pa.Field(gt=0)

    class Config:
        strict = True
        unique: ClassVar[list[str]] = ["technology", "held_out_year", "model", "metric", "group"]


def metrics(predictions: pd.DataFrame, technology: str) -> DataFrame[Metrics]:
    """Metrics of each model over the eligible hours of each held-out year of `predictions`, a
    `Predictions` frame. A capacity factor is an energy over the AC capacity of its hour.

    - correlation: Pearson correlation of the hourly capacity factors of the model and the
      published series;
    - rmse_cf: root mean square of the hourly capacity factor error;
    - bias: energy of the model over the published energy, minus 1, over the year and over each
      season and sky class that has eligible hours;
    - bias_relative_to_year: bias of a season or sky class once the model is scaled to the
      published energy of the year, (1 + bias of the group) / (1 + bias of the year) - 1.
    """
    Predictions.validate(predictions)
    eligible = predictions[predictions["eligible"]]
    if eligible.empty:
        raise ValueError("no eligible hours to evaluate")
    rows: list[tuple[Hashable, str, str, Hashable, float, int]] = []
    for year, fold in eligible.groupby("held_out_year"):
        hours = len(fold)
        published_cf = fold["published_mwh"] / fold["capacity_mw"]
        for column in fold.filter(regex=MODEL_COLUMNS).columns:
            model = column.removesuffix("_mwh")
            modelled_cf = fold[column] / fold["capacity_mw"]
            error_cf = modelled_cf - published_cf
            ratio = fold[column].sum() / fold["published_mwh"].sum()
            rows += [
                (year, model, "correlation", "all", modelled_cf.corr(published_cf), hours),
                (year, model, "rmse_cf", "all", np.sqrt((error_cf**2).mean()), hours),
                (year, model, "bias", "all", ratio - 1, hours),
            ]
            for grouping in ("season", "sky_class"):
                groups = fold.groupby(grouping)
                group_ratios = groups[column].sum() / groups["published_mwh"].sum()
                for group, group_ratio, group_hours in zip(
                    group_ratios.index, group_ratios, groups.size(), strict=True
                ):
                    relative_bias = group_ratio / ratio - 1
                    rows += [
                        (year, model, "bias", group, group_ratio - 1, group_hours),
                        (year, model, "bias_relative_to_year", group, relative_bias, group_hours),
                    ]
    table = pd.DataFrame(
        rows, columns=["held_out_year", "model", "metric", "group", "value", "hours"]
    )
    table.insert(0, "technology", technology)
    return Metrics.validate(table)


def mean_absolute_by_period(table: pd.DataFrame) -> pd.DataFrame:
    """Mean absolute value of each metric of a `Metrics` table over the folds of each period of
    `PERIODS`, with the number of folds and their eligible hours. Undefined values and folds of
    other years, such as a prospective test year, are left out."""
    Metrics.validate(table)
    defined = table.dropna(subset=["value"])
    period_of_year = {year: period for period, years in PERIODS.items() for year in years}
    by_period = defined.assign(
        period=defined["held_out_year"].map(period_of_year),
        absolute_value=defined["value"].abs(),
    )
    keys = ["technology", "period", "model", "metric", "group"]
    summary = by_period.groupby(keys, sort=False).agg(
        folds=("value", "size"),
        mean_absolute_value=("absolute_value", "mean"),
        hours=("hours", "sum"),
    )
    return summary.reset_index()
