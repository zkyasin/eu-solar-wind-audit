"""Feed-in signature of axis 3 (evaluation plan, axis 3, evidence 4, and Deviations).

A producer that consumes part of its solar output on site feeds in what its consumption leaves
over. At the same hour and date its consumption barely depends on the sky, so it feeds in a larger
share of its output on a clear day than on an overcast one: a series of feed-in only rises more
than the solar potential from overcast to clear days, a series of the whole output does not.

The contrast of a series is the clear minus overcast difference of log(published / potential)
within cells of the same local hour, month, year and day type. The potential is the physics model
with nominal parameters, the raw model of the calibration predictions, never fitted to a series.
The hours are the eligible hours of the calibration, whose price is a reference price, with a sky
class (days.sky_classes) and positive published and modelled output. The contrast is the clear
minus overcast coefficient of a fit of the log ratio on the indicators of both classes with one
effect per cell (linearmodels PanelOLS, the within estimator), with standard errors clustered by
day and scaled by n / (n - d - k), d cell effects and k = 2 indicators for n hours.

A series shows the signature when the 90 % interval of its contrast lies above zero. The evidence
is used only when the negative control, éCO2mix, does not show it and the positive control, the
SMARD feed-in, does, on all their eligible hours.
"""

from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd
from linearmodels.panel import PanelOLS
from linearmodels.panel.results import PanelEffectsResults
from scipy.stats import norm

from eu_solar_wind_audit import days, evaluation

LEVEL = 0.90
CLASSES = ("clear", "overcast")
NEGATIVE_CONTROL = "eco2mix"
POSITIVE_CONTROL = "smard"
# Sensitivity without the hours below this modelled capacity factor, most of them at dawn and dusk.
MIN_CAPACITY_FACTOR = 0.05


@dataclass(frozen=True)
class Estimate:
    """Statistic of a signature, its standard error and 90 % interval, and the hours and days it
    rests on."""

    estimate: float
    standard_error: float
    low: float
    high: float
    hours: int
    days: int


def hours_table(published: pd.Series, predictions: pd.DataFrame, country: str) -> pd.DataFrame:
    """The hours of a contrast: one row per eligible hour of `predictions`, an
    evaluation.Predictions frame of `country`, with a sky class and positive output in `published`
    (MWh on the same hours, NaN when missing) and in the raw model. Columns: the cell, the season
    of the local month, the local day, the sky class, the modelled capacity factor and the log
    ratio."""
    hours = _predicted_hours(published, predictions)
    frame = _calendar(hours, country).assign(
        published_mwh=published.to_numpy(),
        raw_mwh=predictions["raw_mwh"].to_numpy(),
        sky_class=predictions["sky_class"].to_numpy(),
        capacity_factor=(predictions["raw_mwh"] / predictions["capacity_mw"]).to_numpy(),
    )
    kept = (
        predictions["eligible"]
        & frame["sky_class"].notna()
        & (frame["published_mwh"] > 0)
        & (frame["raw_mwh"] > 0)
    )
    table = frame[kept]
    log_ratio = np.log(table["published_mwh"] / table["raw_mwh"])
    return table.drop(columns=["published_mwh", "raw_mwh"]).assign(log_ratio=log_ratio)


def contrast(table: pd.DataFrame) -> Estimate:
    """Clear minus overcast contrast of the hours of `table`, a `hours_table`."""
    # The cell is the entity and the local day the period, which each cell holds at most once.
    panel = table.set_index(["cell", "day"])
    indicators = pd.DataFrame({name: panel["sky_class"].eq(name).astype(float) for name in CLASSES})
    fit = PanelOLS(panel["log_ratio"], indicators, entity_effects=True).fit(
        cov_type="clustered", cluster_time=True
    )
    return _estimate(fit, np.array([1.0, -1.0]), table)


def contrasts(table: pd.DataFrame) -> pd.DataFrame:
    """The contrast of the hours of `table`, a `hours_table`, over all of them and in each season,
    first on all hours and then on the hours from MIN_CAPACITY_FACTOR: one row each, with its
    sign, positive or negative when the interval excludes zero and none otherwise."""
    rows = [
        {"min_capacity_factor": min_capacity_factor, **row}
        for min_capacity_factor in (0.0, MIN_CAPACITY_FACTOR)
        for row in _by_season(table[table["capacity_factor"] >= min_capacity_factor], contrast)
    ]
    return _with_signs(rows)


def controls(rows: pd.DataFrame) -> dict[str, bool]:
    """From the `contrasts` of each series in `rows`, with a `series` column: whether the negative
    control does not show the signature and the positive control does, on all their hours, and
    whether both hold, the condition for using the evidence."""
    overall = rows[(rows["season"] == "all") & (rows["min_capacity_factor"] == 0)]
    return _controls(overall, NEGATIVE_CONTROL, POSITIVE_CONTROL, "positive")


def _predicted_hours(series: pd.Series, predictions: pd.DataFrame) -> pd.DatetimeIndex:
    """Hours of `predictions`, an evaluation.Predictions frame, which `series` must be on."""
    evaluation.Predictions.validate(predictions)
    if not series.index.equals(predictions.index):
        raise ValueError("the published series is not on the hours of the predictions")
    return pd.DatetimeIndex(predictions.index)


def _calendar(hours: pd.DatetimeIndex, country: str) -> pd.DataFrame:
    """Cell of each hour in `country`, from its local hour, month, year and day type, with the
    season of the local month and the local day."""
    local = hours.tz_convert(days.TIMEZONE)
    day_types = days.day_types(hours, country).to_numpy()
    cells = [
        f"{hour}-{month}-{year}-{day_type}"
        for hour, month, year, day_type in zip(
            local.hour, local.month, local.year, day_types, strict=True
        )
    ]
    return pd.DataFrame(
        {
            "cell": cells,
            "season": local.month.map(evaluation.SEASON_OF_MONTH),
            "day": days.local_days(hours),
        },
        index=hours,
    )


def _estimate(
    fit: PanelEffectsResults, weights: npt.NDArray[np.float64], table: pd.DataFrame
) -> Estimate:
    """The combination `weights` of the parameters of `fit`, from the hours of `table`, with its
    90 % normal interval."""
    estimate = float(fit.params.to_numpy() @ weights)
    error = float(np.sqrt(weights @ fit.cov.to_numpy() @ weights))
    z = float(norm.ppf(0.5 + LEVEL / 2))
    return Estimate(
        estimate,
        error,
        estimate - z * error,
        estimate + z * error,
        len(table),
        table["day"].nunique(),
    )


def _by_season(
    table: pd.DataFrame, statistic: Callable[[pd.DataFrame], Estimate]
) -> list[dict[str, object]]:
    """`statistic` of all the hours of `table` and of the hours of each season it holds."""
    seasons = [season for season in evaluation.SEASONS if (table["season"] == season).any()]
    rows = []
    for season in ["all", *seasons]:
        part = table if season == "all" else table[table["season"] == season]
        rows.append({"season": season, **asdict(statistic(part))})
    return rows


def _with_signs(rows: list[dict[str, object]]) -> pd.DataFrame:
    """`rows` with the sign of their interval: positive or negative when it excludes zero, none
    otherwise."""
    frame = pd.DataFrame(rows)
    sign = np.select([frame["low"] > 0, frame["high"] < 0], ["positive", "negative"], "none")
    return frame.assign(sign=sign)


def _controls(
    overall: pd.DataFrame, negative_control: str, positive_control: str, side: str
) -> dict[str, bool]:
    """Whether the sign of the negative control in `overall`, one row per series, is not `side`,
    the sign of the signature, whether that of the positive control is, and whether both hold."""
    signs = overall.set_index("series")["sign"]
    negative = bool(signs[negative_control] != side)
    positive = bool(signs[positive_control] == side)
    return {
        "negative_control": negative,
        "positive_control": positive,
        "used": negative and positive,
    }
