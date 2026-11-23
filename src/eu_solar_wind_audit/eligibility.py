"""Hours on which the calibrated potential of France is fitted and evaluated (evaluation plan,
"Scope", "Two model objects", "Axis 1" and Deviations), and the inputs they are chosen from, one
UTC year at a time.

Each rule of `Eligibility` has its own column, so that a report can count the hours that each one
leaves out.
"""

from dataclasses import dataclass
from pathlib import Path

import httpx
import pandas as pd
import pandera.pandas as pa
from pandera.typing import DataFrame, Index, Series

from eu_solar_wind_audit import evaluation, grid, physics, precompute, regimes
from eu_solar_wind_audit.config import Country
from eu_solar_wind_audit.sources import natural_earth, odre, smard

FINAL_STATUSES = ["definitive", "consolidated"]


class Eligibility(pa.DataFrameModel):
    """Whether the hour starting at `time` meets each rule, and all of them (`eligible`):

    - `consolidated_or_definitive`: its published status is definitive or consolidated;
    - `complete`: its published and modelled output are complete;
    - `reference_price`: its price class is reference (P);
    - `final_capacity`: its capacity uses final SDES months only;
    - `final_era5`: its weather is final ERA5, not ERA5T;
    - `training_year`: its UTC year is one of evaluation.TRAINING_YEARS;
    - solar only, `sun_above_horizon`: the capacity-weighted apparent sun elevation at mid-hour
      is above 0 degrees;
    - onshore wind only, `onshore_reference`: the hour is before odre.FIRST_OFFSHORE_POWER or
      from odre.WIND_SPLIT_START, where odre.onshore_hourly gives onshore wind.
    """

    time: Index[grid.UtcTime] = pa.Field(unique=True, check_name=True)
    consolidated_or_definitive: Series[bool]
    complete: Series[bool]
    reference_price: Series[bool]
    final_capacity: Series[bool]
    final_era5: Series[bool]
    training_year: Series[bool]
    sun_above_horizon: Series[bool] | None
    onshore_reference: Series[bool] | None
    eligible: Series[bool]

    @pa.dataframe_check
    @classmethod
    def eligible_meets_every_rule(cls, frame: pd.DataFrame) -> "pd.Series[bool]":
        return frame["eligible"] == frame.drop(columns="eligible").all(axis="columns")

    class Config:
        strict = True


def eligible_hours(
    technology: physics.Technology,
    published: pd.DataFrame,
    covariates: pd.DataFrame,
    price_classes: pd.DataFrame,
) -> DataFrame[Eligibility]:
    """Rules of `Eligibility` for each hour of `published`, the odre.Hourly series that
    `technology` is calibrated on: national solar, or the onshore wind of odre.onshore_hourly.

    `covariates` are the precompute.Covariates of the technology and `price_classes` the classes of
    regimes.classify_hours, both on the hours of `published`.
    """
    odre.Hourly.validate(published)
    precompute.Covariates.validate(covariates)
    hours = pd.DatetimeIndex(published.index)
    for name, frame in [("covariates", covariates), ("price classes", price_classes)]:
        if not frame.index.equals(hours):
            raise ValueError(f"the {name} are not on the hours of the published series")

    rules = pd.DataFrame(
        {
            "consolidated_or_definitive": published["status"].isin(FINAL_STATUSES),
            "complete": published["complete"] & covariates["complete"],
            "reference_price": price_classes["regime"] == "reference",
            "final_capacity": ~covariates["capacity_provisional"],
            "final_era5": covariates["era5_source"] == "ERA5",
            "training_year": hours.year.isin(evaluation.TRAINING_YEARS),
        },
        index=hours,
    )
    if technology == "solar":
        rules["sun_above_horizon"] = covariates["sun_elevation_deg"] > 0
    else:
        # Published wind then includes offshore output, which no series gives separately.
        with_offshore = (hours >= odre.FIRST_OFFSHORE_POWER) & (hours < odre.WIND_SPLIT_START)
        rules["onshore_reference"] = ~with_offshore
    rules["eligible"] = rules.all(axis="columns")
    return Eligibility.validate(rules)


def price_classes(
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    zone: str = "FR",
    client: httpx.Client | None = None,
    cache_dir: Path = smard.CACHE_DIR,
) -> pd.DataFrame:
    """regimes.classify_hours of the day-ahead prices of the bidding `zone`, a key of
    smard.PRICES, for each hour of [start, end), both hour starts in UTC. Prices are read from
    SMARD every quarter-hour (for France, filter 254, which repeats the hourly price before October
    2025); a quarter-hour without a price makes its hour missing.

    The hours just before and after the period are classified with it, then left out: a negative
    or missing neighbour keeps an hour out of the reference class.
    """
    grid.check_period(start, end)
    first, stop = start - grid.HOUR, end + grid.HOUR
    filter_id, region = smard.PRICES[zone]
    prices = smard.fetch_series(
        filter_id, region, "quarterhour", first, stop, client=client, cache_dir=cache_dir
    )
    quarter_hours = pd.date_range(first, stop, freq="15min", inclusive="left", unit="ns")
    classes = regimes.classify_hours(prices.reindex(quarter_hours))
    return classes.loc[start : end - grid.HOUR].rename_axis("time")


@dataclass(frozen=True)
class YearInputs:
    """Calibration inputs of one technology on the hours of one UTC year, all on the same hours:
    the published series, its price classes, the precompute arrays with their covariates, and the
    rules of `Eligibility`."""

    published: DataFrame[odre.Hourly]
    price_classes: pd.DataFrame
    arrays: precompute.SolarArrays | precompute.WindArrays
    eligibility: DataFrame[Eligibility]


def negative_price_hours(inputs: YearInputs) -> pd.Series:
    """Whether each hour of `inputs` has the negative price class and meets every other rule of
    `Eligibility`: the hours of axis 4, which never enter a fit."""
    other_rules = inputs.eligibility.drop(columns=["reference_price", "eligible"])
    return other_rules.all(axis="columns") & (inputs.price_classes["regime"] == "negative")


def year_inputs(
    client: httpx.Client,
    technology: physics.Technology,
    year: int,
    parameters: Country,
    *,
    odre_cache_dir: Path = odre.CACHE_DIR,
    smard_cache_dir: Path = smard.CACHE_DIR,
    era5_cache_dir: Path = physics.ERA5_CACHE_DIR,
) -> YearInputs:
    """Calibration inputs of `technology` on the hours of the UTC year `year`: national solar or
    the onshore wind of odre.onshore_hourly, the `price_classes` of these hours, and the arrays of
    precompute.solar or precompute.wind_onshore, from the SDES capacity and the department
    polygons loaded as physics.model_france loads them for that year.
    """
    start = pd.Timestamp(year, 1, 1, tz="UTC")
    end = pd.Timestamp(year + 1, 1, 1, tz="UTC")
    classes = price_classes(start, end, client=client, cache_dir=smard_cache_dir)
    monthly = physics.load_monthly_capacity(technology, end - grid.HOUR)
    regions = natural_earth.load_admin1(parameters.country)
    arrays: precompute.SolarArrays | precompute.WindArrays
    if technology == "solar":
        published = odre.national_hourly(client, "solar", start, end, cache_dir=odre_cache_dir)
        arrays = precompute.solar(start, end, parameters, monthly, regions, era5_cache_dir)
    else:
        published = odre.onshore_hourly(client, start, end, cache_dir=odre_cache_dir)
        arrays = precompute.wind_onshore(start, end, parameters, monthly, regions, era5_cache_dir)
    rules = eligible_hours(technology, published, arrays.covariates, classes)
    return YearInputs(published, classes, arrays, rules)
