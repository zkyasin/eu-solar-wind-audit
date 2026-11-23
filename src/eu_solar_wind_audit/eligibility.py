"""Hours on which the calibrated potential of France is fitted and evaluated (evaluation plan,
"Scope", "Two model objects", "Axis 1" and Deviations).

Each rule of `Eligibility` has its own column, so that a report can count the hours that each one
leaves out.
"""

import pandas as pd
import pandera.pandas as pa
from pandera.typing import DataFrame, Index, Series

from eu_solar_wind_audit import evaluation, grid, physics, precompute
from eu_solar_wind_audit.sources import odre

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
