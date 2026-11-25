"""Data of the project's web page: small JSON documents built from the results of calibrate,
estimate-modulation, detect-anomalies and audit-perimeter, which the page reads.

Published values come from éCO2mix (Licence Ouverte 2.0) and SMARD (CC BY 4.0); the series of the
Transparency Platform appear only as yearly ratios (docs/data-licences.md).
"""

import json
from collections.abc import Hashable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd

# The potential that the page sets against published output: the calibrated potential with its
# strata and a level fitted on the other years (calibrate).
POTENTIAL = "calibrated_level_mwh"
HOUR = pd.Timedelta(hours=1)
# Decimals kept in the documents: GWh to the MWh, ratios and errors to 1e-4.
GWH_DECIMALS = 3
RATIO_DECIMALS = 4
# A period of four characters is a year, "2025"; months are "2025-06".
YEAR_DIGITS = 4
# Columns of the yearly levels of audit-perimeter on the page: ratios and decisions, without the
# published GWh, which for the Transparency Platform may not be republished.
LEVEL_COLUMNS = [
    "series",
    "year",
    "solar_ratio",
    "wind_ratio",
    "gap",
    "self_consumption_share",
    "distribution_share",
    "self_consumption",
    "distribution",
]


def daily(predictions: pd.DataFrame, held_back: pd.DataFrame, time_zone: str) -> pd.DataFrame:
    """Published output and potential in GWh of each whole local day of `predictions`, the hourly
    predictions of calibrate, with the number of hours of `held_back`, the negative-price hours of
    estimate-modulation, and the output held back at them. A value is missing on a day with one of
    its hours missing."""
    hours = pd.DatetimeIndex(predictions.index).tz_convert(time_zone)
    midnights = pd.date_range(hours[0].ceil("D"), (hours[-1] + HOUR).floor("D"), freq="D")
    calendar = midnights[:-1].strftime("%Y-%m-%d")
    # 23, 24 or 25 hours, with the clock changes.
    hours_in_day = pd.Series((midnights[1:] - midnights[:-1]) / HOUR, index=calendar)
    grouped = predictions[["published_mwh", POTENTIAL]].groupby(hours.strftime("%Y-%m-%d"))
    complete = grouped.count().reindex(calendar).eq(hours_in_day, axis="index")
    energy = (grouped.sum().reindex(calendar) / 1000).where(complete)
    held_days = pd.DatetimeIndex(held_back.index).tz_convert(time_zone).strftime("%Y-%m-%d")
    by_day = held_back.groupby(held_days)["modulation_mwh"]
    return pd.DataFrame(
        {
            "published_gwh": energy["published_mwh"],
            "potential_gwh": energy[POTENTIAL],
            "negative_price_hours": by_day.count().reindex(calendar, fill_value=0),
            "held_back_gwh": (by_day.sum() / 1000).reindex(calendar, fill_value=0.0),
        }
    ).rename_axis("day")


def quality(summary: pd.DataFrame) -> pd.DataFrame:
    """Correlation and RMSE in capacity factor of each model of calibrate's summary, over all the
    held-out hours of each period, with their number of hours."""
    rows = summary[summary["group"] == "all"]
    table = rows.pivot_table(
        index=["period", "model"], columns="metric", values="mean_value", aggfunc="first"
    )
    hours = rows.groupby(["period", "model"])["hours"].first()
    return table[["correlation", "rmse_cf"]].join(hours).reset_index()


def findings(found: pd.DataFrame) -> pd.DataFrame:
    """The findings of detect-anomalies with their evidence as objects and ISO times."""
    return found.assign(
        start=found["start"].map(pd.Timestamp.isoformat),
        end=found["end"].map(pd.Timestamp.isoformat),
        evidence=found["evidence"].map(json.loads),
    )


def modulation_periods(intervals: pd.DataFrame, references: "pd.Series[float]") -> pd.DataFrame:
    """Estimate and 90 % interval of each year and month of estimate-modulation, indexed by
    period, with the published reference of the period where there is one."""
    reference = references.rename(index=str).rename("reference_gwh")
    return intervals.rename(index=str).join(reference).rename_axis("period").reset_index()


def detectable_years(periods: pd.DataFrame, mde_gwh: Mapping[str, float | None]) -> pd.DataFrame:
    """The years of `periods`, as modulation_periods gives them, whose estimate reaches the
    minimal detectable effect of the year in `mde_gwh`: the only estimates the evaluation plan
    publishes without a reference."""
    years = periods[periods["period"].str.len() == YEAR_DIGITS]
    mde = years["period"].map(mde_gwh).astype(float)
    return years[years["estimate_gwh"] >= mde].reset_index(drop=True)


def perimeter(cells: pd.DataFrame, levels: pd.DataFrame, audit: dict[str, Any]) -> dict[str, Any]:
    """The matrix of audit-perimeter, the ratios and decisions of each series and year, and the
    detection thresholds and distribution shares they rest on."""
    return {
        "detection_threshold": audit["detection_threshold"],
        "distribution_share": audit["distribution_share"]["values"],
        "cells": records(cells, RATIO_DECIMALS),
        "years": records(levels[LEVEL_COLUMNS], RATIO_DECIMALS),
    }


def records(frame: pd.DataFrame, decimals: int) -> list[dict[Hashable, Any]]:
    """The rows of `frame` as JSON objects, floats rounded to `decimals` and missing values null."""
    rounded = frame.round(decimals)
    return rounded.astype(object).where(rounded.notna(), None).to_dict(orient="records")


def columns(frame: pd.DataFrame, decimals: int) -> dict[Hashable, list[Any]]:
    """The index and columns of `frame` as JSON arrays, floats rounded to `decimals` and missing
    values null."""
    rounded = frame.round(decimals).reset_index()
    return rounded.astype(object).where(rounded.notna(), None).to_dict(orient="list")


def write(output: Path, name: str, document: object) -> Path:
    """Write `document` as compact JSON to `output/name`."""
    output.mkdir(parents=True, exist_ok=True)
    path = output / name
    text = json.dumps(document, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    path.write_text(text + "\n", encoding="utf-8")
    return path
