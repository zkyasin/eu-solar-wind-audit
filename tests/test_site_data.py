import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from eu_solar_wind_audit import site_data


def hourly(start: str, end: str, **columns: float) -> pd.DataFrame:
    hours = pd.date_range(start, end, freq="h", inclusive="left", tz="UTC", name="time")
    return pd.DataFrame(columns, index=hours)


def test_days_are_local_and_a_day_with_a_missing_hour_has_no_published_output() -> None:
    # 31 March 2024 lasts 23 hours in Paris, from 23:00 UTC the day before.
    predictions = hourly(
        "2024-03-30 23:00", "2024-04-01 22:00", published_mwh=1.0, calibrated_level_mwh=2.0
    )
    predictions.loc["2024-04-01 10:00", "published_mwh"] = np.nan
    held_back = hourly("2024-03-31 11:00", "2024-03-31 13:00", modulation_mwh=500.0)

    days = site_data.daily(predictions, held_back, "Europe/Paris")

    assert days.index.tolist() == ["2024-03-31", "2024-04-01"]
    assert days["published_gwh"].isna().tolist() == [False, True]
    assert days.loc["2024-03-31", "published_gwh"] == pytest.approx(0.023)
    assert days["potential_gwh"].tolist() == pytest.approx([0.046, 0.048])
    assert days["negative_price_hours"].tolist() == [2, 0]
    assert days["held_back_gwh"].tolist() == pytest.approx([1.0, 0.0])


def test_local_days_cut_by_the_first_or_the_last_hour_are_left_out() -> None:
    # In Paris, the UTC hours of 1 and 2 January 2019 cover only 2 January whole.
    predictions = hourly(
        "2019-01-01 00:00", "2019-01-03 00:00", published_mwh=1.0, calibrated_level_mwh=1.0
    )
    held_back = hourly("2019-01-02 12:00", "2019-01-02 12:00", modulation_mwh=0.0)

    days = site_data.daily(predictions, held_back, "Europe/Paris")

    assert days.index.tolist() == ["2019-01-02"]
    assert days["published_gwh"].tolist() == pytest.approx([0.024])


def test_days_without_predictions_stay_in_the_calendar_without_values() -> None:
    predictions = pd.concat(
        [
            hourly("2024-06-01", "2024-06-02", published_mwh=1.0, calibrated_level_mwh=1.0),
            hourly("2024-06-03", "2024-06-04", published_mwh=1.0, calibrated_level_mwh=1.0),
        ]
    )
    held_back = hourly("2024-06-01", "2024-06-01", modulation_mwh=0.0)

    days = site_data.daily(predictions, held_back, "UTC")

    assert days.index.tolist() == ["2024-06-01", "2024-06-02", "2024-06-03"]
    assert days.loc["2024-06-02", ["published_gwh", "potential_gwh"]].isna().all()
    assert days.loc["2024-06-02", "negative_price_hours"] == 0


def test_quality_keeps_the_errors_over_all_hours() -> None:
    summary = pd.DataFrame(
        {
            "period": ["2023-2025"] * 3,
            "model": ["calibrated_level"] * 3,
            "metric": ["correlation", "rmse_cf", "rmse_cf"],
            "group": ["all", "all", "DJF"],
            "mean_value": [0.98, 0.025, 0.04],
            "hours": [11439, 11439, 2000],
        }
    )

    (row,) = site_data.records(site_data.quality(summary), 4)

    assert row == {
        "period": "2023-2025",
        "model": "calibrated_level",
        "correlation": 0.98,
        "rmse_cf": 0.025,
        "hours": 11439,
    }


def test_each_period_of_the_modulation_takes_its_reference_where_there_is_one() -> None:
    intervals = pd.DataFrame(
        {"low_gwh": [600.0, 10.0, 1.0], "high_gwh": [800.0, 30.0, 3.0]},
        index=pd.Index(["2024", "2024-06", "2024-07"], name="period"),
    )
    references = pd.concat([pd.Series({2024: 640.0}), pd.Series({"2024-06": 25.0})]).rename(
        "rte_gwh"
    )

    periods = site_data.modulation_periods(intervals, references)

    assert periods["period"].tolist() == ["2024", "2024-06", "2024-07"]
    assert periods["reference_gwh"].tolist()[:2] == [640.0, 25.0]
    assert pd.isna(periods["reference_gwh"].iloc[2])


def test_the_perimeter_document_leaves_out_published_energy() -> None:
    levels = pd.DataFrame(
        [
            {column: 0.5 for column in site_data.LEVEL_COLUMNS}
            | {"series": "entsoe_nl", "year": 2024, "self_consumption_share": np.nan}
            | {"published_solar_gwh": 487.4, "published_wind_gwh": 22857.4}
        ]
    )
    audit = {
        "detection_threshold": {"self_consumption": 0.091, "distribution": 0.041},
        "distribution_share": {"values": {"NL": 0.496}},
    }

    document = site_data.perimeter(pd.DataFrame({"series": ["entsoe_nl"]}), levels, audit)

    (year,) = document["years"]
    assert set(year) == set(site_data.LEVEL_COLUMNS)
    assert year["self_consumption_share"] is None
    assert document["distribution_share"] == {"NL": 0.496}


def test_documents_are_written_as_strict_json(tmp_path: Path) -> None:
    days = pd.DataFrame(
        {"published_gwh": [1.23456, np.nan]},
        index=pd.Index(["2024-06-01", "2024-06-02"], name="day"),
    )

    path = site_data.write(tmp_path, "daily.json", site_data.columns(days, 3))

    assert json.loads(path.read_text()) == {
        "day": ["2024-06-01", "2024-06-02"],
        "published_gwh": [1.235, None],
    }
    with pytest.raises(ValueError, match="JSON"):
        site_data.write(tmp_path, "nan.json", {"value": float("nan")})
