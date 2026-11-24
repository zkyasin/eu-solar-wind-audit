import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from eu_solar_wind_audit import cli, site_data


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


def write_results(root: Path) -> None:
    """Results of calibrate, estimate-modulation and detect-anomalies for France and Germany, and
    of audit-perimeter, reduced to what the page reads."""
    for country in ("FR", "DE"):
        calibration = root / f"calibration-{country.lower()}"
        modulation = root / f"modulation-{country.lower()}"
        anomalies = root / f"anomalies-{country.lower()}"
        for directory in (calibration, modulation, anomalies):
            directory.mkdir(parents=True)
        for technology in ("solar", "wind_onshore"):
            # The whole local days of 1 and 2 June 2024 in Paris and Berlin.
            hourly(
                "2024-05-31 22:00", "2024-06-02 22:00", published_mwh=1.0, calibrated_level_mwh=2.0
            ).to_parquet(calibration / f"{technology}-predictions.parquet")
            pd.DataFrame(
                {
                    "period": ["2023-2025"] * 2,
                    "model": ["calibrated_level"] * 2,
                    "metric": ["correlation", "rmse_cf"],
                    "group": ["all"] * 2,
                    "mean_value": [0.98, 0.03],
                    "hours": [100, 100],
                }
            ).to_parquet(calibration / f"{technology}-summary.parquet")
            (calibration / f"{technology}-calibration.json").write_text('{"git_describe": "c1"}')
            hourly("2024-06-01 12:00", "2024-06-01 14:00", modulation_mwh=100.0).to_parquet(
                modulation / f"{technology}-modulation-hours.parquet"
            )
            pd.DataFrame(
                {"low_gwh": [1.0], "high_gwh": [3.0], "estimate_gwh": [2.0]},
                index=pd.Index(["2024"], name="period"),
            ).to_parquet(modulation / f"{technology}-modulation-intervals.parquet")
            (modulation / f"{technology}-modulation.json").write_text(
                '{"git_describe": "m1", "mde_gwh": {"2024": 0.5}}'
            )
            if country == "FR":
                for period, label in (("years", 2024), ("months", "2024-06")):
                    pd.DataFrame(
                        {"rte_gwh": [2.5]}, index=pd.Index([label], name="month")
                    ).to_parquet(modulation / f"{technology}-rte-{period}.parquet")
            else:
                pd.DataFrame(
                    {"reference_gwh": [2.2]}, index=pd.Index([2024], name="year")
                ).to_parquet(modulation / f"{technology}-hirth-years.parquet")
        pd.DataFrame(
            {
                "series": [f"{country}.solar"],
                "start": [pd.Timestamp("2024-06-01 10:00", tz="UTC")],
                "end": [pd.Timestamp("2024-06-01 11:00", tz="UTC")],
                "kind": ["gap"],
                "score": [1.0],
                "evidence": ['{"hours": 1}'],
                "provisional": [False],
            }
        ).to_parquet(anomalies / f"{country}-findings.parquet")
        (anomalies / f"{country}-detection.json").write_text('{"git_describe": "d1"}')
    audit = root / "perimeter"
    audit.mkdir()
    pd.DataFrame({"series": ["smard"], "cell": ["feed_in_only"]}).to_parquet(
        audit / "perimeter-cells.parquet"
    )
    levels = dict.fromkeys(site_data.LEVEL_COLUMNS, 0.5) | {"series": "smard", "year": 2024}
    pd.DataFrame([levels]).to_parquet(audit / "perimeter-levels.parquet")
    (audit / "perimeter.json").write_text(
        json.dumps(
            {
                "git_describe": "p1",
                "detection_threshold": {"self_consumption": 0.09, "distribution": 0.04},
                "distribution_share": {"values": {"DE": 0.84}},
            }
        )
    )


def test_the_command_writes_the_documents_of_the_page(tmp_path: Path) -> None:
    write_results(tmp_path / "processed")

    cli.main(
        ["export-site-data", str(tmp_path / "site"), "--processed", str(tmp_path / "processed")]
    )

    site = tmp_path / "site"
    assert sorted(path.name for path in site.iterdir()) == sorted(
        [
            *(f"daily-{c}-{t}.json" for c in ("FR", "DE") for t in ("solar", "wind_onshore")),
            "quality.json",
            "modulation.json",
            "findings.json",
            "perimeter.json",
            "meta.json",
        ]
    )
    days = json.loads((site / "daily-DE-solar.json").read_text())
    assert days["day"] == ["2024-06-01", "2024-06-02"]
    assert days["negative_price_hours"] == [2, 0]
    french, *_ = json.loads((site / "modulation.json").read_text())
    assert french["reference"] == "RTE"
    assert french["periods"] == [
        {
            "period": "2024",
            "low_gwh": 1.0,
            "high_gwh": 3.0,
            "estimate_gwh": 2.0,
            "reference_gwh": 2.5,
        }
    ]
    assert len(json.loads((site / "findings.json").read_text())) == 2
    assert json.loads((site / "meta.json").read_text())["runs"]["audit-perimeter"] == "p1"
