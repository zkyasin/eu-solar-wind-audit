import numpy as np
import pandas as pd
import pytest

from eu_solar_wind_audit import detectability, evaluation, modulation


def utc_stamps(stamps: list[str]) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(stamps, tz="UTC", name="time").as_unit("ns")


def predictions(
    hours: pd.DatetimeIndex,
    published: list[float],
    potential: list[float],
    sky_class: list[str],
    eligible: list[bool] | None = None,
) -> pd.DataFrame:
    return evaluation.Predictions.validate(
        pd.DataFrame(
            {
                "held_out_year": hours.year.astype("int64"),
                "published_mwh": published,
                "capacity_mw": 100.0,
                "calibrated_level_mwh": potential,
                "eligible": eligible or [True] * len(hours),
                "season": hours.month.map(evaluation.SEASON_OF_MONTH),
                "sky_class": sky_class,
            },
            index=hours,
        )
    )


def covariates(hours: pd.DatetimeIndex, elevation: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"sun_elevation_deg": elevation}, index=hours)


def test_daily_errors_use_clear_high_sun_hours_of_the_recent_folds() -> None:
    hours = utc_stamps(
        [
            # 2024-06-03, clear: two hours with the sun high, one low.
            "2024-06-03 10:00",
            "2024-06-03 11:00",
            "2024-06-03 18:00",
            # 2024-06-04, mixed.
            "2024-06-04 11:00",
            # 2022-06-03, clear but in an early fold.
            "2022-06-03 11:00",
            # 2025-06-03, clear, not eligible.
            "2025-06-03 11:00",
        ]
    )
    found = detectability.daily_errors(
        predictions(
            hours,
            published=[110.0, 130.0, 50.0, 10.0, 10.0, 10.0],
            potential=[100.0, 100.0, 10.0, 100.0, 100.0, 100.0],
            sky_class=["clear", "clear", "clear", "mixed", "clear", "clear"],
            eligible=[True, True, True, True, True, False],
        ),
        covariates(hours, [50.0, 55.0, 10.0, 55.0, 55.0, 55.0]),
        "solar",
        "FR",
    )

    assert found.index.tolist() == [pd.Timestamp("2024-06-03")]
    assert found.iloc[0] == pytest.approx(40 / 200)


def test_the_effect_grows_with_the_potential_of_each_day_of_n_hours() -> None:
    errors = pd.Series([0.1, -0.1, 0.3, -0.3])
    hours = modulation.NegativeHours.validate(
        pd.DataFrame(
            {
                "published_mwh": 0.0,
                "potential_mwh": [1000.0, 2000.0, 4000.0, 3000.0],
                "modulation_mwh": [1000.0, 2000.0, 4000.0, 3000.0],
                "thin_stratum": False,
            },
            # Two hours of one Paris day of 2024, one of another, one of 2025.
            index=utc_stamps(
                ["2024-05-01 11:00", "2024-05-01 12:00", "2024-05-02 12:00", "2025-05-01 12:00"]
            ),
        )
    )

    found = detectability.minimal_detectable_effects(errors, hours)

    spread = errors.std() * detectability.Z_SUM / 1000
    assert found.index.tolist() == ["2024", "2025"]
    assert found.tolist() == pytest.approx(
        [spread * np.sqrt(3000.0**2 + 4000.0**2), spread * 3000.0]
    )


def test_the_effect_detects_at_five_percent_with_eighty_percent_power() -> None:
    z_sum = detectability.Z_SUM

    assert z_sum == pytest.approx(1.6449 + 0.8416, abs=1e-4)


def test_german_onshore_wind_keeps_its_fold_of_2023() -> None:
    hours = utc_stamps(["2023-06-03 11:00", "2024-06-03 11:00"])
    found = {
        country: detectability.daily_errors(
            predictions(
                hours,
                published=[110.0, 90.0],
                potential=[100.0, 100.0],
                sky_class=["clear", "clear"],
                eligible=[True, True],
            ),
            covariates(hours, [55.0, 55.0]),
            "wind_onshore",
            country,
        )
        for country in ("FR", "DE")
    }

    assert found["FR"].index.tolist() == [pd.Timestamp("2024-06-03")]
    assert found["DE"].index.tolist() == [pd.Timestamp("2023-06-03"), pd.Timestamp("2024-06-03")]
