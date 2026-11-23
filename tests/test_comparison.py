import math

import pandas as pd
import pytest

from eu_solar_wind_audit import comparison, modulation

MONTHS = ["2024-04", "2024-05", "2025-04", "2025-05"]


def estimated(
    modulation_gwh: list[float], potential_gwh: list[float] | None = None
) -> pd.DataFrame:
    """modulation.Months on MONTHS, each with 40 negative-price hours."""
    potential = potential_gwh or [100.0, 200.0, 150.0, 300.0]
    return modulation.Months.validate(
        pd.DataFrame(
            {
                "negative_hours": 40,
                "potential_gwh": potential,
                "published_gwh": [p - m for p, m in zip(potential, modulation_gwh, strict=True)],
                "modulation_gwh": modulation_gwh,
                "thin_stratum_gwh": 0.0,
            },
            index=pd.Index(MONTHS, name="month"),
        )
    )


def reference(gwh: list[float], technology: str = "solar") -> pd.DataFrame:
    """RTE volumes as references.load_rte_modulation gives them, on MONTHS."""
    return pd.DataFrame(
        {
            "month": pd.PeriodIndex(MONTHS, freq="M"),
            "technology": technology,
            "block": "total",
            "gwh": gwh,
        }
    )


def test_an_estimate_equal_to_rte_passes_every_test() -> None:
    volumes = [30.0, 50.0, 40.0, 90.0]

    found = comparison.compare(estimated(volumes), reference(volumes), "solar")

    assert found.years["ratio"].tolist() == pytest.approx([1.0, 1.0])
    assert found.log_error == 0.0
    assert found.baseline_log_error > 0
    assert found.spearman == pytest.approx(1.0)
    assert found.tests == {"annual_ratio": True, "monthly_detail": True, "monthly_rates": True}


def test_the_baseline_takes_its_rate_from_the_other_years() -> None:
    found = comparison.compare(
        estimated([30.0, 50.0, 40.0, 90.0]), reference([30.0, 50.0, 40.0, 90.0]), "solar"
    )

    # 2025 gives the 2024 rate, (40 + 90) / (150 + 300), and 2024 the 2025 rate, 80 / 300.
    rate_2024, rate_2025 = 130 / 450, 80 / 300
    assert found.months["baseline_gwh"].tolist() == pytest.approx(
        [100 * rate_2024, 200 * rate_2024, 150 * rate_2025, 300 * rate_2025]
    )


def test_an_annual_ratio_outside_the_range_fails() -> None:
    found = comparison.compare(
        estimated([30.0, 50.0, 60.0, 130.0]), reference([30.0, 50.0, 40.0, 90.0]), "solar"
    )

    assert found.years.loc[2025, "ratio"] == pytest.approx(190 / 130)
    assert found.years["within"].tolist() == [True, False]
    assert not found.tests["annual_ratio"]


def test_a_month_without_positive_modulation_has_no_log_error() -> None:
    found = comparison.compare(
        estimated([-5.0, 50.0, 40.0, 90.0]), reference([30.0, 50.0, 40.0, 90.0]), "solar"
    )

    assert math.isinf(found.log_error)
    assert not found.tests["monthly_detail"]


def test_only_months_above_20_gwh_at_rte_with_30_negative_hours_are_tested() -> None:
    months = estimated([30.0, 50.0, 40.0, 90.0])
    months.loc["2025-04", "negative_hours"] = 29

    found = comparison.compare(months, reference([10.0, 50.0, 40.0, 90.0]), "solar")

    assert found.months["tested"].tolist() == [False, True, False, True]


def test_wind_is_compared_with_the_rte_wind_file() -> None:
    rte = pd.concat([reference([1.0] * 4, "solar"), reference([30.0, 50.0, 40.0, 90.0], "wind")])

    found = comparison.compare(estimated([30.0, 50.0, 40.0, 90.0]), rte, "wind_onshore")

    assert found.months["rte_gwh"].tolist() == [30.0, 50.0, 40.0, 90.0]
