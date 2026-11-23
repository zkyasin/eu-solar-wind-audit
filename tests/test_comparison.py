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


def intervals(estimates: dict[str, float], share: float) -> pd.DataFrame:
    """Intervals as estimate-modulation writes them: the estimate of each year, with a half-width
    of `share` of it."""
    estimate = pd.Series(estimates)
    return pd.DataFrame(
        {
            "low_gwh": estimate * (1 - share),
            "high_gwh": estimate * (1 + share),
            "estimate_gwh": estimate,
        }
    )


MDE = pd.Series({"2024": 10.0, "2025": 10.0})


def test_an_estimate_equal_to_rte_with_a_narrow_interval_is_concordant() -> None:
    volumes = [30.0, 50.0, 40.0, 90.0]
    found = comparison.compare(estimated(volumes), reference(volumes), "solar")

    judged = comparison.verdict(found, intervals({"2024": 80.0, "2025": 130.0}, 0.1), MDE)

    assert judged.verdict == "concordant"
    assert judged.years.index.tolist() == ["2024", "2025"]
    assert judged.years["half_width_share"].tolist() == pytest.approx([0.1, 0.1])
    assert not judged.years["inconclusive"].any()


def test_a_wide_interval_or_an_estimate_below_the_effect_is_inconclusive() -> None:
    volumes = [30.0, 50.0, 40.0, 90.0]
    found = comparison.compare(estimated(volumes), reference(volumes), "solar")
    narrow = intervals({"2024": 80.0, "2025": 130.0}, 0.1)

    wide = comparison.verdict(found, intervals({"2024": 80.0, "2025": 130.0}, 0.3), MDE)
    small = comparison.verdict(found, narrow, pd.Series({"2024": 100.0, "2025": 10.0}))

    assert wide.verdict == "inconclusive"
    assert small.verdict == "inconclusive"
    assert small.years["inconclusive"].tolist() == [True, False]
    # 100 GWh is more than 20 % of the 80 GWh of RTE in 2024.
    assert small.years["likely_inconclusive"].tolist() == [True, False]


def test_a_failed_annual_ratio_is_discordant() -> None:
    found = comparison.compare(
        estimated([30.0, 50.0, 60.0, 130.0]), reference([30.0, 50.0, 40.0, 90.0]), "solar"
    )

    judged = comparison.verdict(found, intervals({"2024": 80.0, "2025": 190.0}, 0.1), MDE)

    assert judged.verdict == "discordant"


def test_a_failed_monthly_test_is_concordant_in_volume_only() -> None:
    found = comparison.compare(
        estimated([50.0, 30.0, 90.0, 40.0]), reference([30.0, 50.0, 40.0, 90.0]), "solar"
    )
    assert found.tests["annual_ratio"]
    assert not found.tests["monthly_rates"]

    judged = comparison.verdict(found, intervals({"2024": 80.0, "2025": 130.0}, 0.1), MDE)

    assert judged.verdict == "concordant in volume only"


def test_a_year_without_an_estimate_is_inconclusive() -> None:
    volumes = [30.0, 50.0, 40.0, 90.0]
    found = comparison.compare(estimated(volumes), reference(volumes), "solar")

    judged = comparison.verdict(found, intervals({"2024": 80.0}, 0.1), MDE)

    assert judged.verdict == "inconclusive"
    assert judged.years["inconclusive"].tolist() == [False, True]
