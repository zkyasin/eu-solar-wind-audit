import dataclasses

import numpy as np
import pandas as pd
import pytest
from synthetic import PV, solar_year, wind_year

from eu_solar_wind_audit import bootstrap, days, eligibility, fits, folds, modulation, physics

HOURS = pd.date_range("2024-01-01", "2024-04-01", freq="h", inclusive="left", tz="UTC", unit="ns")


@pytest.fixture(scope="module")
def solar_years() -> list[eligibility.YearInputs]:
    return [solar_year(year) for year in (2019, 2020, 2021)]


def test_each_month_draws_as_many_days_as_it_has() -> None:
    drawn = bootstrap.draw(HOURS, np.random.default_rng(0))

    local = days.local_days(HOURS)
    by_day = drawn.weights.groupby(local)
    # One weight per Europe/Paris day: all its hours are drawn together.
    assert (by_day.min() == by_day.max()).all()
    per_day = by_day.first()
    months = per_day.groupby(pd.DatetimeIndex(per_day.index).to_period("M")).agg(["sum", "size"])
    assert (months["sum"] == months["size"]).all()
    assert drawn.weights.dtype == np.int64


def test_capacity_factors_are_drawn_per_month_within_the_range() -> None:
    drawn = bootstrap.draw(HOURS, np.random.default_rng(0))

    local = days.local_days(HOURS)
    by_month = drawn.capacity_factors.groupby(local.to_period("M"))
    assert (by_month.min() == by_month.max()).all()
    assert drawn.capacity_factors.between(*bootstrap.CAPACITY_RANGE).all()
    assert drawn.dc_ac_ratio in bootstrap.DC_AC_RATIOS


def test_long_blocks_draw_each_day_once(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bootstrap, "MEAN_BLOCK_DAYS", 1e12)

    drawn = bootstrap.draw(HOURS, np.random.default_rng(0))

    assert (drawn.weights == 1).all()


def test_blocks_wrap_around_the_month(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bootstrap, "MEAN_BLOCK_DAYS", 1e12)
    rng = np.random.default_rng(3)
    start = int(np.random.default_rng(3).integers(10))

    positions = bootstrap._stationary(10, rng)

    assert positions.tolist() == [(start + step) % 10 for step in range(10)]


@pytest.mark.parametrize("technology", ["solar", "wind_onshore"])
def test_a_replicate_without_resampling_gives_the_estimate(technology: physics.Technology) -> None:
    years = (
        [solar_year(year) for year in (2019, 2020, 2021)]
        if technology == "solar"
        else [wind_year(year) for year in (2019, 2020, 2021)]
    )
    report = folds.solar(years, PV) if technology == "solar" else folds.wind_onshore(years)
    fit = fits.Fit(technology, report.final, {}, None)
    hours = pd.DatetimeIndex(pd.concat([inputs.arrays.covariates for inputs in years]).index)
    unchanged = bootstrap.Draw(
        weights=pd.Series(1, index=hours),
        dc_ac_ratio=PV.dc_ac_ratio,
        capacity_factors=pd.Series(1.0, index=hours),
    )

    found = bootstrap.replicate(years, technology, PV, unchanged)

    estimate = modulation.months(modulation.negative_hours(years, fit))["modulation_gwh"]
    assert found.index.tolist() == estimate.index.tolist()
    assert found.to_numpy() == pytest.approx(estimate.to_numpy(), rel=1e-9)


def test_replicates_are_the_same_for_the_same_seed(
    solar_years: list[eligibility.YearInputs],
) -> None:
    first = bootstrap.replicates(solar_years, "solar", PV, count=3, seed=1)
    again = bootstrap.replicates(solar_years, "solar", PV, count=3, seed=1)
    other = bootstrap.replicates(solar_years, "solar", PV, count=3, seed=2)

    pd.testing.assert_frame_equal(first, again)
    assert not first.equals(other)
    assert first.index.name == "replicate"
    assert first["dc_ac_ratio"].isin(bootstrap.DC_AC_RATIOS).all()


def test_intervals_take_the_percentiles_of_months_and_of_their_yearly_sums() -> None:
    replicated = pd.DataFrame(
        {
            "2024-05": np.arange(101.0),
            "2024-06": 2 * np.arange(101.0),
            "2025-05": np.full(101, 7.0),
            "dc_ac_ratio": 1.2,
        }
    )

    found = bootstrap.intervals(replicated)

    assert found.index.tolist() == ["2024-05", "2024-06", "2025-05", "2024", "2025"]
    assert found.loc["2024-05"].to_numpy() == pytest.approx([5.0, 95.0])
    assert found.loc["2024"].to_numpy() == pytest.approx([15.0, 285.0])
    assert found.loc["2025"].to_numpy() == pytest.approx([7.0, 7.0])


def test_target_hours_leave_the_fit(solar_years: list[eligibility.YearInputs]) -> None:
    hours = pd.DatetimeIndex(pd.concat([inputs.arrays.covariates for inputs in solar_years]).index)
    eligible = pd.concat([inputs.eligibility["eligible"] for inputs in solar_years])
    targets = eligible & pd.Series(np.arange(hours.size) % 50 == 7, index=hours)
    unchanged = bootstrap.Draw(
        weights=pd.Series(1, index=hours),
        dc_ac_ratio=PV.dc_ac_ratio,
        capacity_factors=pd.Series(1.0, index=hours),
    )
    lowered = []
    for inputs in solar_years:
        published = inputs.published.copy()
        published.loc[targets.reindex(published.index).to_numpy(), "energy_mwh"] -= 10.0
        lowered.append(dataclasses.replace(inputs, published=published))

    before = bootstrap.replicate(solar_years, "solar", PV, unchanged, targets)
    after = bootstrap.replicate(lowered, "solar", PV, unchanged, targets)

    # Outside the fit, 10 MWh less published at each target hour is 10 MWh more estimated.
    month = pd.DatetimeIndex(hours[targets.to_numpy()]).tz_convert("Europe/Paris").strftime("%Y-%m")
    expected = pd.Series(10.0, index=month).groupby(level=0).sum() / 1000
    assert (after - before).to_numpy() == pytest.approx(expected.to_numpy())
