import dataclasses

import numpy as np
import pandas as pd
import pytest
from synthetic import PV, solar_year

from eu_solar_wind_audit import (
    bootstrap,
    days,
    eligibility,
    fits,
    folds,
    modulation,
    placebo,
)

YEARS = (2019, 2020, 2021)


def priced(inputs: eligibility.YearInputs, price: float) -> eligibility.YearInputs:
    """`inputs` whose hours of positive price are priced `price`."""
    classes = inputs.price_classes.copy()
    classes.loc[classes["mean_price"] > 0, "mean_price"] = price
    return dataclasses.replace(inputs, price_classes=classes)


@pytest.fixture(scope="module")
def years() -> list[eligibility.YearInputs]:
    return [priced(solar_year(year), 10.0) for year in YEARS]


@pytest.fixture(scope="module")
def fit(years: list[eligibility.YearInputs]) -> fits.Fit:
    return fits.Fit("solar", folds.solar(years, PV).final, {}, None)


def consumption(years: list[eligibility.YearInputs]) -> pd.Series:
    hours = pd.DatetimeIndex(pd.concat([inputs.published for inputs in years]).index)
    return pd.Series(np.random.default_rng(0).uniform(30e3, 80e3, hours.size), index=hours)


def test_each_n_hour_takes_a_candidate_of_its_cell_once(
    years: list[eligibility.YearInputs], fit: fits.Fit
) -> None:
    found = placebo.match(years, fit, consumption(years))

    negative = pd.concat([eligibility.negative_price_hours(inputs) for inputs in years])
    assert found.counts.index.tolist() == list(YEARS)
    assert (found.counts["matched"] + found.counts["unmatched"]).sum() == negative.sum()
    assert found.pairs.is_unique
    assert set(found.pairs) == set(found.targets.index[found.targets.to_numpy()])
    eligible = pd.concat([inputs.eligibility["eligible"] for inputs in years])
    assert eligible[found.pairs].all()
    n_hours = pd.DatetimeIndex(found.pairs.index)
    placebo_hours = pd.DatetimeIndex(found.pairs.to_numpy())
    for keys in (
        lambda hours: hours.tz_convert("Europe/Paris").month,
        lambda hours: hours.tz_convert("Europe/Paris").hour,
        lambda hours: days.day_types(hours).to_numpy(),
    ):
        assert (keys(n_hours) == keys(placebo_hours)).all()


def test_the_nearest_candidate_is_chosen(
    years: list[eligibility.YearInputs], fit: fits.Fit
) -> None:
    load = consumption(years)
    found = placebo.match(years, fit, load)
    first_n, first_placebo = found.pairs.index[0], found.pairs.iloc[0]
    inputs = years[0]
    hours = pd.DatetimeIndex(inputs.published.index)
    local = hours.tz_convert("Europe/Paris")
    p_star = modulation.calibrated_potential(inputs, fit)
    factor = p_star / inputs.arrays.covariates["capacity_mw"]
    same_cell = (
        (local.month == first_n.tz_convert("Europe/Paris").month)
        & (local.hour == first_n.tz_convert("Europe/Paris").hour)
        & (days.day_types(hours) == days.day_types(pd.DatetimeIndex([first_n])).iloc[0])
        & inputs.eligibility["eligible"]
        & (inputs.price_classes["mean_price"] == 10.0)
    )
    candidates = inputs.eligibility["eligible"] & (inputs.price_classes["mean_price"] == 10.0)
    scale_factor = factor[candidates].std()
    scale_load = load.reindex(hours)[candidates].std()
    distance = ((factor - factor[first_n]) / scale_factor) ** 2 + (
        (load.reindex(hours) - load[first_n]) / scale_load
    ) ** 2

    assert first_placebo == distance[same_cell].idxmin()


def test_hours_priced_outside_the_placebo_range_are_never_chosen(fit: fits.Fit) -> None:
    years = [priced(solar_year(year), 50.0) for year in YEARS]

    found = placebo.match(years, fit, consumption(years))
    report = placebo.evaluate(years, "solar", PV, fit, consumption(years), count=5)

    assert not found.targets.any()
    assert (found.counts["matched"] == 0).all()
    assert report.years.empty
    assert report.years.columns.tolist() == placebo.REPORT_COLUMNS


def test_an_injected_modulation_moves_the_estimate_by_its_volume(
    years: list[eligibility.YearInputs], fit: fits.Fit
) -> None:
    found = placebo.match(years, fit, consumption(years))
    hours = pd.DatetimeIndex(found.targets.index)
    unchanged = bootstrap.Draw(
        weights=pd.Series(1, index=hours),
        dc_ac_ratio=PV.dc_ac_ratio,
        capacity_factors=pd.Series(1.0, index=hours),
    )

    before = bootstrap.replicate(years, "solar", PV, unchanged, found.targets)
    after = bootstrap.replicate(
        placebo.inject(years, found.targets), "solar", PV, unchanged, found.targets
    )

    published = pd.concat([inputs.published["energy_mwh"] for inputs in years])
    injected = placebo.INJECTED_SHARE * published[found.targets.to_numpy()]
    month = pd.DatetimeIndex(injected.index).tz_convert("Europe/Paris").strftime("%Y-%m")
    volume = injected.groupby(month).sum() / 1000
    assert (after - before).to_numpy() == pytest.approx(volume.to_numpy())


def test_the_report_shows_the_injection_moves_the_estimate_by_its_volume(
    years: list[eligibility.YearInputs], fit: fits.Fit
) -> None:
    report = placebo.evaluate(years, "solar", PV, fit, consumption(years), count=5)

    table = report.years
    assert table.index.tolist() == [str(year) for year in YEARS]
    for column in ("gwh", "low_gwh", "high_gwh"):
        shift = table[f"injected_{column}"] - table[f"placebo_{column}"]
        assert shift.to_numpy() == pytest.approx(table["volume_gwh"].to_numpy())
    covers_zero = (table["placebo_low_gwh"] <= 0) & (table["placebo_high_gwh"] >= 0)
    assert table["injection_covered"].tolist() == covers_zero.tolist()
    assert report.pairs.size == table["matched"].sum()
