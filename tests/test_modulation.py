import dataclasses

import numpy as np
import pandas as pd
import pytest
from synthetic import PV, solar_year

from eu_solar_wind_audit import calibration, eligibility, fits, folds, modulation, strata

YEARS = (2019, 2020, 2021)


@pytest.fixture(scope="module")
def years() -> list[eligibility.YearInputs]:
    return [solar_year(year) for year in YEARS]


@pytest.fixture(scope="module")
def fit(years: list[eligibility.YearInputs]) -> fits.Fit:
    report = folds.solar(years, PV)
    return fits.Fit(
        technology="solar",
        final=report.final,
        fold_calibrations={year: fold.calibration for year, fold in report.folds.items()},
        git_describe=None,
    )


def potential_with_level(inputs: eligibility.YearInputs, fit: fits.Fit) -> pd.Series:
    """p* by hand: the potential of the frozen fit, its stratum ratio and the level of its year."""
    potential = calibration.potential_mwh(inputs.arrays, fit.final.calibration)
    with_strata = strata.apply_ratios(fit.final.ratios, potential, inputs.arrays.covariates)
    published = inputs.published["energy_mwh"]
    level = (published[inputs.eligibility["eligible"]].sum()) / (
        with_strata[inputs.eligibility["eligible"]].sum()
    )
    return with_strata * level


def published_at_negative_prices(
    inputs: eligibility.YearInputs, fit: fits.Fit, share: float
) -> eligibility.YearInputs:
    """`inputs` whose negative-price hours publish `share` of their potential p*."""
    negative = eligibility.negative_price_hours(inputs)
    published = inputs.published.copy()
    p_star = potential_with_level(inputs, fit)
    published.loc[negative, "energy_mwh"] = share * p_star[negative]
    return dataclasses.replace(inputs, published=published)


def test_negative_price_hours_compare_the_published_output_with_p_star(
    years: list[eligibility.YearInputs], fit: fits.Fit
) -> None:
    found = modulation.negative_hours(years, fit)

    negative = pd.concat([eligibility.negative_price_hours(inputs) for inputs in years])
    assert found.index.equals(negative.index[negative.to_numpy()])
    p_star = pd.concat([potential_with_level(inputs, fit) for inputs in years])
    assert found["potential_mwh"].to_numpy() == pytest.approx(p_star[found.index].to_numpy())
    assert (found["modulation_mwh"] == found["potential_mwh"] - found["published_mwh"]).all()


def test_a_known_modulation_at_negative_prices_is_recovered_exactly(
    years: list[eligibility.YearInputs], fit: fits.Fit
) -> None:
    # Negative-price hours enter neither the fit, the strata nor the level, so changing them
    # leaves p* as it was.
    modulated = [published_at_negative_prices(inputs, fit, 0.7) for inputs in years]

    found = modulation.negative_hours(modulated, fit)

    assert found["modulation_mwh"].to_numpy() == pytest.approx(
        0.3 * found["potential_mwh"].to_numpy()
    )


def test_months_are_those_of_paris_and_sum_the_modulation(
    years: list[eligibility.YearInputs], fit: fits.Fit
) -> None:
    found = modulation.negative_hours(years, fit)

    table = modulation.months(found)

    paris = pd.DatetimeIndex(found.index).tz_convert("Europe/Paris").strftime("%Y-%m")
    expected = found.groupby(paris)["modulation_mwh"].sum() / 1000
    assert table["modulation_gwh"].to_numpy() == pytest.approx(expected.to_numpy())
    assert table.index.tolist() == expected.index.tolist()
    assert table["negative_hours"].sum() == len(found)
    thin = found["modulation_mwh"].where(found["thin_stratum"], 0.0).groupby(paris).sum() / 1000
    assert table["thin_stratum_gwh"].to_numpy() == pytest.approx(thin.to_numpy())


def test_thin_strata_are_those_with_few_training_hours(
    years: list[eligibility.YearInputs], fit: fits.Fit
) -> None:
    found = modulation.negative_hours(years, fit)

    hour_strata = pd.concat(
        [
            strata.classify_hours(
                calibration.potential_mwh(inputs.arrays, fit.final.calibration),
                inputs.arrays.covariates,
                fit.final.ratios.decile_edges,
            )
            for inputs in years
        ]
    ).loc[found.index]
    table = fit.final.ratios.table.set_index(strata.STRATUM_KEY)["training_hours"]
    keys = pd.MultiIndex.from_frame(hour_strata.astype({"decile": "int64"}))
    expected = table.reindex(keys).to_numpy() < modulation.THIN_STRATUM_HOURS
    assert np.array_equal(found["thin_stratum"].to_numpy(), expected)


def test_a_year_without_negative_prices_adds_no_hour(
    years: list[eligibility.YearInputs], fit: fits.Fit
) -> None:
    inputs = years[0]
    classes = inputs.price_classes.assign(regime="reference")
    without = dataclasses.replace(inputs, price_classes=classes)

    assert modulation.negative_hours([without], fit).empty
