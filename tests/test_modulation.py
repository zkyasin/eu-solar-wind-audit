import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from synthetic import PV, solar_year, wind_year

from eu_solar_wind_audit import calibration, cli, eligibility, fits, folds, modulation, strata
from eu_solar_wind_audit.sources import energinet, entsoe, odre, smard

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


def test_the_command_writes_the_estimate_and_its_comparison_with_rte(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fit: fits.Fit
) -> None:
    def year_inputs(
        client: object, technology: str, year: int, parameters: object
    ) -> eligibility.YearInputs:
        return solar_year(year)

    def national_hourly(
        client: object, series: str, start: pd.Timestamp, end: pd.Timestamp, **_: object
    ) -> pd.DataFrame:
        hours = pd.date_range(start, end, freq="h", inclusive="left", name="time")
        return pd.DataFrame({"energy_mwh": 50e3, "complete": True}, index=hours)

    monkeypatch.setattr(eligibility, "year_inputs", year_inputs)
    monkeypatch.setattr(odre, "national_hourly", national_hourly)
    fits.write(tmp_path / "fits" / "solar.json", fit)
    rte = Path(__file__).parents[1] / "references" / "rte"

    cli.main(
        [
            "estimate-modulation",
            "FR",
            "solar",
            str(tmp_path / "out"),
            "--fits-dir",
            str(tmp_path / "fits"),
            "--references",
            str(rte),
            "--replicates",
            "3",
        ]
    )

    names = {path.name for path in (tmp_path / "out").iterdir()}
    assert names == {
        "solar-modulation-hours.parquet",
        "solar-modulation-months.parquet",
        "solar-rte-months.parquet",
        "solar-rte-years.parquet",
        "solar-modulation-replicates.parquet",
        "solar-modulation-intervals.parquet",
        "solar-modulation-intervals-dc-ac-sensitivity.parquet",
        "solar-verdict-years.parquet",
        "solar-placebo-years.parquet",
        "solar-placebo-pairs.parquet",
        "solar-modulation.json",
    }
    summary = json.loads((tmp_path / "out" / "solar-modulation.json").read_text())
    assert set(summary["tests"]) == {"annual_ratio", "monthly_detail", "monthly_rates"}
    assert summary["bootstrap"]["replicates"] == 3
    # Every fifth hour makes no whole day, so no day has a sky class: no hour measures the
    # detectable effect, and no year is conclusive.
    assert summary["mde_gwh"] == {str(year): None for year in range(2019, 2026)}
    assert summary["verdict"] == "inconclusive"
    # No synthetic hour is priced 5 to 15 EUR/MWh, so no N hour has a placebo.
    assert summary["placebo"]["matched_hours"] == 0
    intervals = pd.read_parquet(tmp_path / "out" / "solar-modulation-intervals.parquet")
    assert intervals.columns.tolist() == ["low_gwh", "high_gwh", "estimate_gwh"]
    assert {"2019", "2020", "2021"} <= set(intervals.index)


def test_the_command_compares_germany_with_hirth_without_a_verdict(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fit: fits.Fit
) -> None:
    def year_inputs(
        client: object, technology: str, year: int, parameters: object
    ) -> eligibility.YearInputs:
        return solar_year(year, country="DE")

    def load(region: str, start: pd.Timestamp, end: pd.Timestamp, **_: object) -> pd.DataFrame:
        hours = pd.date_range(start, end, freq="h", inclusive="left", name="time")
        return pd.DataFrame({"energy_mwh": 50e3, "complete": True}, index=hours)

    monkeypatch.setattr(eligibility, "year_inputs", year_inputs)
    monkeypatch.setattr(smard, "fetch_hourly_load", load)
    fits.write(tmp_path / "fits" / "solar.json", fit)
    hirth = Path(__file__).parents[1] / "references" / "hirth"

    cli.main(
        [
            "estimate-modulation",
            "DE",
            "solar",
            str(tmp_path / "out"),
            "--fits-dir",
            str(tmp_path / "fits"),
            "--references",
            str(hirth),
            "--replicates",
            "3",
        ]
    )

    names = {path.name for path in (tmp_path / "out").iterdir()}
    assert "solar-hirth-years.parquet" in names
    assert not {"solar-rte-years.parquet", "solar-verdict-years.parquet"} & names
    summary = json.loads((tmp_path / "out" / "solar-modulation.json").read_text())
    assert summary["country"] == "DE"
    assert summary["verdict"] is None
    assert set(summary["hirth_ratios"]) == {"2025"}
    years = pd.read_parquet(tmp_path / "out" / "solar-hirth-years.parquet")
    assert years["reference_gwh"].tolist() == [600.0]
    assert {"low_gwh", "high_gwh", "estimate_gwh", "ratio"} <= set(years.columns)


def test_the_command_estimates_a_danish_price_area_without_reference(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fit: fits.Fit
) -> None:
    def year_inputs(
        client: object, technology: str, year: int, parameters: object
    ) -> eligibility.YearInputs:
        return solar_year(year, country="DK1")

    loads: list[tuple[str, str]] = []

    def settlement(
        area: str, quantity: str, start: pd.Timestamp, end: pd.Timestamp, **_: object
    ) -> pd.DataFrame:
        loads.append((area, quantity))
        hours = pd.date_range(start, end, freq="h", inclusive="left", name="time")
        return pd.DataFrame({"energy_mwh": 2e3, "complete": True}, index=hours)

    monkeypatch.setattr(eligibility, "year_inputs", year_inputs)
    monkeypatch.setattr(energinet, "fetch_hourly_settlement", settlement)
    fits.write(tmp_path / "fits" / "solar.json", fit)

    cli.main(
        [
            "estimate-modulation",
            "DK1",
            "solar",
            str(tmp_path / "out"),
            "--fits-dir",
            str(tmp_path / "fits"),
            "--replicates",
            "3",
        ]
    )

    # The placebo matches on the gross consumption of the price area.
    assert loads == [("DK1", "gross_consumption")]
    names = {path.name for path in (tmp_path / "out").iterdir()}
    assert "solar-modulation-intervals.parquet" in names
    assert not {"solar-rte-years.parquet", "solar-hirth-years.parquet"} & names
    summary = json.loads((tmp_path / "out" / "solar-modulation.json").read_text())
    assert summary["country"] == "DK1"
    assert summary["verdict"] is None
    assert "hirth_ratios" not in summary


def test_the_command_estimates_dutch_onshore_wind_without_reference(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    years = [wind_year(year, country="NL") for year in YEARS]
    report = folds.wind_onshore(years)
    fitted = fits.Fit(
        technology="wind_onshore",
        final=report.final,
        fold_calibrations={year: fold.calibration for year, fold in report.folds.items()},
        git_describe=None,
    )

    def year_inputs(
        client: object, technology: str, year: int, parameters: object
    ) -> eligibility.YearInputs:
        return wind_year(year, country="NL")

    areas: list[str] = []

    def load(area: str, start: pd.Timestamp, end: pd.Timestamp, **_: object) -> pd.DataFrame:
        areas.append(area)
        hours = pd.date_range(start, end, freq="h", inclusive="left", name="time")
        return pd.DataFrame({"energy_mwh": 12e3, "complete": True}, index=hours)

    monkeypatch.setattr(eligibility, "year_inputs", year_inputs)
    monkeypatch.setattr(entsoe, "fetch_hourly_load", load)
    fits.write(tmp_path / "fits" / "wind_onshore.json", fitted)

    cli.main(
        [
            "estimate-modulation",
            "NL",
            "wind_onshore",
            str(tmp_path / "out"),
            "--fits-dir",
            str(tmp_path / "fits"),
            "--replicates",
            "3",
        ]
    )

    # The placebo matches on the total load of the NL zone; the Netherlands have no solar model.
    assert areas == ["NL"]
    summary = json.loads((tmp_path / "out" / "wind_onshore-modulation.json").read_text())
    assert summary["country"] == "NL"
    assert summary["verdict"] is None
