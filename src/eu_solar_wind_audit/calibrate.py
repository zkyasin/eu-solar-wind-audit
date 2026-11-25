"""Leave-one-year-out calibration of a country, as the `calibrate` command runs it (evaluation
plan, "Two model objects" and "Axis 1: model quality").

`run` loads the inputs of every year of evaluation.TRAINING_YEARS and `report_of` gives the report
of folds, the composition of the eligible hours of each year with the strata of the final fit,
and, for solar, the DC/AC profile of item 1. `write` saves the tables as parquet next to a JSON
file that describes the run, and the frozen fit with fits.write.
"""

import dataclasses
import hashlib
import importlib.metadata
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
from pandera.typing import DataFrame

from eu_solar_wind_audit import (
    calibration,
    days,
    dc_ac,
    eligibility,
    evaluation,
    fits,
    folds,
    physics,
    precompute,
    strata,
)
from eu_solar_wind_audit.config import Country
from eu_solar_wind_audit.sources import (
    bkg,
    cbs,
    energinet,
    ens,
    gisco,
    mastr,
    natural_earth,
    rivm,
    smard,
)


@dataclass(frozen=True)
class Run:
    """A calibration of `technology` on its yearly `years`: the report of folds, the composition
    of the eligible hours, and the DC/AC report of item 1 for solar."""

    technology: physics.Technology
    years: tuple[eligibility.YearInputs, ...]
    report: folds.Report
    composition: DataFrame[strata.Composition]
    dc_ac: dc_ac.Report | None


def run(
    client: httpx.Client,
    technology: physics.Technology,
    parameters: Country,
    *,
    replicates: int = dc_ac.REPLICATES,
) -> Run:
    """Calibration of `technology` in the country of `parameters` on the inputs of every year of
    evaluation.TRAINING_YEARS, with `replicates` day-bootstrap replicates for the DC/AC profile."""
    years = [
        eligibility.year_inputs(client, technology, year, parameters)
        for year in evaluation.TRAINING_YEARS
    ]
    return report_of(technology, years, parameters, replicates=replicates)


def report_of(
    technology: physics.Technology,
    years: Sequence[eligibility.YearInputs],
    parameters: Country,
    *,
    replicates: int = dc_ac.REPLICATES,
) -> Run:
    """Calibration of `technology` on its yearly inputs `years`, as `run` gives it."""
    if technology == "solar":
        report = folds.solar(years, parameters.pv())
        arrays = [inputs.arrays for inputs in years]
        solar_arrays = [part for part in arrays if isinstance(part, precompute.SolarArrays)]
        profile = dc_ac.report(
            solar_arrays,
            pd.concat([inputs.published["energy_mwh"] for inputs in years]),
            pd.concat([inputs.eligibility["eligible"] for inputs in years]),
            pd.concat([inputs.price_classes for inputs in years]),
            parameters.pv(),
            replicates=replicates,
        )
    else:
        report = folds.wind_onshore(years)
        profile = None
    return Run(technology, tuple(years), report, _composition(years, report.final), profile)


def write(calibrated: Run, output: Path, fits_dir: Path, git_describe: str | None) -> list[Path]:
    """Write the tables of `calibrated` to `output` as `<technology>-<table>.parquet`, the JSON
    file `<technology>-calibration.json` that describes the run, and the frozen fit to
    `fits_dir`/`<technology>.json`. Gives the paths written."""
    technology = calibrated.technology
    report = calibrated.report
    tables = {
        "folds": _fold_table(report),
        "strata": _stratum_tables(report),
        "predictions": report.predictions,
        "metrics": report.metrics,
        "summary": report.period_summary,
        "negative-prices": report.sensitivity.negative_prices,
        "composition": calibrated.composition,
    }
    if calibrated.dc_ac is not None:
        tables |= _dc_ac_tables(calibrated.dc_ac)
    output.mkdir(parents=True, exist_ok=True)
    written = []
    for name, table in tables.items():
        path = output / f"{technology}-{name}.parquet"
        table.to_parquet(path)
        written.append(path)

    description = output / f"{technology}-calibration.json"
    content = describe(calibrated, git_describe)
    description.write_text(json.dumps(content, indent=1) + "\n", encoding="utf-8")
    frozen = fits_dir / f"{technology}.json"
    fits.write(
        frozen,
        fits.Fit(
            technology=technology,
            final=report.final,
            fold_calibrations={year: fold.calibration for year, fold in report.folds.items()},
            git_describe=git_describe,
        ),
    )
    return [*written, description, frozen]


def describe(calibrated: Run, git_describe: str | None) -> dict[str, Any]:
    """What a calibration rests on: the code version, the rules and parameters, and, for each
    year, its hours, the status of its published hours, its weather and capacity vintages, its
    price classes and the SHA-256 of its published hourly series."""
    years = []
    for inputs in calibrated.years:
        covariates = inputs.arrays.covariates
        published = inputs.published
        digest = hashlib.sha256(
            pd.util.hash_pandas_object(published, index=True).to_numpy().tobytes()
        ).hexdigest()
        years.append(
            {
                "year": int(pd.DatetimeIndex(published.index)[0].year),
                "hours": len(published),
                "eligible_hours": int(inputs.eligibility["eligible"].sum()),
                # SMARD gives no publication status.
                "status": published["status"].fillna("none").value_counts().to_dict()
                if "status" in published
                else None,
                "era5_source": covariates["era5_source"].value_counts().to_dict(),
                "provisional_capacity_hours": int(covariates["capacity_provisional"].sum()),
                "price_classes": inputs.price_classes["regime"].value_counts().to_dict(),
                "published_sha256": digest,
            }
        )
    return {
        "technology": calibrated.technology,
        "git_describe": git_describe,
        "package_version": importlib.metadata.version("eu-solar-wind-audit"),
        "fold_years": list(folds.fold_years_of(calibrated.years, calibrated.technology)),
        "estimator": {
            "huber_tuning": calibration.HUBER_TUNING,
            "tolerances": [calibration.FTOL, calibration.XTOL, calibration.GTOL],
            "shrinkage_days": strata.SHRINKAGE_DAYS,
            "dc_ac_bootstrap": None
            if calibrated.dc_ac is None
            else {
                "replicates": int(calibrated.dc_ac.bootstrap.minima.size),
                "seed": calibrated.dc_ac.bootstrap.seed,
            },
        },
        "inputs": _sources(calibrated.years[0].country, calibrated.technology) | {"years": years},
    }


def _sources(country: str, technology: physics.Technology) -> dict[str, str]:
    """The capacity, areas, prices and published series that the inputs of `country` rest on."""
    zone = eligibility.PRICE_ZONES[country]
    prices = f"SMARD.de, day-ahead prices of {zone}, filter {smard.PRICES[zone][0]}"
    if country == "FR":
        return {
            "capacity": f"SDES dashboard, media {physics.SDES_MEDIA[technology]}",
            "regions": f"Natural Earth admin-1 {natural_earth.VERSION}",
            "prices": prices,
            "published": "RTE, éCO2mix via ODRÉ",
        }
    if country == "DE":
        return {
            "capacity": f"Marktstammdatenregister, {mastr.EXPORT_URL}",
            "regions": f"BKG VG5000 districts, state of {bkg.EDITION}",
            "prices": prices,
            "published": f"SMARD.de, feed-in of Germany, filter {smard.FEED_IN[technology]}",
        }
    if country == "NL":
        revisions = ", ".join(
            f"{table} revised {cbs.modified(cbs.download(table))}"
            for table in (cbs.PROVINCES, cbs.HUB_HEIGHTS)
        )
        return {
            "capacity": f"CBS ({revisions}), spread by the RIVM map {rivm.LAYER}",
            "regions": f"GISCO {gisco.EDITION} municipalities",
            "prices": prices,
            "published": "ENTSO-E Transparency Platform, actual generation of onshore wind (B19) "
            "of the NL zone",
        }
    quantity: energinet.Quantity = "solar_total" if technology == "solar" else technology
    capacity = (
        f"Energi Data Service, {energinet.CAPACITY}"
        if technology == "solar"
        else f"Danish Energy Agency, turbine register, {ens.CURRENT_URL} and {ens.HISTORIC_URL}"
    )
    return {
        "capacity": capacity,
        "regions": f"GISCO {gisco.EDITION} municipalities",
        "prices": prices,
        "published": (
            f"Energi Data Service, {energinet.SETTLEMENT} of {country}: "
            + " + ".join(energinet.QUANTITIES[quantity])
        ),
    }


def _composition(
    years: Sequence[eligibility.YearInputs], final: folds.Potential
) -> DataFrame[strata.Composition]:
    """Eligible hours of each year by group, with the strata of the potential of `final`."""
    potential = pd.concat(
        [calibration.potential_mwh(inputs.arrays, final.calibration) for inputs in years]
    )
    covariates = pd.concat([inputs.arrays.covariates for inputs in years])
    hour_strata = strata.classify_hours(potential, covariates, final.ratios.decile_edges)
    eligible = pd.concat([inputs.eligibility["eligible"] for inputs in years])
    return strata.composition(hour_strata, days.sky_classes(covariates), eligible)


def _fold_table(report: folds.Report) -> pd.DataFrame:
    """One row per fit, each fold by its held-out year and then the final fit: its parameters,
    the final pass of its Huber fit and the pooled ratio of its strata."""
    fitted = {str(year): fold for year, fold in report.folds.items()} | {"final": report.final}
    rows = []
    for name, potential in fitted.items():
        values = dataclasses.asdict(potential.calibration)
        fit = values.pop("fit")
        fit["training_years"] = ",".join(str(year) for year in fit["training_years"])
        rows.append({"fit": name, **values, **fit, "pooled_ratio": potential.ratios.pooled_ratio})
    return pd.DataFrame(rows)


def _stratum_tables(report: folds.Report) -> pd.DataFrame:
    """The stratum table of every fit, with the fit as for `_fold_table`."""
    fitted = {str(year): fold for year, fold in report.folds.items()} | {"final": report.final}
    return pd.concat(
        [potential.ratios.table.assign(fit=name) for name, potential in fitted.items()],
        ignore_index=True,
    )


def _dc_ac_tables(found: dc_ac.Report) -> dict[str, pd.DataFrame]:
    """The tables of the DC/AC report: the cost and fitted values of each profile at each ratio,
    the minima, the sample size and the bootstrap minima."""
    profiles = {"sample": found.profile, **found.subsets}
    rows = [
        pd.DataFrame(
            {
                "estimate": name,
                "dc_ac_ratio": precompute.DC_AC_GRID,
                "cost": profile.costs,
                "level": profile.parameters[:, 0],
                "horizontal_share": profile.parameters[:, 1],
                "seasonal_amplitude": profile.parameters[:, 2],
                "f_scale": profile.f_scale,
            }
        )
        for name, profile in profiles.items()
    ]
    return {
        "dc-ac-profiles": pd.concat(rows, ignore_index=True),
        "dc-ac-minima": found.minima,
        "dc-ac-sample": found.sample_size,
        "dc-ac-bootstrap": pd.DataFrame({"minimum": found.bootstrap.minima}),
    }
