"""Command-line entry point."""

import argparse
import dataclasses
import hashlib
import json
import math
import subprocess
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from eu_solar_wind_audit import (
    bootstrap,
    calibrate,
    dc_ac,
    detectability,
    eligibility,
    evaluation,
    fits,
    folds,
    modulation,
    physics,
)
from eu_solar_wind_audit.anomalies import breaks, detection, inputs, thresholds
from eu_solar_wind_audit.config import CONFIG_DIR, Country, load_country
from eu_solar_wind_audit.sources import natural_earth, odre, sdes


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="eu-solar-wind-audit")
    commands = parser.add_subparsers(dest="command", required=True)
    snapshot = commands.add_parser(
        "snapshot-realtime",
        help="save the current éCO2mix real-time export with its retrieval time and checksum",
    )
    snapshot.add_argument("directory", type=Path)
    model = commands.add_parser(
        "model-france",
        help="write the modelled hourly output of mainland France to parquet, with a JSON file "
        "describing its inputs",
    )
    model.add_argument("technology", choices=["solar", "wind_onshore"])
    model.add_argument("start", type=date.fromisoformat, help="first day, UTC")
    model.add_argument("end", type=date.fromisoformat, help="day after the last one, UTC")
    model.add_argument("output", type=Path, help="parquet file; the JSON file goes next to it")
    model.add_argument("--config", type=Path, default=CONFIG_DIR / "fr.toml")
    fit = commands.add_parser(
        "calibrate-france",
        help="fit the calibrated potential of mainland France leaving one year out of 2019-2025, "
        "write its tables with a JSON file describing the run, and the frozen fit",
    )
    fit.add_argument("technology", choices=["solar", "wind_onshore"])
    fit.add_argument("output", type=Path, help="directory of the tables and their description")
    fit.add_argument("--config", type=Path, default=CONFIG_DIR / "fr.toml")
    fit.add_argument("--fits-dir", type=Path, default=fits.FITS_DIR)
    fit.add_argument(
        "--replicates",
        type=int,
        default=dc_ac.REPLICATES,
        help="day-bootstrap replicates of the solar DC/AC profile",
    )
    estimate = commands.add_parser(
        "estimate-modulation",
        help="estimate the output of mainland France modulated during negative prices with the "
        "frozen fit, and compare it with the volumes published by RTE",
    )
    estimate.add_argument("technology", choices=["solar", "wind_onshore"])
    estimate.add_argument("output", type=Path, help="directory of the tables and their summary")
    estimate.add_argument("--config", type=Path, default=CONFIG_DIR / "fr.toml")
    estimate.add_argument("--fits-dir", type=Path, default=fits.FITS_DIR)
    estimate.add_argument("--references", type=Path, default=Path("references/rte"))
    estimate.add_argument(
        "--replicates",
        type=int,
        default=bootstrap.REPLICATES,
        help="bootstrap replicates of the 90 %% interval, none with 0",
    )
    fix = commands.add_parser(
        "fix-anomaly-thresholds",
        help="fix the thresholds of the statistical anomaly detectors of mainland France on "
        "2019-2022 and write them with the reference model they rest on",
    )
    fix.add_argument("--config", type=Path, default=CONFIG_DIR / "fr.toml")
    fix.add_argument("--output", type=Path, default=thresholds.THRESHOLDS_DIR / "FR.json")
    anomalies = commands.add_parser(
        "detect-anomalies",
        help="run the anomaly detectors on the published series of a country over a period, and "
        "write the findings with a JSON file describing the run",
    )
    anomalies.add_argument("country", choices=["FR", "DE"])
    anomalies.add_argument("start", type=date.fromisoformat, help="first day, UTC")
    anomalies.add_argument("end", type=date.fromisoformat, help="day after the last one, UTC")
    anomalies.add_argument(
        "output", type=Path, help="directory of the findings and their description"
    )
    anomalies.add_argument("--config", type=Path, default=CONFIG_DIR / "fr.toml")
    anomalies.add_argument("--thresholds", type=Path, default=thresholds.THRESHOLDS_DIR / "FR.json")
    args = parser.parse_args(argv)

    if args.command == "snapshot-realtime":
        with odre.make_client() as client:
            saved = odre.snapshot_realtime(client, args.directory)
        print(saved.path)
        print(saved.path.with_suffix(".json"))
    elif args.command == "model-france":
        parameters = load_country(args.config)
        start, end = pd.Timestamp(args.start, tz="UTC"), pd.Timestamp(args.end, tz="UTC")
        hourly = physics.model_france(args.technology, start, end, parameters)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        hourly.to_parquet(args.output)
        metadata = run_metadata(args.technology, parameters, hourly)
        sidecar = args.output.with_suffix(".json")
        sidecar.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        print(args.output)
        print(sidecar)
    elif args.command == "calibrate-france":
        parameters = load_country(args.config)
        with odre.make_client() as client:
            calibrated = calibrate.run(
                client, args.technology, parameters, replicates=args.replicates
            )
        for path in calibrate.write(calibrated, args.output, args.fits_dir, git_describe()):
            print(path)
    elif args.command == "estimate-modulation":
        for path in estimate_modulation(
            args.technology,
            args.output,
            load_country(args.config),
            fits.read(args.fits_dir / f"{args.technology}.json"),
            args.references,
            args.replicates,
        ):
            print(path)
    elif args.command == "fix-anomaly-thresholds":
        parameters = load_country(args.config)
        with odre.make_client() as client:
            found = inputs.france_inputs(
                client,
                thresholds.FIX_START,
                thresholds.FIX_END,
                parameters,
                margin=pd.Timedelta(0),
            )
        reference = thresholds.reference_model(parameters, git_describe())
        thresholds.write(args.output, thresholds.fix(found, reference))
        print(args.output)
    elif args.command == "detect-anomalies":
        start, end = pd.Timestamp(args.start, tz="UTC"), pd.Timestamp(args.end, tz="UTC")
        run: dict[str, Any] = {
            "git_describe": git_describe(),
            "ran_at": pd.Timestamp.now(tz="UTC").floor("s").isoformat(),
        }
        fixed = None
        with odre.make_client() as client:
            if args.country == "FR":
                parameters = load_country(args.config)
                fixed = thresholds.read(args.thresholds)
                thresholds.check_reference(
                    fixed, thresholds.reference_model(parameters, git_describe())
                )
                run["thresholds"] = {
                    "path": str(args.thresholds),
                    "sha256": hashlib.sha256(args.thresholds.read_bytes()).hexdigest(),
                    "git_describe": fixed.reference_model["git_describe"],
                }
                # The level breaks read the daily ratios from their first day on.
                found = inputs.france_inputs(client, breaks.FIRST_DAY, end, parameters)
            else:
                found = inputs.germany_inputs(client, start, end)
        detected = detection.detect(found, fixed, start, end)
        for path in detection.write(args.output, found, detected, start, end, run):
            print(path)


def run_metadata(
    technology: physics.Technology, parameters: Country, hourly: pd.DataFrame
) -> dict[str, Any]:
    """Inputs of a model run: parameters, weather and capacity vintages and code version."""
    settings = dataclasses.asdict(parameters)
    settings["era5_box"] = parameters.era5_box._asdict()
    if technology == "wind_onshore":
        settings["wind_models"] = [
            dataclasses.asdict(physics.generation_parameters(generation))
            for generation in parameters.wind_onshore
        ]
    return {
        "technology": technology,
        "first_hour": hourly.index[0].isoformat(),
        "last_hour": hourly.index[-1].isoformat(),
        "parameters": settings,
        "era5_sources": hourly["era5_source"].value_counts().sort_index().to_dict(),
        "capacity": {
            "source": "SDES",
            "url": f"{sdes.BASE_URL}/media/{physics.SDES_MEDIA[technology]}/download",
            "provisional_hours": int(hourly["capacity_provisional"].sum()),
        },
        "regions": f"Natural Earth admin-1 {natural_earth.VERSION}",
        "git_commit": git_commit(),
    }


def estimate_modulation(
    technology: physics.Technology,
    output: Path,
    parameters: Country,
    fit: fits.Fit,
    reference_dir: Path,
    replicates: int,
) -> list[Path]:
    """Estimate the modulation of `technology` over every year of evaluation.TRAINING_YEARS with
    the frozen `fit`, its 90 % interval from `replicates` bootstrap replicates (none with 0), and,
    for solar, the interval with the DC/AC ratios of the sensitivity; give the minimal detectable
    effect of each year from the leave-one-year-out folds; compare the estimate with the RTE
    volumes of `reference_dir` and, with an interval, give the verdict. Write the tables and a
    JSON summary to `output`. Gives the paths written."""
    # The reference series are imported here only, so that calibrate-france never loads them.
    from eu_solar_wind_audit import comparison, references

    with odre.make_client() as client:
        years = [
            eligibility.year_inputs(client, technology, year, parameters)
            for year in evaluation.TRAINING_YEARS
        ]
    hours = modulation.negative_hours(years, fit)
    months = modulation.months(hours)
    report = (
        folds.solar(years, parameters.solar) if technology == "solar" else folds.wind_onshore(years)
    )
    covariates = pd.concat([inputs.arrays.covariates for inputs in years])
    errors = detectability.daily_errors(report.predictions, covariates, technology)
    mde_gwh = detectability.minimal_detectable_effects(errors, hours)
    references.verify_manifest(reference_dir)
    found = comparison.compare(months, references.load_rte_modulation(reference_dir), technology)

    output.mkdir(parents=True, exist_ok=True)
    tables: dict[str, pd.DataFrame] = {
        "modulation-hours": hours,
        "modulation-months": months,
        "rte-months": found.months,
        "rte-years": found.years,
    }
    outcome = None
    if replicates > 0:
        estimate = months["modulation_gwh"]
        by_year = estimate.groupby(estimate.index.str[:4]).sum()
        estimates = pd.concat([estimate, by_year]).rename("estimate_gwh")
        ratios = {"intervals": bootstrap.DC_AC_RATIOS}
        if technology == "solar":
            ratios["intervals-dc-ac-sensitivity"] = bootstrap.DC_AC_SENSITIVITY
        for name, dc_ac_ratios in ratios.items():
            replicated = bootstrap.replicates(
                years, technology, parameters.solar, count=replicates, dc_ac_ratios=dc_ac_ratios
            )
            interval = bootstrap.intervals(replicated).join(estimates)
            tables[f"modulation-{name}"] = interval
            if name == "intervals":
                tables["modulation-replicates"] = replicated
        judged = comparison.verdict(found, tables["modulation-intervals"], mde_gwh)
        tables["verdict-years"] = judged.years
        outcome = judged.verdict
    written = []
    for name, table in tables.items():
        path = output / f"{technology}-{name}.parquet"
        table.to_parquet(path)
        written.append(path)
    summary = {
        "technology": technology,
        "git_describe": git_describe(),
        "fit_git_describe": fit.git_describe,
        "log_error": _finite(found.log_error),
        "baseline_log_error": _finite(found.baseline_log_error),
        "spearman": _finite(found.spearman),
        "tests": found.tests,
        "bootstrap": {
            "replicates": replicates,
            "seed": bootstrap.SEED,
            "mean_block_days": bootstrap.MEAN_BLOCK_DAYS,
            "percentiles": list(bootstrap.PERCENTILES),
        },
        "daily_error_sd": _finite(float(errors.std())),
        "error_days": len(errors),
        "mde_gwh": {year: _finite(float(value)) for year, value in mde_gwh.items()},
        "verdict": outcome,
    }
    path = output / f"{technology}-modulation.json"
    path.write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    return [*written, path]


def _finite(value: float) -> float | None:
    return value if math.isfinite(value) else None


def git_describe() -> str | None:
    """`git describe --always --dirty` of the working tree, or None outside a repository."""
    try:
        result = subprocess.run(
            ["git", "describe", "--always", "--dirty"], capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()
