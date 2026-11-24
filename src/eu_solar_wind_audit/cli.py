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
    figures,
    fits,
    folds,
    modulation,
    physics,
    placebo,
)
from eu_solar_wind_audit.anomalies import breaks, detection, inputs, thresholds
from eu_solar_wind_audit.config import CONFIG_DIR, Country, load_country
from eu_solar_wind_audit.sources import natural_earth, odre, sdes, smard

# A week of June 2025 with negative prices on five of its days, at whose hours published onshore
# wind was 52 % of its potential (data/processed/modulation-fr).
WEEK_START = pd.Timestamp("2025-06-02", tz="Europe/Paris")
# The year of findings DE-1 and DE-2 of docs/anomalies-2019-2025.md.
SUM_OF_PARTS_YEAR = 2025
TECHNOLOGIES: tuple[physics.Technology, ...] = ("solar", "wind_onshore")


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
        "calibrate",
        help="fit the calibrated potential of a country leaving one year out of 2019-2025, write "
        "its tables with a JSON file describing the run, and the frozen fit",
    )
    fit.add_argument("country", choices=list(eligibility.PRICE_ZONES))
    fit.add_argument("technology", choices=["solar", "wind_onshore"])
    fit.add_argument("output", type=Path, help="directory of the tables and their description")
    fit.add_argument("--config", type=Path, help="config/countries/<country>.toml by default")
    fit.add_argument("--fits-dir", type=Path, help="fits/<COUNTRY> by default")
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
    draw = commands.add_parser(
        "draw-figures",
        help="draw the figures of the README, in a light and a dark version, from the results of "
        "estimate-modulation and of detect-anomalies for Germany",
    )
    draw.add_argument("output", type=Path, help="directory of the SVG files")
    draw.add_argument("--modulation", type=Path, default=Path("data/processed/modulation-fr"))
    draw.add_argument("--anomalies", type=Path, default=Path("data/processed/anomalies-de"))
    draw.add_argument("--config", type=Path, default=CONFIG_DIR / "fr.toml")
    draw.add_argument("--fits-dir", type=Path, default=fits.FITS_DIR)
    draw.add_argument("--references", type=Path, default=Path("references/rte"))
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
    elif args.command == "calibrate":
        config = args.config or CONFIG_DIR / f"{args.country.lower()}.toml"
        parameters = load_country(config)
        if parameters.country != args.country:
            parser.error(f"{config} describes {parameters.country}, not {args.country}")
        with odre.make_client() as client:
            calibrated = calibrate.run(
                client, args.technology, parameters, replicates=args.replicates
            )
        fits_dir = args.fits_dir or fits.FITS_ROOT / args.country
        for path in calibrate.write(calibrated, args.output, fits_dir, git_describe()):
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
    elif args.command == "draw-figures":
        drawn = f"Drawn on {date.today().isoformat()} at commit {(git_commit() or 'unknown')[:12]}."
        for path in [
            *draw_week(args.output, load_country(args.config), args.fits_dir, drawn),
            *draw_modulation(args.output, args.modulation, args.references, drawn),
            *draw_sum_of_parts(args.output, args.anomalies, drawn),
        ]:
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
    volumes of `reference_dir` and, with an interval, give the verdict and run the placebo and the
    injection. Write the tables and a JSON summary to `output`. Gives the paths written."""
    # The reference series are imported here only, so that calibrate never loads them.
    from eu_solar_wind_audit import comparison, references

    first = pd.Timestamp(evaluation.TRAINING_YEARS[0], 1, 1, tz="UTC")
    end = pd.Timestamp(evaluation.TRAINING_YEARS[-1] + 1, 1, 1, tz="UTC")
    with odre.make_client() as client:
        years = [
            eligibility.year_inputs(client, technology, year, parameters)
            for year in evaluation.TRAINING_YEARS
        ]
        consumption = odre.national_hourly(client, "load", first, end)
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
    placebo_years = None
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
        checked = placebo.evaluate(
            years,
            technology,
            parameters.solar,
            fit,
            consumption["energy_mwh"],
            count=replicates,
        )
        tables["placebo-years"] = checked.years
        tables["placebo-pairs"] = checked.pairs.to_frame()
        placebo_years = checked.years
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
        "placebo": None
        if placebo_years is None
        else {
            "matched_hours": int(placebo_years["matched"].sum()),
            "unmatched_hours": int(placebo_years["unmatched"].sum()),
            "years": len(placebo_years),
            "years_within": int(placebo_years["placebo_within"].sum()),
            "injection_coverage": _finite(float(placebo_years["injection_covered"].mean())),
        },
    }
    path = output / f"{technology}-modulation.json"
    path.write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    return [*written, path]


def draw_week(output: Path, parameters: Country, fits_dir: Path, drawn: str) -> list[Path]:
    """Draw the published output of mainland France against the potential of the frozen fits of
    `fits_dir` over the week from WEEK_START, with its negative-price hours."""
    start = WEEK_START.tz_convert("UTC")
    end = (WEEK_START + pd.DateOffset(days=7)).tz_convert("UTC")
    hourly: dict[str, pd.DataFrame] = {}
    fitted_by: set[str] = set()
    statuses: set[str] = set()
    weather: set[str] = set()
    with odre.make_client() as client:
        for technology in TECHNOLOGIES:
            year = eligibility.year_inputs(client, technology, WEEK_START.year, parameters)
            fit = fits.read(fits_dir / f"{technology}.json")
            frame = pd.DataFrame(
                {
                    "published_mwh": year.published["energy_mwh"],
                    "potential_mwh": modulation.calibrated_potential(year, fit),
                }
            )
            week = (frame.index >= start) & (frame.index < end)
            hourly[technology] = frame[week]
            fitted_by.add(str(fit.git_describe))
            statuses.update(year.published.loc[week, "status"].dropna())
            weather.update(year.arrays.covariates.loc[week, "era5_source"])
    # Both technologies read the same day-ahead prices.
    negative = year.price_classes["regime"].eq("negative")[week]
    return figures.write_week(
        output,
        hourly,
        negative,
        f"Potential: frozen fit {', '.join(sorted(fitted_by))}, "
        f"{' and '.join(sorted(weather))} weather (Copernicus Climate Change Service). "
        f"Published: RTE, éCO2mix {' and '.join(sorted(statuses))} data via ODRÉ, Licence Ouverte "
        f"2.0. Prices: Bundesnetzagentur | SMARD.de.\n{drawn}",
    )


def draw_modulation(
    output: Path, modulation_dir: Path, reference_dir: Path, drawn: str
) -> list[Path]:
    """Draw the yearly estimates of estimate-modulation in `modulation_dir`, with their interval,
    against the RTE volumes of `reference_dir`."""
    years: dict[str, pd.DataFrame] = {}
    runs: set[str] = set()
    for technology in TECHNOLOGIES:
        summary = json.loads(
            (modulation_dir / f"{technology}-modulation.json").read_text(encoding="utf-8")
        )
        runs.add(
            f"{summary['git_describe']} with the frozen fit {summary['fit_git_describe']}, "
            f"{summary['bootstrap']['replicates']} bootstrap replicates"
        )
        rte = pd.read_parquet(modulation_dir / f"{technology}-rte-years.parquet")
        intervals = pd.read_parquet(modulation_dir / f"{technology}-modulation-intervals.parquet")
        estimates = intervals.loc[rte.index.astype(str), ["estimate_gwh", "low_gwh", "high_gwh"]]
        years[technology] = estimates.set_axis(rte.index).join(rte["rte_gwh"])
    manifest = json.loads((reference_dir / "manifest.json").read_text(encoding="utf-8"))
    updated = max(file["last_modified"][:10] for file in manifest["files"].values())
    return figures.write_modulation(
        output,
        years,
        f"Estimate: estimate-modulation at {'; '.join(sorted(runs))}. RTE: source RTE, "
        f"analysesetdonnees.rte-france.com, last updated {updated}.\n{drawn}",
    )


def draw_sum_of_parts(output: Path, anomalies_dir: Path, drawn: str) -> list[Path]:
    """Draw the daily gap between the solar feed-in of Germany and the sum of its control areas
    over SUM_OF_PARTS_YEAR, with the parts_gap findings of detect-anomalies in `anomalies_dir`."""
    time_zone = "Europe/Berlin"
    start = pd.Timestamp(SUM_OF_PARTS_YEAR, 1, 1, tz=time_zone).tz_convert("UTC")
    end = pd.Timestamp(SUM_OF_PARTS_YEAR + 1, 1, 1, tz=time_zone).tz_convert("UTC")
    energy = {}
    for area in inputs.SMARD_AREAS:
        feed_in = smard.fetch_hourly_feed_in(area, "solar", start, end)
        energy[area] = feed_in["energy_mwh"].where(feed_in["complete"])
    total = energy.pop("DE")
    found = pd.read_parquet(anomalies_dir / "DE-findings.parquet")
    flagged = found[
        (found["series"] == "smard.DE.solar")
        & (found["kind"] == "parts_gap")
        & (found["start"] < end)
        & (found["end"] > start)
    ]
    run = json.loads((anomalies_dir / "DE-detection.json").read_text(encoding="utf-8"))
    return figures.write_sum_of_parts(
        output,
        figures.daily_gap_gwh(total, pd.DataFrame(energy), time_zone),
        flagged,
        time_zone,
        f"Germany, solar feed-in, {SUM_OF_PARTS_YEAR}: national total minus the sum of its four "
        "control areas",
        "Data: Bundesnetzagentur | SMARD.de, CC BY 4.0, quarter-hour feed-in of Germany and of "
        f"50Hertz, Amprion, TenneT and TransnetBW. Findings: detect-anomalies at "
        f"{run['git_describe']}.\n{drawn}",
    )


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
