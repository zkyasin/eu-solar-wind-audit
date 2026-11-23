"""Command-line entry point."""

import argparse
import dataclasses
import json
import subprocess
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from eu_solar_wind_audit import calibrate, dc_ac, fits, physics
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
