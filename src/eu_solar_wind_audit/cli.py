"""Command-line entry point."""

import argparse
from pathlib import Path

from eu_solar_wind_audit.sources import odre


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="eu-solar-wind-audit")
    commands = parser.add_subparsers(dest="command", required=True)
    snapshot = commands.add_parser(
        "snapshot-realtime",
        help="save the current éCO2mix real-time export with its retrieval time and checksum",
    )
    snapshot.add_argument("directory", type=Path)
    args = parser.parse_args(argv)

    if args.command == "snapshot-realtime":
        with odre.make_client() as client:
            saved = odre.snapshot_realtime(client, args.directory)
        print(saved.path)
        print(saved.path.with_suffix(".json"))
