"""Conversion of real éCO2mix definitive data around the 2025 clock changes. Each fixture holds
the three local days around the change, 96 rows per local day as published."""

from pathlib import Path

import pandas as pd
import pytest

from eu_solar_wind_audit.grid import hourly_from_period_end_means

FIXTURES = Path(__file__).parent / "fixtures" / "eco2mix"
# Retrieved 2026-11-22: https://odre.opendatasoft.com/api/explore/v2.1/catalog/datasets/eco2mix-national-cons-def/exports/csv?select=date_heure,solaire,eolien&where=date%20IN%20(%222025-03-29%22,%222025-03-30%22,%222025-03-31%22)&order_by=date,heure
SPRING = FIXTURES / "eco2mix-national-cons-def-2025-03-30.csv"
# Retrieved 2026-11-22: https://odre.opendatasoft.com/api/explore/v2.1/catalog/datasets/eco2mix-national-cons-def/exports/csv?select=date_heure,solaire,eolien&where=date%20IN%20(%222025-10-25%22,%222025-10-26%22,%222025-10-27%22)&order_by=date,heure
AUTUMN = FIXTURES / "eco2mix-national-cons-def-2025-10-26.csv"


def read_export(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.pop("date_heure"), utc=True))
    return frame


def utc(*stamps: str) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(stamps, tz="UTC")


@pytest.mark.parametrize("path", [SPRING, AUTUMN])
def test_exports_fill_production_every_other_quarter_hour(path: Path) -> None:
    export = read_export(path)

    minutes = pd.DatetimeIndex(export.index).minute

    assert len(export) == 3 * 96
    filled = export["eolien"].notna()
    assert (export["solaire"].notna() == filled).all()
    assert (minutes[filled] % 30 == 0).all()
    assert (minutes[~filled] % 30 == 15).all()


def test_spring_export_repeats_utc_stamps() -> None:
    export = read_export(SPRING)

    repeated = export.index[export.index.duplicated()]

    assert repeated.equals(
        utc("2025-03-30 01:00", "2025-03-30 01:15", "2025-03-30 01:30", "2025-03-30 01:45")
    )


def test_autumn_export_lacks_the_repeated_hour() -> None:
    stamps = pd.DatetimeIndex(read_export(AUTUMN).index)

    expected = pd.date_range(stamps[0], stamps[-1], freq="15min")

    assert not stamps.has_duplicates
    assert expected.difference(stamps).equals(
        utc("2025-10-26 00:00", "2025-10-26 00:15", "2025-10-26 00:30", "2025-10-26 00:45")
    )


@pytest.mark.parametrize(
    ("path", "affected"),
    [
        # The half-hours ending 01:00 and 01:30 are both repeated.
        (SPRING, utc("2025-03-30 00:00", "2025-03-30 01:00")),
        # The half-hours ending 00:00 and 00:30 are missing.
        (AUTUMN, utc("2025-10-25 23:00", "2025-10-26 00:00")),
    ],
)
@pytest.mark.parametrize("column", ["solaire", "eolien"])
def test_clock_change_hours_are_incomplete(
    path: Path, affected: pd.DatetimeIndex, column: str
) -> None:
    hourly = hourly_from_period_end_means(read_export(path)[column])

    # The first and last hours lack the half-hour outside the export.
    assert hourly.index[~hourly["complete"]].equals(hourly.index[[0, -1]].union(affected))
    assert hourly.loc[affected, "energy_mwh"].isna().all()
    assert hourly["energy_mwh"].isna().equals(~hourly["complete"])


@pytest.mark.parametrize(
    ("path", "hour", "energy_mwh"),
    [
        # Half-hours ending 23:30 (6790 MW) and 00:00 (7036 MW).
        (SPRING, "2025-03-29 23:00", 6913.0),
        # Half-hours ending 02:30 (7438 MW) and 03:00 (7250 MW), after the repeated stamps.
        (SPRING, "2025-03-30 02:00", 7344.0),
        # Half-hours ending 01:30 (10697 MW) and 02:00 (10364 MW), after the missing hour.
        (AUTUMN, "2025-10-26 01:00", 10530.5),
    ],
)
def test_hours_next_to_the_clock_change_use_their_own_half_hours(
    path: Path, hour: str, energy_mwh: float
) -> None:
    hourly = hourly_from_period_end_means(read_export(path)["eolien"])

    assert hourly.loc[pd.Timestamp(hour, tz="UTC"), "energy_mwh"] == energy_mwh
