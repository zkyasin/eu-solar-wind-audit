"""Reference series published by producers, kept for comparison only.

Calibration code must never read them: the model would then be judged on its own training data.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

RTE_MONTHS = pd.period_range("2020-01", "2026-10", freq="M")
# Order of the three stacked blocks in each RTE file. They match the "Global", "Jours ouvrés" and
# "Jours non ouvrés" series of the charts on analysesetdonnees.rte-france.com (see the manifest).
RTE_BLOCKS = ("total", "working_days", "non_working_days")
RTE_FILES = {
    "solar": ("RTE_PAD_solaire_modulation.csv", "Volume de modulation de la production solaire"),
    "wind": ("RTE_PAD_eolien_modulation.csv", "Volume de modulation de la production éolienne"),
}
RTE_COLUMNS = ["Date", "Filière", "Valeur (GWh)"]
TOLERANCE_GWH = 1e-9
HIRTH_FILE = "market_curtailment_germany.csv"
HIRTH_COLUMNS = ["year", "technology", "twh", "source"]
SDES_FILE = "autoconsommation_electricite_france_2018_2024.xlsx"
SDES_SHEET = "Figures"
# Columns of the table of self-consumption by technology, after the year.
SDES_TECHNOLOGIES = (
    "Éolienne",
    "Hydraulique",
    "Photovoltaïque",
    "Thermique hors cogénération",
    "Thermique en cogénération",
)
TOLERANCE_TWH = 1e-9


def load_rte_modulation(directory: Path) -> pd.DataFrame:
    """Monthly volumes modulated during negative prices, in GWh, one row per month, technology
    and block."""
    frames = [
        _read_rte_modulation(directory / name, technology, label)
        for technology, (name, label) in RTE_FILES.items()
    ]
    return pd.concat(frames, ignore_index=True)


def _read_rte_modulation(path: Path, technology: str, label: str) -> pd.DataFrame:
    raw = pd.read_csv(
        path,
        sep=";",
        decimal=",",
        encoding="utf-8-sig",
        dtype={"Date": str, "Filière": str, "Valeur (GWh)": float},
    )
    if list(raw.columns) != RTE_COLUMNS:
        raise ValueError(f"{path.name}: expected columns {RTE_COLUMNS}, found {list(raw.columns)}")
    labels = set(raw["Filière"])
    if labels != {label}:
        raise ValueError(f"{path.name}: expected only {label!r}, found {sorted(labels, key=str)}")
    shape = (len(RTE_BLOCKS), len(RTE_MONTHS))
    if len(raw) != shape[0] * shape[1]:
        raise ValueError(
            f"{path.name}: expected {shape[0]} blocks of {shape[1]} months, found {len(raw)} rows"
        )

    months = raw["Date"].to_numpy().reshape(shape)
    if not (months == RTE_MONTHS.strftime("%Y-%m").to_numpy()).all():
        raise ValueError(
            f"{path.name}: each block must run month by month from {RTE_MONTHS[0]} "
            f"to {RTE_MONTHS[-1]}"
        )

    total, working, non_working = raw["Valeur (GWh)"].to_numpy().reshape(shape)
    # Written so that a missing value, which makes the gap NaN, also fails.
    mismatch = ~(np.abs(total - working - non_working) <= TOLERANCE_GWH)
    if mismatch.any():
        raise ValueError(
            f"{path.name}: total differs from working plus non-working days, "
            f"first in {RTE_MONTHS[mismatch][0]}"
        )

    return pd.DataFrame(
        {
            "month": pd.PeriodIndex(raw["Date"], freq="M"),
            "technology": technology,
            "block": np.repeat(RTE_BLOCKS, shape[1]),
            "gwh": raw["Valeur (GWh)"],
        }
    )


def load_hirth_curtailment(directory: Path) -> pd.DataFrame:
    """Yearly market-based curtailment of Germany in Hirth (2026), in GWh, one row per year and
    technology: `wind`, onshore and offshore together, and `solar`."""
    path = directory / HIRTH_FILE
    raw = pd.read_csv(path, dtype={"year": int, "technology": str, "twh": float, "source": str})
    if list(raw.columns) != HIRTH_COLUMNS:
        raise ValueError(
            f"{path.name}: expected columns {HIRTH_COLUMNS}, found {list(raw.columns)}"
        )
    if not raw["technology"].isin(["wind", "solar"]).all():
        raise ValueError(f"{path.name}: expected the technologies wind and solar")
    return pd.DataFrame(
        {"year": raw["year"], "technology": raw["technology"], "gwh": 1000.0 * raw["twh"]}
    )


def load_sdes_self_consumption(directory: Path) -> "pd.Series[float]":
    """Yearly self-consumption of solar photovoltaics in France, overseas departments included, in
    GWh, indexed by year: the first table of the SDES data file, self-consumption by technology in
    TWh, whose technologies must add up to its total."""
    path = directory / SDES_FILE
    raw = pd.read_excel(path, sheet_name=SDES_SHEET, header=None)
    (header,) = np.flatnonzero(raw[0] == "Annee").tolist()
    width = len(SDES_TECHNOLOGIES)
    labels = raw.iloc[header + 1].iloc[1 : 1 + width].tolist()
    if labels != list(SDES_TECHNOLOGIES) or raw.iat[header, 1 + width] != "Total":
        raise ValueError(
            f"{path.name}: expected the columns {list(SDES_TECHNOLOGIES)} and Total, found {labels}"
        )

    first = header + 2
    # The table runs down to the first row without a year.
    years = pd.to_numeric(raw[0].iloc[first:], errors="coerce")
    count = int(years.notna().cumprod().sum())
    table = raw.iloc[first : first + count, : width + 2].astype(float)
    table.columns = pd.Index(["year", *SDES_TECHNOLOGIES, "Total"])
    total = table[list(SDES_TECHNOLOGIES)].sum(axis=1)
    mismatch = ~(np.abs(total - table["Total"]) <= TOLERANCE_TWH)
    if mismatch.any():
        raise ValueError(
            f"{path.name}: technologies do not add up to the total, "
            f"first in {int(table['year'][mismatch].iloc[0])}"
        )
    return pd.Series(
        1000.0 * table["Photovoltaïque"].to_numpy(),
        index=pd.Index(table["year"].astype(int), name="year"),
        name="gwh",
    )


def verify_manifest(directory: Path) -> None:
    """Raise if a file listed in `directory/manifest.json` differs from its recorded size or
    SHA-256."""
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    for name, entry in manifest["files"].items():
        data = (directory / name).read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if len(data) != entry["size_bytes"] or digest != entry["sha256"]:
            raise ValueError(
                f"{name}: expected {entry['size_bytes']} bytes with SHA-256 {entry['sha256']}, "
                f"found {len(data)} bytes with SHA-256 {digest}"
            )
