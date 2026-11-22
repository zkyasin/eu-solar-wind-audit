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
        raise ValueError(f"{path.name}: expected only {label!r}, found {sorted(labels)}")
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
