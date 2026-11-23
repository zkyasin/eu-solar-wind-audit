"""Frozen calibrated potential of mainland France, one JSON file per technology under FITS_DIR.

The final fit of folds is the potential of axis 4 and of the prospective test of 2026, which read
it from the committed file and never refit it (evaluation plan, Deviations). The file also keeps
the parameters of each leave-one-year-out fold and the `git describe` of the code that fitted
them. Floats are written by json.dumps, which keeps them exact; a stratum without training hours
has a null raw ratio.
"""

import dataclasses
import json
import math
from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from eu_solar_wind_audit import calibration, folds, physics, strata

FITS_DIR = Path("fits/FR")


@dataclass(frozen=True)
class Fit:
    """A frozen fit: its technology, the final potential, the calibration of each fold keyed by
    its held-out year, and the `git describe` of the code that fitted them."""

    technology: physics.Technology
    final: folds.Potential
    fold_calibrations: dict[int, folds.Calibration]
    git_describe: str | None


def write(path: Path, fit: Fit) -> None:
    """Write `fit` to `path` as JSON."""
    content = {
        "technology": fit.technology,
        "git_describe": fit.git_describe,
        "final": {
            "calibration": dataclasses.asdict(fit.final.calibration),
            "strata": {
                "decile_edges": list(fit.final.ratios.decile_edges),
                "pooled_ratio": fit.final.ratios.pooled_ratio,
                "table": [_without_nan(row) for row in fit.final.ratios.table.to_dict("records")],
            },
        },
        "folds": {
            str(year): dataclasses.asdict(fitted)
            for year, fitted in sorted(fit.fold_calibrations.items())
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=1, allow_nan=False) + "\n", encoding="utf-8")


def read(path: Path) -> Fit:
    """The fit that `write` wrote to `path`."""
    content = json.loads(path.read_text(encoding="utf-8"))
    technology: physics.Technology = content["technology"]
    final = content["final"]
    ratios = strata.StratumRatios(
        decile_edges=tuple(final["strata"]["decile_edges"]),
        pooled_ratio=final["strata"]["pooled_ratio"],
        table=strata.StratumTable.validate(
            pd.DataFrame(final["strata"]["table"]).astype({"raw_ratio": float})
        ),
    )
    return Fit(
        technology=technology,
        final=folds.Potential(_calibration(technology, final["calibration"]), ratios),
        fold_calibrations={
            int(year): _calibration(technology, fitted) for year, fitted in content["folds"].items()
        },
        git_describe=content["git_describe"],
    )


def _calibration(technology: physics.Technology, content: dict[str, Any]) -> folds.Calibration:
    fit = calibration.HuberFit(
        **{**content["fit"], "training_years": tuple(content["fit"]["training_years"])}
    )
    values = {name: value for name, value in content.items() if name != "fit"}
    if technology == "solar":
        return calibration.SolarCalibration(**values, fit=fit)
    return calibration.WindCalibration(**values, fit=fit)


def _without_nan(row: Mapping[Hashable, Any]) -> dict[str, Any]:
    return {
        str(name): None if isinstance(value, float) and math.isnan(value) else value
        for name, value in row.items()
    }
