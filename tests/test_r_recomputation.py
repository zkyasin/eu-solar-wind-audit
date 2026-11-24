import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from eu_solar_wind_audit import cli, modulation, signature

pytestmark = pytest.mark.skipif(shutil.which("Rscript") is None, reason="Rscript not found")

SCRIPT = Path(__file__).parents[1] / "R" / "recompute.R"


def calendar() -> pd.DataFrame:
    """Twelve hours of each day of May and June 2024, with the cell, season and day columns of the
    signature tables."""
    time = pd.date_range("2024-05-01", "2024-07-01", freq="h", inclusive="left", tz="UTC")
    time = time[(time.hour >= 6) & (time.hour < 18)].as_unit("ns").rename("time")
    return pd.DataFrame(
        {
            "cell": [f"{stamp.hour}-{stamp.month}" for stamp in time],
            "season": np.where(time.month == 5, "MAM", "JJA"),
            "day": time.tz_localize(None).normalize(),
        },
        index=time,
    )


def signatures(directory: Path, rng: np.random.Generator) -> None:
    """The outputs of feed-in-signature and consumption-signature for synthetic hours."""
    feed_in = calendar()
    sky = pd.Series(rng.choice(["clear", "mixed", "overcast"], 61), index=feed_in["day"].unique())
    feed_in["sky_class"] = feed_in["day"].map(sky)
    feed_in["capacity_factor"] = rng.uniform(0.0, 0.8, len(feed_in))
    effect = feed_in["sky_class"].map({"clear": 0.05, "mixed": 0.0, "overcast": -0.03})
    feed_in["log_ratio"] = effect + rng.normal(0.0, 0.05, len(feed_in))
    rows = signature.contrasts(feed_in).assign(series="smard")
    cli.write_signature(directory, "feed-in-signature", rows, {"smard": feed_in}, {})

    load = calendar()
    load["raw_mwh"] = rng.uniform(0.0, 1000.0, len(load))
    load["load_mwh"] = 5000.0 - 0.1 * load["raw_mwh"] + rng.normal(0.0, 20.0, len(load))
    rows = signature.dips(load).assign(series="smard_consumption")
    cli.write_signature(directory, "consumption-signature", rows, {"smard_consumption": load}, {})


def modulation_months(directory: Path, rng: np.random.Generator) -> None:
    """The hours and months that estimate-modulation writes, for N hours on both sides of the
    end of May in Europe/Paris time."""
    time = pd.DatetimeIndex(
        ["2024-05-31 12:00", "2024-05-31 21:00", "2024-05-31 22:00", "2024-06-01 12:00"],
        tz="UTC",
        name="time",
    ).as_unit("ns")
    potential = rng.uniform(100.0, 1000.0, len(time))
    published = potential * rng.uniform(0.2, 1.0, len(time))
    hours = pd.DataFrame(
        {
            "published_mwh": published,
            "potential_mwh": potential,
            "modulation_mwh": potential - published,
            "thin_stratum": [False, True, False, True],
        },
        index=time,
    )
    directory.mkdir(parents=True)
    hours.to_parquet(directory / "solar-modulation-hours.parquet")
    modulation.months(hours).to_parquet(directory / "solar-modulation-months.parquet")


def recompute(tmp_path: Path) -> tuple[subprocess.CompletedProcess[str], dict[str, Any]]:
    report = tmp_path / "report.json"
    finished = subprocess.run(
        [
            "Rscript",
            str(SCRIPT),
            "--modulation",
            str(tmp_path / "modulation"),
            "--signature",
            str(tmp_path / "signature"),
            "--output",
            str(report),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return finished, json.loads(report.read_text(encoding="utf-8"))


def test_r_finds_the_python_sums_and_regressions(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    modulation_months(tmp_path / "modulation", rng)
    signatures(tmp_path / "signature", rng)

    finished, report = recompute(tmp_path)

    assert finished.returncode == 0, finished.stderr
    assert report["passed"] is True
    # Two months of five columns, six contrasts and three dips with their standard errors.
    assert len(report["comparisons"]) == 10 + 2 * 6 + 2 * 3


def test_r_reports_a_value_it_does_not_find(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    modulation_months(tmp_path / "modulation", rng)
    signatures(tmp_path / "signature", rng)
    path = tmp_path / "modulation" / "solar-modulation-months.parquet"
    months = pd.read_parquet(path)
    months["modulation_gwh"] += np.where(months.index == "2024-06", 1e-3, 0.0)
    months.to_parquet(path)

    finished, report = recompute(tmp_path)

    assert finished.returncode == 1
    failed = [row["statistic"] for row in report["comparisons"] if not row["pass"]]
    assert failed == ["modulation solar 2024-06 modulation_gwh"]
