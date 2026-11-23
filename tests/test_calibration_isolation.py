"""The calibration never reads the reference series it is judged against (evaluation plan, "Two
model objects"): the calibrate-france command runs end to end in a separate process, from a
directory without references/, where importing the reference module fails."""

import subprocess
import sys
from pathlib import Path

TESTS = Path(__file__).parent
CONFIG = TESTS.parent / "config" / "countries" / "fr.toml"

# Run with the tests directory and the configuration file as arguments, on synthetic yearly inputs.
SCRIPT = """
import importlib.abc
import sys


class RefuseReferences(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name == "eu_solar_wind_audit.references":
            raise ImportError("the calibration imported the reference series")
        return None


sys.meta_path.insert(0, RefuseReferences())
sys.path.insert(0, sys.argv[1])

from synthetic import wind_year

from eu_solar_wind_audit import cli, eligibility

eligibility.year_inputs = lambda client, technology, year, parameters: wind_year(year)
cli.main(
    ["calibrate-france", "wind_onshore", "out", "--fits-dir", "fits", "--config", sys.argv[2]]
)
assert "eu_solar_wind_audit.references" not in sys.modules
print("isolated")
"""


def test_calibration_never_reads_the_reference_series(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-c", SCRIPT, str(TESTS), str(CONFIG)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[-1] == "isolated"
    assert (tmp_path / "fits" / "wind_onshore.json").exists()
    assert not (tmp_path / "references").exists()
