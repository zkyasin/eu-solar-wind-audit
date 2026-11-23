import tomllib
from importlib.metadata import version
from pathlib import Path

PYPROJECT = Path(__file__).parents[1] / "pyproject.toml"


def test_installed_package_has_the_project_version() -> None:
    project = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]
    assert version("eu-solar-wind-audit") == project["version"]
