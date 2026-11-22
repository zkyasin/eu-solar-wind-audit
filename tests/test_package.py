from importlib.metadata import version


def test_package_is_installed() -> None:
    assert version("eu-solar-wind-audit") == "0.0.0"
