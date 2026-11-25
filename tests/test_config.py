from pathlib import Path

import pytest

from eu_solar_wind_audit import precompute
from eu_solar_wind_audit.config import Country, WindGeneration, load_country
from eu_solar_wind_audit.pv import PVParameters, System
from eu_solar_wind_audit.sources.era5 import Box

COUNTRIES = Path(__file__).parents[1] / "config" / "countries"
FRANCE = COUNTRIES / "fr.toml"
GERMANY = COUNTRIES / "de.toml"

MINIMAL = """
country = "XX"
time_zone = "UTC"
perimeter = "metropolitan_without_corsica"

[era5_box]
west = 1.0
south = 2.0
east = 3.0
north = 4.0

[solar]
dc_ac_ratio = 1.3

[[solar.systems]]
tilt = 30.0
azimuth = 180.0
share = 0.75

[[solar.systems]]
tilt = 10.0
azimuth = 90.0
share = 0.25
tracker = true

[[wind_onshore.generations]]
turbine = "VestasV82_1.65MW_82"
hub_height_m = 80.0
last_commissioning_year = 2012

[[wind_onshore.generations]]
turbine = "IEA_Reference_3.4MW_130"
hub_height_m = 100.0
"""


GENERATION = """[[wind_onshore.generations]]
turbine = "VestasV82_1.65MW_82"
hub_height_m = 85.0
last_commissioning_year = {year}
"""


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "xx.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_file_is_parsed_into_the_model_dataclasses(tmp_path: Path) -> None:
    country = load_country(write(tmp_path, MINIMAL))

    assert country == Country(
        country="XX",
        time_zone="UTC",
        perimeter="metropolitan_without_corsica",
        era5_box=Box(west=1.0, south=2.0, east=3.0, north=4.0),
        solar=PVParameters(
            fleet=(System(30.0, 180.0, 0.75), System(10.0, 90.0, 0.25, tracker=True)),
            dc_ac_ratio=1.3,
        ),
        wind_onshore=(
            WindGeneration("VestasV82_1.65MW_82", 80.0, 2012),
            WindGeneration("IEA_Reference_3.4MW_130", 100.0),
        ),
    )


def test_pv_fields_absent_from_the_file_keep_their_defaults(tmp_path: Path) -> None:
    country = load_country(write(tmp_path, MINIMAL))

    defaults = PVParameters(fleet=country.pv().fleet)
    assert country.pv().gamma_pdc == defaults.gamma_pdc
    assert country.pv().dc_losses == defaults.dc_losses


def test_a_country_can_have_no_solar_model(tmp_path: Path) -> None:
    start, end = MINIMAL.index("[solar]"), MINIMAL.index("[[wind_onshore")
    country = load_country(write(tmp_path, MINIMAL[:start] + MINIMAL[end:]))

    assert country.solar is None
    with pytest.raises(ValueError, match="no solar model"):
        country.pv()


def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    text = MINIMAL.replace("dc_ac_ratio = 1.3", "dc_ac = 1.3")

    with pytest.raises(TypeError, match="dc_ac"):
        load_country(write(tmp_path, text))


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ('perimeter = "metropolitan_without_corsica"', 'perimeter = "all"', "perimeter"),
        ("hub_height_m = 100.0", "hub_height_m = 100.0\nlast_commissioning_year = 2020", "last"),
        ("last_commissioning_year = 2012", "", "last"),
        ('turbine = "IEA_Reference_3.4MW_130"', 'turbine = "IEA_Reference_15MW_240"', "onshore"),
        ("share = 0.25", "share = 0.3", "sum to 1"),
    ],
)
def test_invalid_settings_are_rejected(tmp_path: Path, old: str, new: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        load_country(write(tmp_path, MINIMAL.replace(old, new)))


def test_generations_must_follow_each_other(tmp_path: Path) -> None:
    earlier = "last_commissioning_year = 2012\n\n" + GENERATION.format(year=2010)

    with pytest.raises(ValueError, match="increasing"):
        load_country(write(tmp_path, MINIMAL.replace("last_commissioning_year = 2012\n", earlier)))


def test_france_file_is_valid() -> None:
    france = load_country(FRANCE)

    assert france.country == "FR"
    assert france.perimeter == "metropolitan_without_corsica"
    years = [generation.last_commissioning_year for generation in france.wind_onshore]
    assert years == [2012, 2020, None]


def test_germany_file_is_valid() -> None:
    germany = load_country(GERMANY)

    assert germany.perimeter == "germany"
    assert len(germany.pv().fleet) == 7
    assert sum(system.share for system in germany.pv().fleet) == pytest.approx(1.0)
    assert [generation.last_commissioning_year for generation in germany.wind_onshore] == [
        2005,
        2012,
        2016,
        2020,
        None,
    ]


@pytest.mark.parametrize("zone", ["DK1", "DK2"])
def test_danish_files_are_valid(zone: str) -> None:
    denmark = load_country(COUNTRIES / f"{zone.lower()}.toml")

    assert denmark.country == zone
    assert denmark.perimeter == "price_area"
    assert denmark.pv().fleet == load_country(GERMANY).pv().fleet
    assert [generation.last_commissioning_year for generation in denmark.wind_onshore] == [
        2000,
        2012,
        None,
    ]


@pytest.mark.parametrize(
    "path",
    [FRANCE, GERMANY, COUNTRIES / "dk1.toml", COUNTRIES / "dk2.toml"],
    ids=["FR", "DE", "DK1", "DK2"],
)
def test_the_dc_ac_ratio_of_each_country_is_on_the_grid_of_precompute(path: Path) -> None:
    assert load_country(path).pv().dc_ac_ratio in precompute.DC_AC_GRID
