"""Country settings of the physics model, one TOML file per country in `config/countries/`."""

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast, get_args

from eu_solar_wind_audit import wind
from eu_solar_wind_audit.pv import PVParameters, System
from eu_solar_wind_audit.sources.era5 import Box

CONFIG_DIR = Path("config/countries")

# price_area: the municipalities of the bidding zone that `country` names.
Perimeter = Literal["metropolitan_without_corsica", "germany", "price_area"]


@dataclass(frozen=True)
class WindGeneration:
    """Turbines commissioned up to the end of `last_commissioning_year`, after the previous
    generation; None for the last generation."""

    turbine: str
    hub_height_m: float
    last_commissioning_year: int | None = None


@dataclass(frozen=True)
class Country:
    country: str
    time_zone: str
    perimeter: Perimeter
    era5_box: Box
    solar: PVParameters
    wind_onshore: tuple[WindGeneration, ...]

    def __post_init__(self) -> None:
        if self.perimeter not in get_args(Perimeter):
            raise ValueError(f"unknown perimeter {self.perimeter!r}")
        years = [generation.last_commissioning_year for generation in self.wind_onshore]
        if not years or years[-1] is not None or None in years[:-1]:
            raise ValueError("only the last wind generation has no last commissioning year")
        bounded = cast(list[int], years[:-1])
        if bounded != sorted(set(bounded)):
            raise ValueError("wind generations must be in increasing commissioning order")
        onshore = wind.turbines().query("technology == 'onshore'").index
        unknown = [g.turbine for g in self.wind_onshore if g.turbine not in onshore]
        if unknown:
            raise ValueError(f"not onshore turbines of wind.turbines(): {unknown}")


def load_country(path: Path) -> Country:
    with path.open("rb") as file:
        raw: dict[str, Any] = tomllib.load(file)
    solar = dict(raw["solar"])
    systems = tuple(System(**system) for system in solar.pop("systems"))
    return Country(
        country=raw["country"],
        time_zone=raw["time_zone"],
        perimeter=raw["perimeter"],
        era5_box=Box(**raw["era5_box"]),
        solar=PVParameters(fleet=systems, **solar),
        wind_onshore=tuple(
            WindGeneration(**generation) for generation in raw["wind_onshore"]["generations"]
        ),
    )
