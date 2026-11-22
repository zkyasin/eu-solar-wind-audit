import os
from pathlib import Path

import pytest

from eu_solar_wind_audit.sources import emodnet

pytestmark = pytest.mark.skipif(os.environ.get("RUN_NETWORK_TESTS") != "1", reason="network")

# Longitude and latitude bounds of the French and German sea areas.
BOXES = {
    "FR": ((-6.0, 10.0), (41.0, 51.5)),
    "DE": ((3.0, 15.0), (53.3, 55.7)),
}


@pytest.mark.parametrize("country", sorted(BOXES))
def test_operating_farms_lie_in_the_national_sea_area(country: str, tmp_path: Path) -> None:
    farms = emodnet.load_offshore_farms([country], cache_dir=tmp_path)

    (west, east), (south, north) = BOXES[country]
    assert len(farms) >= 4
    assert (farms["country"] == country).all()
    assert (farms["capacity_mw"] > 0).all()
    assert farms["longitude"].between(west, east).all()
    assert farms["latitude"].between(south, north).all()


def test_every_status_of_the_five_countries_passes_the_schema(tmp_path: Path) -> None:
    farms = emodnet.load_offshore_farms(
        list(emodnet.COUNTRIES), tuple(emodnet.STATUSES.values()), cache_dir=tmp_path
    )

    assert set(farms["country"]) == set(emodnet.COUNTRIES)
