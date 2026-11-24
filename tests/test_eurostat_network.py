import os

import pytest

from eu_solar_wind_audit.sources import eurostat

pytestmark = pytest.mark.skipif(os.environ.get("RUN_NETWORK_TESTS") != "1", reason="network")


@pytest.fixture(scope="module")
def release(tmp_path_factory: pytest.TempPathFactory) -> eurostat.Release:
    return eurostat.fetch_gross_production(cache_dir=tmp_path_factory.mktemp("eurostat"))


def test_every_country_and_technology_has_2019_to_2024(release: eurostat.Release) -> None:
    production = release.production
    years = production.groupby(["country", "technology"])["year"].agg(set)

    assert len(years) == len(eurostat.COUNTRIES) * len(eurostat.FUELS)
    assert all(set(range(2019, 2025)) <= found for found in years)


def test_german_solar_in_2024_is_between_60_and_90_twh(release: eurostat.Release) -> None:
    production = release.production
    (value,) = production.loc[
        (production["country"] == "DE")
        & (production["technology"] == "solar")
        & (production["year"] == 2024),
        "gwh",
    ]

    assert 60_000 < value < 90_000
