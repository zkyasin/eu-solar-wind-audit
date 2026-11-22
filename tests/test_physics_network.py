import os
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from eu_solar_wind_audit import grid, physics
from eu_solar_wind_audit.config import load_country
from eu_solar_wind_audit.sources import odre

pytestmark = pytest.mark.skipif(os.environ.get("RUN_NETWORK_TESTS") != "1", reason="network")

FRANCE = Path(__file__).parents[1] / "config" / "countries" / "fr.toml"


@pytest.fixture(scope="module")
def era5_cache(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("era5")


@pytest.mark.parametrize("technology", ["solar", "wind_onshore"])
@pytest.mark.parametrize("first", [date(2025, 1, 14), date(2025, 6, 10)])
def test_three_days_are_close_to_the_published_national_output(
    era5_cache: Path, technology: physics.Technology, first: date
) -> None:
    last = first + pd.Timedelta(days=2)
    start = pd.Timestamp(first, tz="UTC")
    end = pd.Timestamp(last, tz="UTC") + pd.Timedelta(days=1)

    modelled = physics.model_france(technology, start, end, load_country(FRANCE), era5_cache)
    with odre.make_client() as client:
        if technology == "solar":
            published = odre.fetch_national_consolidated(client, first, last)["solar_mw"]
        else:
            # National wind includes offshore farms; the 12 mainland regions split it.
            regional = odre.fetch_regional_consolidated(client, first, last)
            published = regional.groupby(level="time")["wind_onshore_mw"].sum(min_count=12)
    hourly = grid.on_hours(grid.hourly_from_period_end_means(published), start, end)
    both = pd.DataFrame({"model": modelled["energy_mwh"], "published": hourly["energy_mwh"]})

    assert both.notna().all().all()
    daily = both.resample("D").sum()
    ratio = daily["model"] / daily["published"]
    assert ratio.between(0.5, 2.5).all(), ratio
    assert both["model"].corr(both["published"]) > 0.8
