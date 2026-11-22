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
# Published national wind includes offshore farms (about 1.5 GW in 2025), so the onshore model
# is compared with a total that is a few percent larger.
COLUMNS = {"solar": "solar_mw", "wind_onshore": "wind_mw"}


@pytest.fixture(scope="module")
def era5_cache(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("era5")


@pytest.mark.parametrize("technology", ["solar", "wind_onshore"])
@pytest.mark.parametrize("first", [date(2025, 1, 14), date(2025, 6, 10)])
def test_three_days_are_close_to_the_published_national_output(
    era5_cache: Path, technology: physics.Technology, first: date
) -> None:
    last = first + pd.Timedelta(days=2)
    start, end = pd.Timestamp(first), pd.Timestamp(last) + pd.Timedelta(days=1)

    modelled = physics.model_france(technology, start, end, load_country(FRANCE), era5_cache)
    with odre.make_client() as client:
        published = odre.fetch_national_consolidated(client, first, last)
    hourly = grid.hourly_from_period_end_means(published[COLUMNS[technology]])["energy_mwh"]
    both = pd.DataFrame({"model": modelled["energy_mwh"], "published": hourly}).loc[modelled.index]

    assert both.notna().all().all()
    daily = both.resample("D").sum()
    ratio = daily["model"] / daily["published"]
    assert ratio.between(0.5, 2.5).all(), ratio
    assert both["model"].corr(both["published"]) > 0.8
