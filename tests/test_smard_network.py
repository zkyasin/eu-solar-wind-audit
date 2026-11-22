import json
import os
from pathlib import Path

import httpx
import pandas as pd
import pytest

from eu_solar_wind_audit.sources import smard

pytestmark = pytest.mark.skipif(os.environ.get("RUN_NETWORK_TESTS") != "1", reason="network")

# Fixture: https://www.smard.de/app/chart_data/254/DE/254_DE_quarterhour_1747000800000.json,
# retrieved 2026-11-22.
FRENCH_WEEK = Path(__file__).parent / "fixtures" / "smard" / "254_DE_quarterhour_1747000800000.json"


def test_published_french_week_still_matches_the_fixture(tmp_path: Path) -> None:
    start = pd.Timestamp(1747000800000, unit="ms", tz="UTC")

    prices = smard.fetch_series(
        *smard.PRICES["FR"], "quarterhour", start, start + pd.Timedelta(days=7), cache_dir=tmp_path
    )

    rows = json.loads(FRENCH_WEEK.read_bytes())["series"]
    assert prices.to_list() == [value for _, value in rows]


def test_french_prices_are_not_served_under_region_fr(tmp_path: Path) -> None:
    start = pd.Timestamp("2025-05-12", tz="UTC")

    with pytest.raises(httpx.HTTPStatusError) as error:
        smard.fetch_series(
            254, "FR", "hour", start, start + pd.Timedelta(hours=1), cache_dir=tmp_path
        )

    assert error.value.response.status_code == 404


def test_offshore_wind_is_published_for_50hertz(tmp_path: Path) -> None:
    start = pd.Timestamp("2025-01-06", tz="UTC")

    offshore = smard.fetch_series(
        smard.WIND_OFFSHORE,
        "50Hertz",
        "hour",
        start,
        start + pd.Timedelta(days=1),
        cache_dir=tmp_path,
    )

    assert len(offshore) == 24
    assert offshore.notna().all()
    assert (offshore >= 0).all()
