import os
from pathlib import Path

import httpx
import pandas as pd
import pytest

from eu_solar_wind_audit.sources import mastr

pytestmark = pytest.mark.skipif(os.environ.get("RUN_NETWORK_TESTS") != "1", reason="network")


def test_download_page_links_a_recent_export_that_answers_with_a_zip() -> None:
    with httpx.Client(timeout=mastr.TIMEOUT, follow_redirects=True) as client:
        url = mastr.find_export_url(client)
        response = client.head(url)

    # The export is built each night, but some nights are skipped.
    age = pd.Timestamp.now().normalize() - mastr.export_date_of(Path(url.rsplit("/")[-1]))
    assert pd.Timedelta(0) <= age <= pd.Timedelta(days=3)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith(mastr.ZIP_TYPES)
    assert int(response.headers["content-length"]) > 1_000_000_000
