import os

import httpx
import pytest

from eu_solar_wind_audit.sources import mastr

pytestmark = pytest.mark.skipif(os.environ.get("RUN_NETWORK_TESTS") != "1", reason="network")


def test_pinned_export_is_still_served_as_a_zip() -> None:
    with httpx.Client(timeout=mastr.TIMEOUT, follow_redirects=True) as client:
        response = client.head(mastr.EXPORT_URL)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith(mastr.ZIP_TYPES)
    # Size of the file whose checksum is pinned.
    assert int(response.headers["content-length"]) == 3_218_297_640
