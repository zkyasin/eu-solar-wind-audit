from pathlib import Path

import httpx
import pytest

from eu_solar_wind_audit.sources import sdes

SOLAR_URL = "https://www.statistiques.developpement-durable.gouv.fr/media/9714/download?inline"
WIND_URL = "https://www.statistiques.developpement-durable.gouv.fr/media/9715/download?inline"

XLSX_HEADERS = {"content-type": sdes.XLSX}
WORKBOOK = b"PK\x03\x04 workbook bytes"


def test_download_writes_the_workbook(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, content=WORKBOOK, headers=XLSX_HEADERS)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        path = sdes.download_dashboard(client, sdes.SOLAR_2026_Q2, tmp_path / "sdes")

    assert str(requests[0].url) == SOLAR_URL
    assert path.read_bytes() == WORKBOOK
    assert list((tmp_path / "sdes").iterdir()) == [path]


def test_download_reuses_the_cached_workbook(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, content=WORKBOOK, headers=XLSX_HEADERS)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        first = sdes.download_dashboard(client, sdes.WIND_2026_Q2, tmp_path)
        second = sdes.download_dashboard(client, sdes.WIND_2026_Q2, tmp_path)

    assert first == second
    assert len(requests) == 1
    assert str(requests[0].url) == WIND_URL


def test_download_rejects_an_html_page(tmp_path: Path) -> None:
    page = httpx.Response(200, text="<html></html>", headers={"content-type": "text/html"})
    transport = httpx.MockTransport(lambda request: page)

    with httpx.Client(transport=transport) as client, pytest.raises(ValueError, match="xlsx"):
        sdes.download_dashboard(client, sdes.SOLAR_2026_Q2, tmp_path)

    assert list(tmp_path.iterdir()) == []


def test_failed_download_leaves_no_file(tmp_path: Path) -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(404))

    with httpx.Client(transport=transport) as client, pytest.raises(httpx.HTTPStatusError):
        sdes.download_dashboard(client, sdes.SOLAR_2026_Q2, tmp_path)

    assert list(tmp_path.iterdir()) == []
