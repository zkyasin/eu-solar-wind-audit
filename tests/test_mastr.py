import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import httpx
import pandas as pd
import pytest

from eu_solar_wind_audit.sources import mastr

# Fixtures: the page https://www.marktstammdatenregister.de/MaStR/Datendownload, and a zip cut from
# https://download.marktstammdatenregister.de/Gesamtdatenexport_20261122_26.2.zip, both retrieved
# 2026-11-22. The zip keeps the original UTF-16 LE records byte for byte: 25 units of
# EinheitenWind.xml, 6 + 6 units of EinheitenSolar_30.xml split into EinheitenSolar_1.xml and
# EinheitenSolar_2.xml, and the Katalogwerte.xml values of categories 4, 39, 87, 107 and 108.
FIXTURES = Path(__file__).parent / "fixtures" / "mastr"
EXPORT = FIXTURES / "Gesamtdatenexport_20261122_26.2.zip"
PAGE = FIXTURES / "Datendownload_2026-11-22.html"
EXPORT_URL = "https://download.marktstammdatenregister.de/Gesamtdatenexport_20261122_26.2.zip"
ZIP_TYPE = "application/x-zip-compressed"

Handler = Callable[[httpx.Request], httpx.Response]


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("time.sleep", lambda seconds: None)


def serving(handler: Handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_find_export_url_takes_the_daily_export_not_the_quarterly_ones() -> None:
    page = PAGE.read_bytes()
    with serving(lambda request: httpx.Response(200, content=page)) as client:
        assert mastr.find_export_url(client) == EXPORT_URL


def test_find_export_url_fails_without_a_daily_export_link() -> None:
    page = PAGE.read_text().replace(EXPORT_URL, "")
    with (
        serving(lambda request: httpx.Response(200, text=page)) as client,
        pytest.raises(ValueError, match="expected one daily export link"),
    ):
        mastr.find_export_url(client)


def range_server(content: bytes, requests: list[httpx.Request], honour_range: bool) -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        headers = {"content-type": ZIP_TYPE}
        requested = request.headers.get("range")
        if requested and honour_range:
            offset = int(requested.removeprefix("bytes=").removesuffix("-"))
            return httpx.Response(206, headers=headers, content=content[offset:])
        return httpx.Response(200, headers=headers, content=content)

    return handler


def test_download_resumes_a_partial_file_and_records_its_checksum(tmp_path: Path) -> None:
    content = EXPORT.read_bytes()
    (tmp_path / f"{EXPORT.name}.part").write_bytes(content[:5000])
    requests: list[httpx.Request] = []

    with serving(range_server(content, requests, honour_range=True)) as client:
        path = mastr.download_export(client, EXPORT_URL, tmp_path)

    assert [request.headers.get("range") for request in requests] == ["bytes=5000-"]
    assert path == tmp_path / EXPORT.name
    assert path.read_bytes() == content
    assert not (tmp_path / f"{EXPORT.name}.part").exists()
    metadata = json.loads((tmp_path / f"{EXPORT.name}.json").read_text())
    assert metadata["url"] == EXPORT_URL
    assert metadata["size"] == len(content)
    assert metadata["sha256"] == hashlib.sha256(content).hexdigest()


def test_download_starts_over_when_the_server_ignores_the_range(tmp_path: Path) -> None:
    content = EXPORT.read_bytes()
    (tmp_path / f"{EXPORT.name}.part").write_bytes(b"stale")
    requests: list[httpx.Request] = []

    with serving(range_server(content, requests, honour_range=False)) as client:
        path = mastr.download_export(client, EXPORT_URL, tmp_path)

    assert path.read_bytes() == content


def test_download_is_cached(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []
    with serving(range_server(EXPORT.read_bytes(), requests, honour_range=True)) as client:
        first = mastr.download_export(client, EXPORT_URL, tmp_path)
        second = mastr.download_export(client, EXPORT_URL, tmp_path)

    assert first == second
    assert len(requests) == 1


def test_download_retries_a_dropped_connection(tmp_path: Path) -> None:
    content = EXPORT.read_bytes()
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if len(calls) == 1:
            raise httpx.ReadError("connection reset")
        return httpx.Response(200, headers={"content-type": ZIP_TYPE}, content=content)

    with serving(handler) as client:
        path = mastr.download_export(client, EXPORT_URL, tmp_path)

    assert len(calls) == 2
    assert path.read_bytes() == content


def test_download_never_caches_an_html_page(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/html"}, content=b"<html/>")

    with serving(handler) as client, pytest.raises(ValueError, match="not a zip file"):
        mastr.download_export(client, EXPORT_URL, tmp_path)

    assert not (tmp_path / EXPORT.name).exists()


def test_export_date_comes_from_the_file_name() -> None:
    assert mastr.export_date_of(EXPORT) == pd.Timestamp("2026-11-22")
    with pytest.raises(ValueError, match="not a daily export"):
        mastr.export_date_of(Path("Gesamtdatenexport.zip"))
