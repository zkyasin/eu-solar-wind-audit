import hashlib
import json
import zipfile
from collections.abc import Callable, Hashable
from pathlib import Path
from typing import Any

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


@pytest.fixture(scope="module")
def units() -> pd.DataFrame:
    return mastr.read_units(EXPORT)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("time.sleep", lambda seconds: None)


def unit(units: pd.DataFrame, unit_id: str) -> dict[Hashable, Any]:
    return units.set_index("unit_id").loc[unit_id].to_dict()


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


@pytest.mark.parametrize(
    ("name", "is_solar"),
    [
        ("EinheitenSolar.xml", True),
        ("EinheitenSolar_1.xml", True),
        ("EinheitenSolar_67.xml", True),
        ("AnlagenEegSolar_1.xml", False),
        ("EinheitenSolar_1.xsd", False),
        ("EinheitenStromSpeicher_1.xml", False),
    ],
)
def test_solar_members_are_the_numbered_unit_tables(name: str, is_solar: bool) -> None:
    assert bool(mastr.SOLAR_MEMBER.fullmatch(name)) is is_solar


def test_read_units_counts_each_technology(units: pd.DataFrame) -> None:
    assert units["technology"].value_counts().to_dict() == {
        "wind_onshore": 13,
        "solar": 12,
        "wind_offshore": 12,
    }


def test_read_units_parses_powers_dates_and_location(units: pd.DataFrame) -> None:
    row = unit(units, "SEE940146675093")

    assert row["technology"] == "wind_onshore"
    assert row["status"] == "operating"
    assert row["net_kw"] == 3000.0
    assert row["commissioned"] == pd.Timestamp("2017-09-01")
    assert pd.isna(row["decommissioned"])
    assert row["municipality_key"] == "06636200"
    assert row["district"] == "06636"
    assert row["longitude"] == pytest.approx(9.739374)
    assert row["latitude"] == pytest.approx(51.270068)
    assert row["wind_farm"] == "Windpark Kreuzstein"


def test_solar_gross_power_is_dc_and_net_power_is_ac(units: pd.DataFrame) -> None:
    row = unit(units, "SEE996242407533")

    assert row["gross_kw"] == 87.88
    assert row["net_kw"] == 70.0


def test_decommissioning_and_temporary_shutdown_dates(units: pd.DataFrame) -> None:
    decommissioned = unit(units, "SEE946913378938")
    resumed = unit(units, "SEE913926374813")
    shut_down = unit(units, "SEE955357427270")

    assert decommissioned["status"] == "decommissioned"
    assert decommissioned["decommissioned"] == pd.Timestamp("2021-03-01")
    assert resumed["status"] == "operating"
    assert resumed["shutdown_start"] == pd.Timestamp("2023-03-15")
    assert resumed["shutdown_end"] == pd.Timestamp("2025-08-28")
    assert shut_down["status"] == "temporarily_shut_down"
    assert pd.isna(shut_down["shutdown_end"])


def test_planned_units_have_no_commissioning_date(units: pd.DataFrame) -> None:
    planned = units[units["status"].eq("planned")]

    assert set(planned["unit_id"]) == {"SEE908040981818", "SEE950179793688", "SEE972490644852"}
    assert planned["commissioned"].isna().all()


def test_a_five_digit_municipality_key_is_a_district(units: pd.DataFrame) -> None:
    row = unit(units, "SEE953220423381")

    assert row["municipality_key"] == "05334"
    assert row["district"] == "05334"


def test_offshore_units_have_a_sea_and_cluster_but_no_district(units: pd.DataFrame) -> None:
    offshore = units[units["technology"].eq("wind_offshore")]
    on_land = units[units["technology"].ne("wind_offshore")]

    assert offshore["district"].isna().all()
    assert offshore["sea"].value_counts().to_dict() == {"north": 8, "baltic": 4}
    assert unit(units, "SEE991181426935")["cluster"] == "N-6"
    assert unit(units, "SEE919087229481")["cluster"] == "O-3"
    assert on_land[["sea", "cluster"]].isna().all().all()


def test_a_changed_catalogue_is_refused(tmp_path: Path) -> None:
    changed = tmp_path / EXPORT.name
    with zipfile.ZipFile(EXPORT) as source, zipfile.ZipFile(changed, "w") as target:
        for name in source.namelist():
            content = source.read(name)
            if name == mastr.CATALOGUE_MEMBER:
                content = content.replace(
                    "In Betrieb".encode("utf-16-le"), "Aktiv".encode("utf-16-le")
                )
            target.writestr(name, content)

    with pytest.raises(ValueError, match="catalogue values changed"):
        mastr.read_units(changed)


def test_export_date_comes_from_the_file_name() -> None:
    assert mastr.export_date_of(EXPORT) == pd.Timestamp("2026-11-22")
    with pytest.raises(ValueError, match="not a daily export"):
        mastr.export_date_of(Path("Gesamtdatenexport.zip"))
