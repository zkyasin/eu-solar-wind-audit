import hashlib
import json
import zipfile
from collections.abc import Callable, Hashable
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd
import pytest

from eu_solar_wind_audit.sources import mastr

# Fixtures: the page https://www.marktstammdatenregister.de/MaStR/Datendownload, and a zip cut from
# https://download.marktstammdatenregister.de/Gesamtdatenexport_20261122_26.2.zip, both retrieved
# 2026-11-22. The zip keeps the original UTF-16 LE records byte for byte: 25 units of
# EinheitenWind.xml plus SEE973525112162 and SEE901254459012 (former district keys), 6 + 6 units of
# EinheitenSolar_30.xml split into EinheitenSolar_1.xml and EinheitenSolar_2.xml, and the
# Katalogwerte.xml values of categories 4, 39, 87, 107 and 108.
FIXTURES = Path(__file__).parent / "fixtures" / "mastr"
EXPORT = FIXTURES / "Gesamtdatenexport_20261122_26.2.zip"
PAGE = FIXTURES / "Datendownload_2026-11-22.html"
EXPORT_URL = "https://download.marktstammdatenregister.de/Gesamtdatenexport_20261122_26.2.zip"
ZIP_TYPE = "application/x-zip-compressed"

Handler = Callable[[httpx.Request], httpx.Response]


@pytest.fixture(scope="module")
def units() -> pd.DataFrame:
    return mastr.read_units(EXPORT)


@pytest.fixture(scope="module")
def panel(units: pd.DataFrame) -> pd.DataFrame:
    return mastr.monthly_capacity(units, pd.Timestamp("2000-01-31"), pd.Timestamp("2026-10-31"))


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("time.sleep", lambda seconds: None)


def unit(units: pd.DataFrame, unit_id: str) -> dict[Hashable, Any]:
    return units.set_index("unit_id").loc[unit_id].to_dict()


def capacity(panel: pd.DataFrame, month_end: str, district: str | None, technology: str) -> float:
    rows = panel[
        panel["month_end"].eq(pd.Timestamp(month_end))
        & (panel["district"].isna() if district is None else panel["district"].eq(district))
        & panel["technology"].eq(technology)
    ]
    assert len(rows) == 1
    return float(rows["capacity_mw_ac"].iloc[0])


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
        "wind_onshore": 15,
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


@pytest.mark.parametrize(
    ("unit_id", "municipality_key", "district"),
    [("SEE973525112162", "03152002", "03159"), ("SEE901254459012", "16056000", "16063")],
)
def test_former_district_keys_give_the_current_district(
    units: pd.DataFrame, unit_id: str, municipality_key: str, district: str
) -> None:
    row = unit(units, unit_id)

    assert row["municipality_key"] == municipality_key
    assert row["district"] == district


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


@pytest.mark.parametrize(
    ("month_end", "expected_mw"),
    [("2019-11-30", 0.0), ("2019-12-31", 8.4), ("2026-10-31", 8.4)],
)
def test_units_count_from_the_end_of_their_commissioning_month(
    panel: pd.DataFrame, month_end: str, expected_mw: float
) -> None:
    # Two 4.2 MW turbines commissioned on 2019-12-23 and 2019-12-30.
    assert capacity(panel, month_end, "07231", "wind_onshore") == expected_mw


def test_a_unit_commissioned_on_a_month_end_counts_at_that_month_end(panel: pd.DataFrame) -> None:
    assert capacity(panel, "2017-07-31", "05334", "wind_onshore") == 0.0
    assert capacity(panel, "2017-08-31", "05334", "wind_onshore") == 3.2


@pytest.mark.parametrize(
    ("district", "last_counted", "first_dropped"),
    [
        # Decommissioned on 2021-03-01 and on 2023-06-30.
        ("01055", "2021-02-28", "2021-03-31"),
        ("03353", "2023-05-31", "2023-06-30"),
    ],
)
def test_units_stop_counting_at_their_decommissioning(
    panel: pd.DataFrame, district: str, last_counted: str, first_dropped: str
) -> None:
    assert capacity(panel, last_counted, district, "wind_onshore") > 0
    assert capacity(panel, first_dropped, district, "wind_onshore") == 0.0


def test_a_unit_decommissioned_in_its_commissioning_month_never_counts(panel: pd.DataFrame) -> None:
    assert panel[panel["district"].eq("08135")]["capacity_mw_ac"].eq(0).all()


def test_a_unit_decommissioned_before_its_commissioning_never_counts(units: pd.DataFrame) -> None:
    changed = units.copy()
    # A 3.45 MW turbine commissioned on 2019-03-29, alone in its district.
    changed.loc[changed["unit_id"].eq("SEE973525112162"), "decommissioned"] = pd.Timestamp(
        "2019-01-15"
    )

    panel = mastr.monthly_capacity(changed, pd.Timestamp("2018-12-31"), pd.Timestamp("2026-10-31"))

    assert panel[panel["district"].eq("03159")]["capacity_mw_ac"].eq(0).all()


def test_temporarily_shut_down_units_keep_counting(panel: pd.DataFrame) -> None:
    assert capacity(panel, "2026-10-31", "07331", "wind_onshore") == 0.6
    assert capacity(panel, "2024-06-30", "13072", "wind_onshore") == 3.37


def test_planned_and_future_units_are_left_out(panel: pd.DataFrame) -> None:
    assert "08326" not in set(panel["district"])
    assert "05112" not in set(panel["district"])
    # Commissioned on 2026-11-04, after the last month of the panel.
    assert capacity(panel, "2026-10-31", "05762", "wind_onshore") == 0.0


def test_a_planned_unit_never_counts_even_with_a_commissioning_date(units: pd.DataFrame) -> None:
    changed = units.copy()
    # A planned 5.7 MW turbine, alone in its district.
    changed.loc[changed["unit_id"].eq("SEE950179793688"), "commissioned"] = pd.Timestamp(
        "2020-05-01"
    )

    panel = mastr.monthly_capacity(changed, pd.Timestamp("2026-10-31"), pd.Timestamp("2026-10-31"))

    assert "08326" not in set(panel["district"])


def test_a_decommissioned_unit_without_a_date_never_counts(units: pd.DataFrame) -> None:
    changed = units.copy()
    changed.loc[changed["unit_id"].eq("SEE946913378938"), "decommissioned"] = pd.NaT

    panel = mastr.monthly_capacity(changed, pd.Timestamp("2010-01-31"), pd.Timestamp("2026-10-31"))

    assert "01055" not in set(panel["district"])


def test_solar_capacity_is_given_in_ac_and_dc(panel: pd.DataFrame) -> None:
    # Three units commissioned in December 2022: 7.2, 25 and 5 kW AC; 7.2, 25 and 6 kWp DC.
    rows = panel[panel["district"].eq("03241") & panel["month_end"].eq(pd.Timestamp("2022-12-31"))]

    assert rows[["capacity_mw_ac", "capacity_mw_dc"]].to_numpy().tolist() == [[0.0372, 0.0382]]


def test_wind_has_no_dc_capacity(panel: pd.DataFrame) -> None:
    assert panel.loc[panel["technology"].ne("solar"), "capacity_mw_dc"].isna().all()
    assert panel.loc[panel["technology"].eq("solar"), "capacity_mw_dc"].notna().all()


def test_offshore_wind_is_national_in_the_panel(panel: pd.DataFrame) -> None:
    offshore = panel[panel["technology"].eq("wind_offshore")]

    assert offshore["district"].isna().all()
    # Five alpha ventus turbines of 5 MW, from August 2009 to April 2010.
    assert capacity(panel, "2009-07-31", None, "wind_offshore") == 0.0
    assert capacity(panel, "2009-08-31", None, "wind_offshore") == 10.0
    assert capacity(panel, "2026-10-31", None, "wind_offshore") == 49.61


def test_panel_covers_every_month_from_start_to_end(units: pd.DataFrame) -> None:
    panel = mastr.monthly_capacity(units, pd.Timestamp("2019-01-31"), pd.Timestamp("2019-12-31"))
    months = panel["month_end"].drop_duplicates()

    assert months.tolist() == list(pd.date_range("2019-01-31", "2019-12-31", freq="ME"))
    # Units commissioned before the start are counted.
    assert capacity(panel, "2019-01-31", "16051", "solar") == 0.0
    assert capacity(panel, "2019-01-31", "05374", "solar") == 0.07


def test_national_totals_match_the_units_in_operation(
    units: pd.DataFrame, panel: pd.DataFrame
) -> None:
    month_end = pd.Timestamp("2026-10-31")
    active = units[
        units["commissioned"].le(month_end)
        & ~units["decommissioned"].le(month_end)
        & units["status"].ne("planned")
    ]
    expected = active.groupby("technology")["net_kw"].sum() / 1000
    totals = panel[panel["month_end"].eq(month_end)].groupby("technology")["capacity_mw_ac"].sum()

    pd.testing.assert_series_equal(totals, expected, check_names=False)


def test_offshore_wind_is_grouped_by_cluster_and_era5_cell(units: pd.DataFrame) -> None:
    table = mastr.offshore_wind(units, pd.Timestamp("2026-10-31")).set_index("wind_farms")

    alpha_ventus = unit(table.rename_axis("unit_id").reset_index(), "alpha ventus")
    assert (alpha_ventus["sea"], alpha_ventus["cluster"]) == ("north", "N-2")
    assert (alpha_ventus["cell_latitude"], alpha_ventus["cell_longitude"]) == (54.0, 6.5)
    assert alpha_ventus["units"] == 5
    assert alpha_ventus["capacity_mw"] == 25.0
    assert alpha_ventus["first_commissioning_month"] == pd.Timestamp("2009-08-31")
    assert alpha_ventus["last_commissioning_month"] == pd.Timestamp("2010-04-30")
    assert table.loc["Baltic 2", "cell_latitude"] == 55.0
    assert table.loc["Baltic 2", "cell_longitude"] == 13.0
    # Planned units are left out.
    assert "NC 2" not in table.index
    assert table["capacity_mw"].sum() == pytest.approx(49.61)


def test_offshore_wind_leaves_out_units_commissioned_after_the_date(units: pd.DataFrame) -> None:
    table = mastr.offshore_wind(units, pd.Timestamp("2009-12-31"))

    assert table[["wind_farms", "units", "capacity_mw"]].to_numpy().tolist() == [
        ["alpha ventus", 2, 10.0]
    ]


def test_offshore_wind_counts_a_unit_from_its_commissioning_date(units: pd.DataFrame) -> None:
    # The second alpha ventus turbine was commissioned on 2009-08-21.
    table = mastr.offshore_wind(units, pd.Timestamp("2009-08-21"))

    assert table["units"].tolist() == [2]


def test_offshore_wind_leaves_out_decommissioned_units(units: pd.DataFrame) -> None:
    changed = units.copy()
    veja_mate = changed["unit_id"].eq("SEE991181426935")
    changed.loc[veja_mate, "status"] = "decommissioned"
    changed.loc[veja_mate, "decommissioned"] = pd.Timestamp("2026-10-31")

    table = mastr.offshore_wind(changed, pd.Timestamp("2026-10-31"))

    assert "OWP Veja Mate" not in set(table["wind_farms"])
    assert table["capacity_mw"].sum() == pytest.approx(49.61 - 6.3)


def test_the_only_implausible_unit_of_the_fixture_is_dated_1900(units: pd.DataFrame) -> None:
    report = mastr.implausible_units(units, pd.Timestamp("2026-11-22"))

    # A 0.6 kW solar unit commissioned on 1900-04-22, as registered.
    assert report[["unit_id", "problem"]].to_numpy().tolist() == [
        ["SEE970825764190", "commissioning date before 1950"]
    ]


@pytest.mark.parametrize(
    ("column", "value", "problem"),
    [
        ("net_kw", -5.0, "power not positive"),
        ("gross_kw", np.nan, "power not positive"),
        ("net_kw", 4000.0, "net power above gross power"),
        ("gross_kw", 250_000.0, "power above the plausible maximum"),
        ("commissioned", pd.Timestamp("2027-01-01"), "commissioning date in the future"),
        ("commissioned", pd.Timestamp("1900-01-01"), "commissioning date before 1950"),
        ("decommissioned", pd.Timestamp("2027-01-01"), "decommissioning date in the future"),
        ("decommissioned", pd.Timestamp("2015-01-01"), "decommissioned before commissioning"),
        ("municipality_key", "0533", "municipality key malformed"),
        ("district", "17001", "district outside the German state codes"),
        ("municipality_key", None, "on land without a municipality key"),
    ],
)
def test_implausible_units_are_reported(
    units: pd.DataFrame, column: str, value: Any, problem: str
) -> None:
    changed = units.copy()
    # A 749 kWp solar unit commissioned on 2020-09-30.
    changed.loc[changed["unit_id"].eq("SEE983469276431"), column] = value

    report = mastr.implausible_units(changed, pd.Timestamp("2026-11-22"))

    changed_unit = report[report["unit_id"].eq("SEE983469276431")]
    assert changed_unit["problem"].tolist() == [problem]


def test_extract_capacity_downloads_and_writes_the_tables_with_their_provenance(
    tmp_path: Path,
) -> None:
    page = PAGE.read_bytes()
    content = EXPORT.read_bytes()

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == mastr.DOWNLOAD_PAGE:
            return httpx.Response(200, content=page)
        if str(request.url) == EXPORT_URL:
            return httpx.Response(200, headers={"content-type": ZIP_TYPE}, content=content)
        return httpx.Response(404)

    with serving(handler) as client:
        path = mastr.extract_capacity(
            cache_dir=tmp_path / "raw", output_dir=tmp_path / "out", client=client
        )

    panel = pd.read_parquet(path)
    assert panel["month_end"].min() == pd.Timestamp("2019-01-31")
    # The export is dated 2026-11-22: November is not complete.
    assert panel["month_end"].max() == pd.Timestamp("2026-10-31")
    assert len(pd.read_parquet(tmp_path / "out" / "mastr_units.parquet")) == 39
    assert len(pd.read_parquet(tmp_path / "out" / "mastr_offshore_wind.parquet")) == 5
    metadata = json.loads(path.with_suffix(".json").read_text())
    assert metadata["file"] == EXPORT.name
    assert metadata["sha256"] == hashlib.sha256(content).hexdigest()
    assert metadata["export_date"] == "2026-11-22"
    assert {"retrieved_at", "extracted_at"} <= metadata.keys()


def test_write_capacity_reads_an_export_downloaded_by_hand(tmp_path: Path) -> None:
    export = tmp_path / EXPORT.name
    export.write_bytes(EXPORT.read_bytes())

    path = mastr.write_capacity(export, tmp_path / "out")

    metadata = json.loads(path.with_suffix(".json").read_text())
    assert metadata["file"] == EXPORT.name
    assert metadata["sha256"] == hashlib.sha256(export.read_bytes()).hexdigest()
    assert metadata["url"] is None
    assert metadata["retrieved_at"] is None
    assert metadata["export_date"] == "2026-11-22"


def test_export_date_comes_from_the_file_name() -> None:
    assert mastr.export_date_of(EXPORT) == pd.Timestamp("2026-11-22")
    with pytest.raises(ValueError, match="not a daily export"):
        mastr.export_date_of(Path("Gesamtdatenexport.zip"))
