from pathlib import Path

import httpx
import pandas as pd
import pytest

from eu_solar_wind_audit.sources import sdes

SOLAR_URL = "https://www.statistiques.developpement-durable.gouv.fr/media/9714/download?inline"
WIND_URL = "https://www.statistiques.developpement-durable.gouv.fr/media/9715/download?inline"
FIXTURES = Path(__file__).parent / "fixtures" / "sdes"
# SOLAR_URL on 2026-11-22, sheet "solaire PV", rows 1-4, 51-55, 63, 93-98 and 112-125 kept.
SOLAR = FIXTURES / "solaire-pv-2026-t2.xlsx"
# WIND_URL on 2026-11-22, sheets "éolien_terrestre" and "éolien_offshore", same rows as SOLAR.
WIND = FIXTURES / "eolien-2026-t2.xlsx"

BRETAGNE = {"22", "29", "35", "56"}
NORMANDIE = {"14", "27", "50", "61", "76"}
OVERSEAS = {"971", "972", "973", "974", "976"}

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


def at(panel: pd.DataFrame, code: str, technology: str, quarter_end: str) -> pd.Series:
    selected = panel[
        (panel["department_code"] == code)
        & (panel["technology"] == technology)
        & (panel["quarter_end"] == pd.Timestamp(quarter_end))
    ]
    assert len(selected) == 1
    return selected.iloc[0]


def test_solar_panel_has_one_row_per_department_and_quarter() -> None:
    panel = sdes.read_capacity(SOLAR)

    assert set(panel["department_code"]) == BRETAGNE | NORMANDIE | {"20"} | OVERSEAS
    assert set(panel["technology"]) == {"solar"}
    assert panel["quarter_end"].min() == pd.Timestamp("2005-03-31")
    assert panel["quarter_end"].max() == pd.Timestamp("2026-06-30")
    assert len(panel) == 15 * 86


def test_solar_capacity_is_read_in_mw_as_published() -> None:
    panel = sdes.read_capacity(SOLAR)

    row = at(panel, "22", "solar", "2026-06-30")
    assert row["department"] == "Côtes-d'Armor"
    assert row["region"] == "Bretagne"
    assert row["installations"] == 12856
    assert row["capacity_mw"] == pytest.approx(246.72976)
    assert at(panel, "76", "solar", "2024-12-31")["region"] == "Normandie"


def test_single_row_regions_are_their_own_department() -> None:
    panel = sdes.read_capacity(SOLAR)

    corsica = at(panel, "20", "solar", "2026-06-30")
    assert (corsica["department"], corsica["region"]) == ("Corse", "Corse")
    assert corsica["capacity_mw"] == pytest.approx(284.63908)
    assert at(panel, "974", "solar", "2026-06-30")["region"] == "La Réunion"


def test_only_the_last_quarter_is_provisional() -> None:
    panel = sdes.read_capacity(SOLAR)

    provisional = panel.loc[panel["provisional"], "quarter_end"]
    assert set(provisional) == {pd.Timestamp("2026-06-30")}
    assert len(provisional) == 15


def test_a_sheet_in_another_unit_is_rejected(tmp_path: Path) -> None:
    raw = pd.read_excel(SOLAR, sheet_name="solaire PV", header=None)
    raw.iat[1, 0] = "Unité : puissance en kW"
    path = tmp_path / "kw.xlsx"
    raw.to_excel(path, sheet_name="solaire PV", header=False, index=False)

    with pytest.raises(ValueError, match="kW"):
        sdes.read_capacity(path)


def test_wind_panel_separates_onshore_and_offshore() -> None:
    panel = sdes.read_capacity(WIND)

    assert set(panel["technology"]) == {"wind_onshore", "wind_offshore"}
    assert at(panel, "22", "wind_onshore", "2026-06-30")["capacity_mw"] == pytest.approx(542.8106)
    assert at(panel, "22", "wind_offshore", "2026-06-30")["capacity_mw"] == pytest.approx(496)
    assert at(panel, "76", "wind_offshore", "2026-06-30")["installations"] == 3
    assert at(panel, "76", "wind_offshore", "2026-06-30")["capacity_mw"] == pytest.approx(745)


def test_totals_are_read_per_technology() -> None:
    solar = sdes.read_totals(SOLAR)
    wind = sdes.read_totals(WIND)

    assert set(solar["total"]) == {"Total métropole", "Total DOM", "Total France"}
    assert set(wind["total"]) == {"Total métropole", "Total DROM", "Total France"}
    metropole = solar[
        (solar["total"] == "Total métropole") & (solar["quarter_end"] == pd.Timestamp("2024-12-31"))
    ]
    assert metropole["capacity_mw"].item() == pytest.approx(24364.617472)
    offshore = wind[
        (wind["total"] == "Total métropole")
        & (wind["technology"] == "wind_offshore")
        & (wind["quarter_end"] == pd.Timestamp("2026-06-30"))
    ]
    assert offshore["capacity_mw"].item() == pytest.approx(2316.2)


def test_metropolitan_perimeter_drops_corsica_and_overseas() -> None:
    panel = sdes.read_capacity(SOLAR)

    mainland = sdes.metropolitan_without_corsica(panel)

    assert set(mainland["department_code"]) == BRETAGNE | NORMANDIE
    june = mainland[mainland["quarter_end"] == pd.Timestamp("2026-06-30")]
    # Sum of the Bretagne and Normandie region rows of the workbook.
    assert june["capacity_mw"].sum() == pytest.approx(1171.964331 + 717.98565)
