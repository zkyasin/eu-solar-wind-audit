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


def quarterly(capacity: dict[str, float], *, technology: str = "solar") -> pd.DataFrame:
    """Panel of one department with the last quarter provisional."""
    quarter_ends = pd.to_datetime(list(capacity))
    return pd.DataFrame(
        {
            "department_code": "01",
            "department": "Ain",
            "region": "Auvergne-Rhône-Alpes",
            "quarter_end": quarter_ends,
            "technology": technology,
            "installations": 0,
            "capacity_mw": list(capacity.values()),
            "provisional": quarter_ends == quarter_ends.max(),
        }
    )


def monthly_values(monthly: pd.DataFrame, technology: str = "solar") -> pd.Series:
    selected = monthly[monthly["technology"] == technology]
    return selected.set_index("month_end")["capacity_mw"]


def test_month_ends_are_linear_in_time_between_quarter_ends() -> None:
    panel = quarterly({"2025-12-31": 100.0, "2026-03-31": 190.0})

    values = monthly_values(sdes.monthly_capacity(panel))

    # 31 and 59 of the 90 days between the two quarter ends.
    assert values.to_dict() == pytest.approx(
        {
            pd.Timestamp("2025-12-31"): 100.0,
            pd.Timestamp("2026-01-31"): 131.0,
            pd.Timestamp("2026-02-28"): 159.0,
            pd.Timestamp("2026-03-31"): 190.0,
        }
    )


def test_a_missing_quarter_is_bridged_linearly() -> None:
    panel = quarterly({"2025-06-30": 0.0, "2025-12-31": 184.0})

    values = monthly_values(sdes.monthly_capacity(panel))

    assert values[pd.Timestamp("2025-07-31")] == pytest.approx(31.0)
    assert values[pd.Timestamp("2025-09-30")] == pytest.approx(92.0)


def test_months_after_the_last_final_quarter_are_provisional() -> None:
    panel = quarterly({"2025-12-31": 100.0, "2026-03-31": 190.0})

    monthly = sdes.monthly_capacity(panel).set_index("month_end")

    assert not monthly.loc[pd.Timestamp("2025-12-31"), "provisional"]
    assert monthly.loc[pd.Timestamp("2026-01-31") :, "provisional"].all()


def test_a_panel_without_final_quarters_is_all_provisional() -> None:
    panel = quarterly({"2025-12-31": 100.0, "2026-03-31": 190.0})
    panel["provisional"] = True

    monthly = sdes.monthly_capacity(panel)

    assert monthly["provisional"].all()


def test_each_technology_ends_at_its_own_last_quarter() -> None:
    panel = pd.concat(
        [
            quarterly({"2025-12-31": 10.0, "2026-06-30": 20.0}),
            quarterly({"2025-12-31": 5.0, "2026-03-31": 6.0}, technology="wind_onshore"),
        ]
    )

    monthly = sdes.monthly_capacity(panel)

    assert monthly_values(monthly).index.max() == pd.Timestamp("2026-06-30")
    assert monthly_values(monthly, "wind_onshore").index.max() == pd.Timestamp("2026-03-31")


def test_department_names_are_kept_per_technology() -> None:
    wind = quarterly({"2025-12-31": 5.0, "2026-03-31": 6.0}, technology="wind_onshore")
    wind["department"] = "AIN"
    panel = pd.concat([quarterly({"2025-12-31": 10.0, "2026-03-31": 20.0}), wind])

    monthly = sdes.monthly_capacity(panel)

    assert len(monthly) == 2 * 4
    names = monthly.groupby("technology")["department"].unique()
    assert names.map(list).to_dict() == {"solar": ["Ain"], "wind_onshore": ["AIN"]}


def test_quarter_ends_keep_their_published_values() -> None:
    panel = sdes.read_capacity(WIND)

    monthly = sdes.monthly_capacity(panel)

    keys = ["department_code", "technology", "quarter_end"]
    at_quarter_ends = monthly.rename(columns={"month_end": "quarter_end"}).merge(panel, on=keys)
    assert len(at_quarter_ends) == len(panel)
    assert (at_quarter_ends["capacity_mw_x"] == at_quarter_ends["capacity_mw_y"]).all()
    assert (at_quarter_ends["department_x"] == at_quarter_ends["department_y"]).all()


def test_capacity_after_the_last_quarter_follows_the_last_four_quarters() -> None:
    # 365 MW over the last four quarters, 1 MW a day; the last quarter alone grew 295 MW.
    panel = quarterly(
        {
            "2025-06-30": 100.0,
            "2025-09-30": 130.0,
            "2025-12-31": 150.0,
            "2026-03-31": 170.0,
            "2026-06-30": 465.0,
        }
    )

    monthly = sdes.monthly_capacity(panel, until=pd.Timestamp("2026-08-31"))

    values = monthly_values(monthly)
    assert values[pd.Timestamp("2026-07-31")] == pytest.approx(496.0)
    assert values[pd.Timestamp("2026-08-31")] == pytest.approx(527.0)
    assert values.index.max() == pd.Timestamp("2026-08-31")
    provisional = monthly.set_index("month_end")["provisional"]
    assert not provisional[pd.Timestamp("2026-03-31")]
    assert provisional[pd.Timestamp("2026-04-30") :].all()


def test_extended_capacity_never_goes_below_zero() -> None:
    panel = quarterly({"2025-06-30": 100.0, "2026-06-30": 10.0})

    values = monthly_values(sdes.monthly_capacity(panel, until=pd.Timestamp("2027-06-30")))

    assert values[pd.Timestamp("2026-07-31")] == pytest.approx(10.0 - 90.0 * 31 / 365)
    assert values[pd.Timestamp("2027-06-30")] == 0.0


def test_until_before_the_last_quarter_cuts_the_months() -> None:
    panel = quarterly({"2025-12-31": 100.0, "2026-03-31": 190.0})

    values = monthly_values(sdes.monthly_capacity(panel, until=pd.Timestamp("2026-01-31")))

    assert list(values.index) == [pd.Timestamp("2025-12-31"), pd.Timestamp("2026-01-31")]
