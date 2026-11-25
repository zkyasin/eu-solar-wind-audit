import hashlib
from pathlib import Path

import httpx
import pandas as pd
import pytest

from eu_solar_wind_audit.sources import ens

# The current file names the placement column "Kommune" and the municipality column "Type af
# placering", and writes numbers as text with a decimal comma.
CURRENT_HEADER = [
    "Møllenummer (GSRN)",
    "Parknummer (relation)",
    "Dato for oprindelig nettilslutning",
    "Dato for afmeldning",
    "Kapacitet (kW)",
    "Rotor-diameter (m)",
    "Navhøjde (m)",
    "Fabrikat",
    "Typebetegnelse",
    "Kommune",
    "Type af placering",
    "2023-01",
]
HISTORIC_HEADER = [
    "Møllenummer (GSRN)",
    "Dato for oprindelig nettilslutning",
    "Dato for afmeldning",
    "Kapacitet (kW)",
    "Rotor-diameter (m)",
    "Navhøjde (m)",
    "Fabrikat",
    "Typebetegnelse",
    "Kommune",
    "Type af placering",
    "1995",
]
RANDERS = "570715000000000001"
SLAGELSE = "570715000000000002"
SEA = "570715000000000003"
HOUSEHOLD = "570715000000000004"


def workbook(path: Path, header: list[str], rows: list[list[object]]) -> Path:
    """A sheet laid out as the register: two rows of column groups, then the header."""
    groups = [["Stamdata", *[None] * (len(header) - 1)], [None] * len(header)]
    pd.DataFrame([*groups, header, *rows]).to_excel(path, header=False, index=False)
    return path


def current_row(turbine: str, connected: str | None, **values: object) -> list[object]:
    row = {
        "turbine": turbine,
        "park": None,
        "connected": connected,
        "decommissioned": None,
        "capacity": "2300",
        "rotor": "92,6",
        "hub": "80",
        "make": "SIEMENS",
        "type": "SWT 2.3-93",
        "placement": "LAND",
        "municipality": "730 - Randers",
        "production": "704005,88",
    } | values
    return list(row.values())


def register(tmp_path: Path, current: list[list[object]] | None = None) -> tuple[Path, Path]:
    current_path = workbook(
        tmp_path / "current.xlsx",
        CURRENT_HEADER,
        current
        or [
            current_row(RANDERS, "2010-05-20"),
            current_row(
                SEA, "2003-01-15", capacity="2300", placement="HAV", municipality="0 - Ukendt (0)"
            ),
            current_row(
                HOUSEHOLD,
                "2015-06-15",
                capacity="10",
                rotor="0,1",
                hub="0",
                placement=None,
                municipality="787 - Thisted",
            ),
        ],
    )
    historic_path = workbook(
        tmp_path / "historic.xlsx",
        HISTORIC_HEADER,
        [
            [
                SLAGELSE,
                pd.Timestamp("1995-11-21"),
                pd.Timestamp("2015-03-01"),
                600.0,
                44.0,
                0.1,
                "BONUS",
                "600",
                "330 - Slagelse",
                "LAND",
                1000.0,
            ]
        ],
    )
    return current_path, historic_path


def test_both_layouts_read_into_the_same_columns(tmp_path: Path) -> None:
    turbines = ens.read_turbines(register(tmp_path)).set_index("turbine")

    assert list(turbines.columns) == [
        "connected",
        "decommissioned",
        "capacity_kw",
        "rotor_diameter_m",
        "hub_height_m",
        "municipality",
        "offshore",
    ]
    assert turbines.loc[RANDERS, "municipality"] == 730
    assert turbines.loc[RANDERS, "rotor_diameter_m"] == pytest.approx(92.6)
    assert turbines.loc[SLAGELSE, "municipality"] == 330
    assert turbines.loc[SLAGELSE, "decommissioned"] == pd.Timestamp("2015-03-01")
    assert turbines.loc[SLAGELSE, "capacity_kw"] == 600.0


def test_unknown_sizes_municipalities_and_placements(tmp_path: Path) -> None:
    turbines = ens.read_turbines(register(tmp_path)).set_index("turbine")

    assert turbines[["rotor_diameter_m", "hub_height_m"]].loc[HOUSEHOLD].isna().all()
    assert turbines.loc[[SLAGELSE], "hub_height_m"].isna().all()
    assert turbines.loc[[SEA], "municipality"].isna().all()
    assert turbines["offshore"].to_dict() == {
        SLAGELSE: False,
        SEA: True,
        RANDERS: False,
        HOUSEHOLD: False,
    }


def test_repeated_rows_count_once_and_a_moved_turbine_keeps_both_sites(tmp_path: Path) -> None:
    rows = [
        current_row(RANDERS, "2010-05-20"),
        current_row(RANDERS, "2010-05-20"),
        current_row(HOUSEHOLD, "2015-06-15", decommissioned="2023-04-17", capacity="10"),
        current_row(HOUSEHOLD, "2023-04-20", capacity="10", municipality="787 - Thisted"),
        current_row(SEA, None),
    ]

    turbines = ens.read_turbines(register(tmp_path, rows))

    assert turbines["turbine"].value_counts().to_dict() == {SLAGELSE: 1, RANDERS: 1, HOUSEHOLD: 2}


def test_capacity_counts_from_connection_to_decommissioning(tmp_path: Path) -> None:
    rows = [
        current_row(RANDERS, "2020-03-15", decommissioned="2020-06-01", capacity="2000"),
        current_row(HOUSEHOLD, "2020-04-30", capacity="10", municipality="787 - Thisted"),
        current_row(SEA, "2020-01-10", placement="HAV"),
        current_row("570715000000000005", "2020-01-10", capacity=None),
    ]
    turbines = ens.read_turbines(register(tmp_path, rows))

    panel = ens.monthly_capacity(turbines, pd.Timestamp("2020-02-29"), pd.Timestamp("2020-07-31"))

    assert set(panel["municipality"]) == {330, 730, 787}
    wide = panel.pivot(index="month_end", columns="municipality", values="capacity_mw")
    assert wide[730].to_list() == [0.0, 2.0, 2.0, 2.0, 0.0, 0.0]
    assert wide[787].to_list() == [0.0, 0.0, 0.01, 0.01, 0.01, 0.01]
    # Slagelse's turbine was decommissioned in 2015.
    assert (wide[330] == 0.0).all()


def test_generations_weight_hub_heights_by_capacity(tmp_path: Path) -> None:
    rows = [
        current_row(RANDERS, "1999-05-20", capacity="600", rotor="44", hub="40"),
        current_row(HOUSEHOLD, "1999-06-15", capacity="1800", rotor="0", hub="60"),
        current_row("570715000000000006", "2018-01-10", capacity="3600", rotor="126", hub="90"),
        current_row(SEA, "2018-01-10", capacity="8000", placement="HAV"),
    ]
    turbines = ens.read_turbines(register(tmp_path, rows))

    generations = ens.wind_generations(turbines, pd.Timestamp("2025-12-31"), [2000, None])

    assert generations["capacity_mw"].to_list() == pytest.approx([2.4, 3.6])
    assert generations["hub_height_m"].to_list() == pytest.approx([55.0, 90.0])
    # The 1800 kW turbine has no known rotor: 600 kW over the area swept by 44 m.
    assert generations["specific_power_w_m2"].iloc[0] == pytest.approx(
        600_000 / (3.14159265 * 22**2)
    )


def test_a_file_without_a_placement_column_raises(tmp_path: Path) -> None:
    rows = [current_row(RANDERS, "2010-05-20", placement="land")]

    with pytest.raises(ValueError, match="expected one column"):
        ens.read_turbines(register(tmp_path, rows))


def test_download_is_checked_then_cached(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    current, historic = register(source)
    bodies = {ens.CURRENT_URL: current.read_bytes(), ens.HISTORIC_URL: historic.read_bytes()}
    monkeypatch.setattr(ens, "CURRENT_SHA256", hashlib.sha256(bodies[ens.CURRENT_URL]).hexdigest())
    monkeypatch.setattr(
        ens, "HISTORIC_SHA256", hashlib.sha256(bodies[ens.HISTORIC_URL]).hexdigest()
    )
    requests: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(str(request.url))
        return httpx.Response(200, content=bodies[str(request.url)])

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        first = ens.download_registers(client, tmp_path / "raw")
        second = ens.download_registers(client, tmp_path / "raw")

    assert requests == [ens.CURRENT_URL, ens.HISTORIC_URL]
    assert first == second
    assert [path.name for path in first] == [
        "vinddata_2026-10.xlsx",
        "historiske_vinddata_2022.xlsx",
    ]
