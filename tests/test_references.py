import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from eu_solar_wind_audit.references import (
    DOCUMENTATION_FILE,
    RTE_FILES,
    SDES_FILE,
    SDES_SHEET,
    load_documentation,
    load_hirth_curtailment,
    load_rte_modulation,
    load_sdes_self_consumption,
    verify_manifest,
)

RTE = Path(__file__).parents[1] / "references" / "rte"
HIRTH = Path(__file__).parents[1] / "references" / "hirth"
SDES = Path(__file__).parents[1] / "references" / "sdes"
DOCUMENTATION = Path(__file__).parents[1] / "references" / "documentation"
# Positions in the SDES sheet, read without a header: the row of technology labels, the row of
# 2020, and the photovoltaic column.
SDES_LABELS_ROW = 20
SDES_2020_ROW = 23
SDES_PHOTOVOLTAIC_COLUMN = 3
SOLAR = "RTE_PAD_solaire_modulation.csv"
WIND = "RTE_PAD_eolien_modulation.csv"
# Working-day row of April 2025 in the solar file; the value appears nowhere else.
SOLAR_LINE = b'2025-04;"Volume de modulation de la production solaire";170,6127391286\r\n'


@pytest.fixture
def rte_copy(tmp_path: Path) -> Path:
    for path in RTE.iterdir():
        shutil.copy(path, tmp_path / path.name)
    return tmp_path


def replace_once(path: Path, old: bytes, new: bytes) -> None:
    data = path.read_bytes()
    assert data.count(old) == 1
    path.write_bytes(data.replace(old, new))


def test_committed_files_match_the_manifest() -> None:
    verify_manifest(RTE)


def test_manifest_lists_every_file_the_loader_reads() -> None:
    manifest = json.loads((RTE / "manifest.json").read_text(encoding="utf-8"))

    assert set(manifest["files"]) == {name for name, _ in RTE_FILES.values()}


def test_load_gives_one_row_per_month_technology_and_block() -> None:
    frame = load_rte_modulation(RTE)

    assert list(frame.columns) == ["month", "technology", "block", "gwh"]
    assert frame["month"].dtype == pd.PeriodDtype("M")
    assert len(frame) == 2 * 3 * 82
    assert not frame.duplicated(["month", "technology", "block"]).any()
    assert frame["month"].min() == pd.Period("2020-01", "M")
    assert frame["month"].max() == pd.Period("2026-10", "M")
    assert set(frame["technology"]) == {"solar", "wind"}
    assert set(frame["block"]) == {"total", "working_days", "non_working_days"}


def test_values_keep_their_decimal_comma_digits() -> None:
    frame = load_rte_modulation(RTE).astype({"month": str})
    gwh = frame.set_index(["technology", "block", "month"])["gwh"]

    assert gwh.loc[("solar", "total", "2020-03")] == 1.6957262375795
    assert gwh.loc[("solar", "working_days", "2025-04")] == 170.6127391286
    assert gwh.loc[("wind", "total", "2026-10")] == 3.1710732693913


def test_total_is_working_plus_non_working_days() -> None:
    frame = load_rte_modulation(RTE)
    wide = frame.pivot_table(index=["technology", "month"], columns="block", values="gwh")

    gap = wide["total"] - wide["working_days"] - wide["non_working_days"]
    assert gap.abs().max() <= 1e-9
    assert wide["working_days"].sum() > 0
    assert wide["non_working_days"].sum() > 0


@pytest.mark.parametrize(
    ("technology", "year", "gwh"),
    [("solar", 2024, 639), ("solar", 2025, 1461), ("wind", 2024, 895), ("wind", 2025, 1197)],
)
def test_annual_totals_match_the_published_figures(technology: str, year: int, gwh: int) -> None:
    frame = load_rte_modulation(RTE)
    selected = frame[
        (frame["technology"] == technology)
        & (frame["block"] == "total")
        & (frame["month"].dt.year == year)
    ]

    assert len(selected) == 12
    assert round(selected["gwh"].sum()) == gwh


def test_a_changed_block_value_breaks_the_total(rte_copy: Path) -> None:
    replace_once(rte_copy / SOLAR, SOLAR_LINE, SOLAR_LINE.replace(b"170,61", b"170,62"))

    with pytest.raises(ValueError, match=r"total differs .* first in 2025-04"):
        load_rte_modulation(rte_copy)


def test_a_missing_value_breaks_the_total(rte_copy: Path) -> None:
    replace_once(rte_copy / SOLAR, SOLAR_LINE, SOLAR_LINE.replace(b"170,6127391286", b""))

    with pytest.raises(ValueError, match=r"total differs .* first in 2025-04"):
        load_rte_modulation(rte_copy)


def test_a_dropped_month_is_rejected(rte_copy: Path) -> None:
    replace_once(rte_copy / SOLAR, SOLAR_LINE, b"")

    with pytest.raises(ValueError, match="3 blocks of 82 months, found 245 rows"):
        load_rte_modulation(rte_copy)


def test_a_misdated_month_is_rejected(rte_copy: Path) -> None:
    replace_once(rte_copy / SOLAR, SOLAR_LINE, SOLAR_LINE.replace(b"2025-04", b"2025-05"))

    with pytest.raises(ValueError, match="month by month from 2020-01 to 2026-10"):
        load_rte_modulation(rte_copy)


def test_swapped_files_are_rejected(rte_copy: Path) -> None:
    shutil.copy(RTE / WIND, rte_copy / SOLAR)

    with pytest.raises(ValueError, match="expected only 'Volume de modulation de la production"):
        load_rte_modulation(rte_copy)


def test_an_empty_label_is_rejected(rte_copy: Path) -> None:
    label = b'"Volume de modulation de la production solaire"'
    replace_once(rte_copy / SOLAR, SOLAR_LINE, SOLAR_LINE.replace(label, b""))

    with pytest.raises(ValueError, match="expected only 'Volume de modulation de la production"):
        load_rte_modulation(rte_copy)


def test_a_changed_file_fails_the_manifest(rte_copy: Path) -> None:
    # Same size, one digit changed: only the checksum can tell.
    replace_once(rte_copy / SOLAR, SOLAR_LINE, SOLAR_LINE.replace(b"1286", b"1287"))

    with pytest.raises(ValueError, match=f"{SOLAR}: expected 16337 bytes"):
        verify_manifest(rte_copy)


def test_hirth_files_match_their_manifest() -> None:
    verify_manifest(HIRTH)


def test_hirth_volumes_are_those_of_the_paper_in_gwh() -> None:
    found = load_hirth_curtailment(HIRTH).set_index(["technology", "year"])["gwh"]

    assert found.to_dict() == {
        ("wind", 2023): 1800.0,
        ("wind", 2024): 1600.0,
        ("wind", 2025): 2100.0,
        ("solar", 2025): 600.0,
    }


def test_sdes_files_match_their_manifest() -> None:
    manifest = json.loads((SDES / "manifest.json").read_text(encoding="utf-8"))

    assert set(manifest["files"]) == {SDES_FILE}
    verify_manifest(SDES)


def test_sdes_photovoltaic_self_consumption_is_read_in_gwh() -> None:
    found = load_sdes_self_consumption(SDES)

    assert found.index.tolist() == list(range(2018, 2025))
    # The publication rounds them to 0.2 TWh and 1.8 TWh.
    assert found[2020] == pytest.approx(223.97, abs=0.01)
    assert found[2024] == pytest.approx(1777.01, abs=0.01)


@pytest.fixture
def sdes_copy(tmp_path: Path) -> Path:
    shutil.copy(SDES / SDES_FILE, tmp_path / SDES_FILE)
    return tmp_path


def edit_sdes(directory: Path, row: int, column: int, value: str | float) -> None:
    path = directory / SDES_FILE
    sheet = pd.read_excel(path, sheet_name=SDES_SHEET, header=None)
    sheet.iat[row, column] = value
    sheet.to_excel(path, sheet_name=SDES_SHEET, header=False, index=False)


def test_an_unchanged_rewrite_of_the_sdes_sheet_reads_the_same(sdes_copy: Path) -> None:
    edit_sdes(sdes_copy, SDES_LABELS_ROW, SDES_PHOTOVOLTAIC_COLUMN, "Photovoltaïque")

    # Writing the workbook again moves values by about 1e-13 GWh.
    pd.testing.assert_series_equal(
        load_sdes_self_consumption(sdes_copy), load_sdes_self_consumption(SDES)
    )


def test_a_renamed_sdes_technology_is_rejected(sdes_copy: Path) -> None:
    edit_sdes(sdes_copy, SDES_LABELS_ROW, SDES_PHOTOVOLTAIC_COLUMN, "Solaire")

    with pytest.raises(ValueError, match="Solaire"):
        load_sdes_self_consumption(sdes_copy)


def test_a_changed_sdes_value_breaks_the_total(sdes_copy: Path) -> None:
    edit_sdes(sdes_copy, SDES_2020_ROW, SDES_PHOTOVOLTAIC_COLUMN, 0.5)

    with pytest.raises(ValueError, match="first in 2020"):
        load_sdes_self_consumption(sdes_copy)


def test_documentation_files_match_their_manifest() -> None:
    manifest = json.loads((DOCUMENTATION / "manifest.json").read_text(encoding="utf-8"))

    assert set(manifest["files"]) == {DOCUMENTATION_FILE}
    verify_manifest(DOCUMENTATION)


def test_the_validation_series_are_documented_as_the_plan_expects() -> None:
    statements = load_documentation(DOCUMENTATION).set_index(["series", "dimension"])["content"]

    assert statements[("eco2mix", "self_consumption")] == "included"
    assert statements[("eco2mix", "distribution")] == "covered"
    assert statements[("smard", "self_consumption")] == "excluded"
    assert statements[("smard", "distribution")] == "covered"
    assert statements[("energy_charts_total", "self_consumption")] == "included"


@pytest.fixture
def documentation_copy(tmp_path: Path) -> Path:
    for path in DOCUMENTATION.iterdir():
        shutil.copy(path, tmp_path / path.name)
    return tmp_path


def rewrite_statements(directory: Path, statements: pd.DataFrame) -> None:
    (directory / DOCUMENTATION_FILE).write_text(statements.to_csv(index=False), encoding="utf-8")


def test_an_unknown_content_is_rejected(documentation_copy: Path) -> None:
    statements = load_documentation(documentation_copy)
    statements.loc[0, "content"] = "partial"
    rewrite_statements(documentation_copy, statements)

    with pytest.raises(ValueError, match="partial"):
        load_documentation(documentation_copy)


def test_a_second_statement_on_one_dimension_is_rejected(documentation_copy: Path) -> None:
    statements = load_documentation(documentation_copy)
    rewrite_statements(documentation_copy, pd.concat([statements, statements.iloc[[0]]]))

    with pytest.raises(ValueError, match="two statements on eco2mix self_consumption"):
        load_documentation(documentation_copy)


def test_a_document_missing_from_the_manifest_is_rejected(documentation_copy: Path) -> None:
    statements = load_documentation(documentation_copy)
    statements.loc[0, "document"] = "press_article"
    rewrite_statements(documentation_copy, statements)

    with pytest.raises(ValueError, match="press_article"):
        load_documentation(documentation_copy)
