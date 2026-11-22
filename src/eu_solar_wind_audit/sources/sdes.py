"""Installed solar and wind capacity of France by department, from the quarterly dashboards of the
SDES, the statistics service of the French ministry for the environment."""

from pathlib import Path
from typing import ClassVar

import httpx
import pandas as pd
import pandera.pandas as pa
from pandera.typing.pandas import Series

BASE_URL = "https://www.statistiques.developpement-durable.gouv.fr"
# Each quarterly publication gets new media ids; these are the second quarter 2026 files.
SOLAR_2026_Q2 = 9714
WIND_2026_Q2 = 9715
RAW_DIR = Path("data/raw/sdes")
XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

SHEETS = {
    "solaire PV": "solar",
    "éolien_terrestre": "wind_onshore",
    "éolien_offshore": "wind_offshore",
}
# The capacities match AC figures (Eurostat 2024), so they are used as AC without conversion.
UNIT_ROW = "Unité : puissance en MW"
CORSICA = ("20", "2A", "2B")


class Panel(pa.DataFrameModel):
    department_code: Series[str] = pa.Field(str_matches=r"^(\d{2}|2[AB]|97\d)$")
    department: Series[str]
    region: Series[str]
    quarter_end: Series[pd.Timestamp]
    technology: Series[str] = pa.Field(isin=list(SHEETS.values()))
    installations: Series[int] = pa.Field(ge=0)
    capacity_mw: Series[float] = pa.Field(ge=0)
    provisional: Series[bool]

    class Config:
        strict = True
        unique: ClassVar[list[str]] = ["department_code", "technology", "quarter_end"]


def download_dashboard(client: httpx.Client, media_id: int, raw_dir: Path = RAW_DIR) -> Path:
    """Path of the dashboard workbook `media_id`, downloaded unless already cached."""
    dest = raw_dir / f"sdes-media-{media_id}.xlsx"
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".part")
    url = f"{BASE_URL}/media/{media_id}/download?inline"
    with client.stream("GET", url) as response:
        response.raise_for_status()
        # Never cache an error or maintenance page under the workbook name.
        if not response.headers.get("content-type", "").startswith(XLSX):
            raise ValueError(f"{url}: not an xlsx file ({response.headers.get('content-type')})")
        with partial.open("wb") as file:
            for chunk in response.iter_bytes():
                file.write(chunk)
    partial.replace(dest)
    return dest


def _read_sheet(book: pd.ExcelFile, sheet: str) -> pd.DataFrame:
    """Every labelled row of a sheet in long form: one row per sheet row and quarter end."""
    raw = pd.read_excel(book, sheet_name=sheet, header=None, dtype=object)
    if raw.iat[1, 0] != UNIT_ROW:
        raise ValueError(f"{sheet}: expected {UNIT_ROW!r}, found {raw.iat[1, 0]!r}")
    measures = raw.iloc[3, 3:].to_list()
    if measures != ["nombre", "puissance"] * (len(measures) // 2):
        raise ValueError(f"{sheet}: value columns are not nombre/puissance pairs")
    dates = raw.iloc[2, 3:].to_list()
    quarter_ends = pd.DatetimeIndex(dates[::2], name="quarter_end")
    if dates[::2] != dates[1::2] or not quarter_ends.is_quarter_end.all():
        raise ValueError(f"{sheet}: header dates are not quarter ends in pairs")

    # Blank lines and notes have no values.
    body = raw.iloc[4:].dropna(subset=raw.columns[3:], how="all")
    rows = pd.DataFrame(
        {
            "label": body[0],
            "department_code": body[2],
            # Corsica and the overseas departments are single rows with a code and no region.
            "department": body[1].fillna(body[0]),
            "region": body[0].ffill(),
        }
    )
    values = pd.DataFrame(
        {
            "installations": body.iloc[:, 3::2].set_axis(quarter_ends, axis=1).stack(),
            "capacity_mw": body.iloc[:, 4::2].set_axis(quarter_ends, axis=1).stack(),
        }
    ).astype({"installations": "int64", "capacity_mw": "float64"})
    long = rows.join(values.reset_index("quarter_end"))
    long.insert(4, "technology", SHEETS[sheet])
    long["provisional"] = long["quarter_end"] == quarter_ends.max()
    return long.reset_index(drop=True)


def _read_workbook(path: Path) -> pd.DataFrame:
    with pd.ExcelFile(path) as book:
        sheets = [str(name) for name in book.sheet_names if name in SHEETS]
        if not sheets:
            raise ValueError(f"{path}: none of the sheets {list(SHEETS)}")
        return pd.concat([_read_sheet(book, sheet) for sheet in sheets], ignore_index=True)


def read_capacity(path: Path) -> pd.DataFrame:
    """Quarterly panel of installations and capacity per department and technology.

    The last quarter of each sheet is provisional and revised in the next publications.
    """
    long = _read_workbook(path)
    departments = long[long["department_code"].notna()].drop(columns="label")
    departments = departments.astype(
        {"department_code": "str", "department": "str", "region": "str"}
    )
    return Panel.validate(departments.reset_index(drop=True))


def read_totals(path: Path) -> pd.DataFrame:
    """The "Total ..." rows of each sheet, such as "Total métropole" and "Total France"."""
    long = _read_workbook(path)
    totals = long[long["label"].str.startswith("Total", na=False)]
    columns = ["label", "technology", "quarter_end", "installations", "capacity_mw", "provisional"]
    return totals[columns].rename(columns={"label": "total"}).reset_index(drop=True)


def metropolitan_without_corsica(panel: pd.DataFrame) -> pd.DataFrame:
    """Rows of mainland France, the perimeter of the national éCO2mix series."""
    code = panel["department_code"]
    return panel[(code.str.len() == 2) & ~code.isin(CORSICA)]
