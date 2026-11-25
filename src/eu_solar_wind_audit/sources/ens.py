"""Master data register of wind turbines of the Danish Energy Agency (Energistyrelsen,
Stamdataregister for vindkraftanlæg): every turbine connected to the Danish grid, with its dates of
grid connection and decommissioning, capacity, rotor diameter, hub height, municipality and
placement on land or at sea.

The agency publishes it in two files (ens.dk, « Data: Oversigt over energisektoren », read
2026-11-25): the current register, about every two months, which keeps the turbines decommissioned
in the three years before the current one, and a historic file of the turbines decommissioned
earlier, once a year. Both are pinned. The page names no licence, so the register enters the model
only and none of its rows is published.

As read on the pinned files:
- the current file swaps the headers of the municipality and placement columns, so both columns
  are found by their content;
- unknown rotor diameters and hub heights are written 0.1 or 0, and test turbines have no
  capacity;
- household turbines, 25 kW or less, have no placement;
- the municipality "0 - Ukendt (0)" is unknown;
- some turbines are listed twice with the same values, and a turbine moved to another site keeps
  its number on a second row.
"""

from collections.abc import Callable, Sequence
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import pandera.pandas as pa
from pandera.typing import DataFrame, Series

from eu_solar_wind_audit.sources import pinned
from eu_solar_wind_audit.sources.mastr import WindGenerations

# The register at the end of October 2026, uploaded in November 2026, and the turbines
# decommissioned up to the end of 2022, uploaded in September 2026. Checksums computed on the
# downloads of 2026-11-22 and 2026-11-25.
CURRENT_URL = "https://ens.dk/media/8748/download"
CURRENT_SHA256 = "167e8bf514f13cfd7d69846bcda97d64824e8213304c5c9d92b26ce8c772fba9"
HISTORIC_URL = "https://ens.dk/media/8746/download"
HISTORIC_SHA256 = "e08632169f78cbfe6cbe6eb9c8c15cc21ac4c1bab9c7478115a3c8afe9fe92b4"
RAW_DIR = Path("data/raw/ens")

# Rows 0 and 1 group the columns; row 2 names them.
HEADER_ROW = 2
COLUMNS = {
    "Møllenummer (GSRN)": "turbine",
    "Dato for oprindelig nettilslutning": "connected",
    "Dato for afmeldning": "decommissioned",
    "Kapacitet (kW)": "capacity_kw",
    "Rotor-diameter (m)": "rotor_diameter_m",
    "Navhøjde (m)": "hub_height_m",
}
PLACEMENTS = {"LAND", "HAV"}
MUNICIPALITY = r"(\d+) - "
UNKNOWN_SIZE_M = 0.1


class Turbines(pa.DataFrameModel):
    """One row per turbine and site: `turbine` is its GSRN number, `municipality` the number of
    its municipality, null when unknown, and `offshore` tells a turbine at sea."""

    turbine: Series[str] = pa.Field(str_matches=r"^\d{18}$")
    connected: Series[pa.DateTime]
    decommissioned: Series[pa.DateTime] = pa.Field(nullable=True)
    capacity_kw: Series[float] = pa.Field(ge=0, nullable=True)
    rotor_diameter_m: Series[float] = pa.Field(gt=0, nullable=True)
    hub_height_m: Series[float] = pa.Field(gt=0, nullable=True)
    municipality: Series[pd.Int64Dtype] = pa.Field(gt=0, nullable=True)
    offshore: Series[bool]

    class Config:
        strict = True


class MonthlyCapacity(pa.DataFrameModel):
    """Capacity in MW of the onshore turbines of each municipality at each month end."""

    month_end: Series[pa.DateTime]
    municipality: Series[int]
    capacity_mw: Series[float] = pa.Field(ge=0)

    class Config:
        strict = True


def download_registers(
    client: httpx.Client | None = None, raw_dir: Path = RAW_DIR
) -> tuple[Path, Path]:
    """Paths of the pinned current and historic files, downloaded and checked unless cached."""
    current = pinned.download(
        CURRENT_URL, CURRENT_SHA256, raw_dir / "vinddata_2026-10.xlsx", client
    )
    historic = pinned.download(
        HISTORIC_URL, HISTORIC_SHA256, raw_dir / "historiske_vinddata_2022.xlsx", client
    )
    return current, historic


def read_turbines(paths: tuple[Path, Path] | None = None) -> DataFrame[Turbines]:
    """`Turbines` of the current and historic files, downloaded when `paths` is None. Repeated
    rows count once, and turbines without a grid connection date are left out."""
    current, historic = paths or download_registers()
    turbines = pd.concat([_read(current), _read(historic)], ignore_index=True)
    turbines = turbines.drop_duplicates().dropna(subset=["connected"])
    return Turbines.validate(turbines.sort_values(["connected", "turbine"], ignore_index=True))


def monthly_capacity(
    turbines: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp
) -> DataFrame[MonthlyCapacity]:
    """Capacity of the onshore turbines at each month end from `start` to `end`, naive dates, by
    municipality. A turbine counts at a month end on or after its grid connection date and before
    its decommissioning date. Turbines at sea, without a municipality or without a capacity are
    left out."""
    counted = Turbines.validate(turbines)
    counted = counted[
        ~counted["offshore"] & counted["municipality"].notna() & counted["capacity_kw"].notna()
    ]
    capacity_mw = counted["capacity_kw"] / 1000
    events = pd.concat(
        [
            pd.DataFrame(
                {
                    "month_end": counted["connected"] + pd.offsets.MonthEnd(0),
                    "municipality": counted["municipality"],
                    "change": capacity_mw,
                }
            ),
            pd.DataFrame(
                {
                    "month_end": counted["decommissioned"] + pd.offsets.MonthEnd(0),
                    "municipality": counted["municipality"],
                    "change": -capacity_mw,
                }
            ).dropna(subset=["month_end"]),
        ],
        ignore_index=True,
    )
    changes = events.groupby(["month_end", "municipality"])["change"].sum().unstack(fill_value=0)
    months = pd.date_range(min(changes.index.min(), start), end, freq="ME", name="month_end")
    # Capacities have at most 2 decimals in kW, so 5 in MW; rounding removes the residue of sums.
    cumulative = changes.reindex(months, fill_value=0.0).cumsum().loc[start:].round(5)
    panel = cumulative.melt(value_name="capacity_mw", ignore_index=False).reset_index()
    panel["month_end"] = panel["month_end"].astype("datetime64[ns]")
    panel["municipality"] = panel["municipality"].astype("int64")
    return MonthlyCapacity.validate(
        panel.sort_values(["month_end", "municipality"], ignore_index=True)
    )


def wind_generations(
    turbines: pd.DataFrame, as_of: pd.Timestamp, last_years: Sequence[int | None]
) -> DataFrame[WindGenerations]:
    """`WindGenerations` of the onshore turbines in operation at `as_of`, one row per window of
    grid connection years ending with each of `last_years`, the last of which is None. Hub heights
    and rotor diameters count where known."""
    counted = Turbines.validate(turbines)
    active = counted[
        ~counted["offshore"]
        & counted["capacity_kw"].notna()
        & (counted["connected"] <= as_of)
        & ~(counted["decommissioned"] <= as_of)
    ]
    year = active["connected"].dt.year.to_numpy()
    ends = [np.inf if last is None else float(last) for last in last_years]
    rows = []
    for last, low, high in zip(last_years, [-np.inf, *ends], ends, strict=False):
        window = active[(year > low) & (year <= high)]
        with_hub = window.dropna(subset=["hub_height_m"])
        with_rotor = window.dropna(subset=["rotor_diameter_m"])
        swept_m2 = np.pi * (with_rotor["rotor_diameter_m"] / 2) ** 2
        rows.append(
            {
                "last_commissioning_year": last,
                "capacity_mw": window["capacity_kw"].sum() / 1000,
                "hub_height_m": float(
                    np.average(with_hub["hub_height_m"], weights=with_hub["capacity_kw"])
                ),
                "specific_power_w_m2": float(
                    with_rotor["capacity_kw"].sum() * 1000 / swept_m2.sum()
                ),
            }
        )
    table = pd.DataFrame(rows).astype({"last_commissioning_year": "Int64"})
    return WindGenerations.validate(table)


def _read(path: Path) -> pd.DataFrame:
    """The turbines of one file of the register, in the columns of `Turbines`."""
    raw = pd.read_excel(path, header=HEADER_ROW, dtype=object)
    text = raw.astype("string")
    municipality = _column_where(
        raw, lambda values: bool(values.astype("string").str.match(MUNICIPALITY).all())
    )
    placement = _column_where(raw, lambda values: bool(values.isin(PLACEMENTS).all()))
    number = text[municipality].str.extract(MUNICIPALITY, expand=False).astype("Int64")
    frame = pd.DataFrame(
        {
            "turbine": text["Møllenummer (GSRN)"],
            "connected": pd.to_datetime(raw["Dato for oprindelig nettilslutning"]),
            "decommissioned": pd.to_datetime(raw["Dato for afmeldning"]),
            "capacity_kw": _numbers(text["Kapacitet (kW)"]),
            "rotor_diameter_m": _sizes(text["Rotor-diameter (m)"]),
            "hub_height_m": _sizes(text["Navhøjde (m)"]),
            "municipality": number.mask(number == 0),
            "offshore": raw[placement].eq("HAV").astype(bool),
        }
    )
    return frame.astype({"connected": "datetime64[ns]", "decommissioned": "datetime64[ns]"})


def _column_where(raw: pd.DataFrame, rule: Callable[[pd.Series], bool]) -> str:
    """The one column whose values, nulls left out, all follow `rule`."""
    found = [
        column for column in raw.columns if raw[column].notna().any() and rule(raw[column].dropna())
    ]
    if len(found) != 1:
        raise ValueError(f"expected one column, found {found}")
    return str(found[0])


def _numbers(text: "pd.Series[str]") -> "pd.Series[float]":
    """Numbers written with a decimal comma or as plain numbers."""
    numbers: pd.Series[float] = pd.to_numeric(text.str.replace(",", ".", regex=False)).astype(float)
    return numbers


def _sizes(text: "pd.Series[str]") -> "pd.Series[float]":
    """Sizes in m, null where the register writes 0.1 or 0 for an unknown size."""
    sizes = _numbers(text)
    return sizes.mask(sizes <= UNKNOWN_SIZE_M)
