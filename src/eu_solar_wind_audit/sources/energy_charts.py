"""Installed capacity from the Energy-Charts API of Fraunhofer ISE (licence CC BY 4.0, attribution
"energy-charts.info", as stated in the `license` field of each response).

A month's value is the capacity installed at the end of that month. Monthly values are published
for Germany only.
"""

from typing import Any, ClassVar

import pandas as pd
import pandera.pandas as pa
from pandera.typing import DataFrame, Series

TECHNOLOGIES = ("solar_ac", "solar_dc", "wind_onshore", "wind_offshore")
MW_PER_UNIT = {"MW": 1.0, "GW": 1000.0}


class InstalledCapacity(pa.DataFrameModel):
    month: Series[pa.DateTime]
    technology: Series[str] = pa.Field(isin=TECHNOLOGIES)
    capacity_mw: Series[float] = pa.Field(ge=0)

    @pa.check("month")
    @classmethod
    def first_day_of_month(cls, month: "pd.Series[pd.Timestamp]") -> "pd.Series[bool]":
        return month.dt.is_month_start & (month == month.dt.normalize())

    class Config:
        strict = True
        unique: ClassVar[list[str]] = ["month", "technology"]


def parse_installed_power(payload: dict[str, Any]) -> DataFrame[InstalledCapacity]:
    """Tidy panel from a /v2/installed_power response; null values are left out.

    The month in progress at `generated_at` is dropped: its value is not yet an end-of-month
    capacity.
    """
    units = {series["id"]: series.get("unit") or payload["unit"] for series in payload["series"]}
    technologies = [technology for technology in TECHNOLOGIES if technology in units]
    unknown = {units[technology] for technology in technologies} - MW_PER_UNIT.keys()
    if unknown:
        raise ValueError(f"unexpected units: {sorted(unknown)}")

    # Timestamps are local midnights with a seasonal UTC offset.
    timezone = payload["timezone"]
    stamps = pd.to_datetime([row["timestamp"] for row in payload["data"]], utc=True)
    months = stamps.tz_convert(timezone).tz_localize(None).as_unit("ns")
    generated_at = pd.Timestamp(payload["generated_at"]).tz_convert(timezone).tz_localize(None)
    ended = months + pd.offsets.MonthBegin(1) <= generated_at

    values = pd.DataFrame(
        [row["values"] for row in payload["data"]],
        index=pd.Index(months, name="month"),
        columns=pd.Index(technologies, name="technology"),
        dtype=float,
    )
    capacity_mw = values[ended] * [MW_PER_UNIT[units[technology]] for technology in technologies]
    tidy = capacity_mw.reset_index().melt(id_vars="month", value_name="capacity_mw").dropna()
    return InstalledCapacity.validate(tidy.reset_index(drop=True))
