"""Axis 3 of the evaluation plan, what each published series contains, from its yearly level.

Eurostat gross production counts all the output, including the output that producers consume on
site. Each published series is set against it, solar against wind: wind has no self-consumption,
so its ratio carries what the two technologies share, such as a smaller territory or missing
grids. The gap is the log of the solar ratio over the wind ratio. A series that leaves out a share
s of solar output, the self-consumed share, has a gap of log(1 - s); one that includes it has a
gap of zero.

The error of the gap is measured on the validation series, whose content is known: the root mean
square, over their years, of the gap minus the gap of their content. A year is decided only when s
reaches the detection threshold, the share that a one-sided test at 5 % detects with 80 % power
given that error, as for the minimal detectable effect of axis 4. It then goes to the content
whose gap is closer to the observed one; below the threshold it is inconclusive (evaluation plan,
axis 3, Decision rules).
"""

from typing import ClassVar, Literal

import numpy as np
import pandas as pd
import pandera.pandas as pa
from pandera.typing import DataFrame, Series

from eu_solar_wind_audit.detectability import Z_SUM
from eu_solar_wind_audit.sources import sdes

Content = Literal["included", "excluded"]

# Published national series and their countries. Energy-Charts total power is the SMARD feed-in
# times a monthly factor plus the self-consumption estimated by Fraunhofer ISE
# (sources/energy_charts.py).
SERIES = {"eco2mix": "FR", "smard": "DE", "energy_charts_total": "DE"}
# Whether the solar output of the validation series includes self-consumption (evaluation plan,
# axis 3). Energy-Charts total power is classified too, but its errors are those of SMARD.
VALIDATION: dict[str, Content] = {"eco2mix": "included", "smard": "excluded"}
Technology = Literal["solar", "wind"]
TECHNOLOGIES: tuple[Technology, ...] = ("solar", "wind")
TIMEZONE = "Europe/Paris"


class Levels(pa.DataFrameModel):
    """Yearly level of a published series against Eurostat gross production: the published output
    in GWh, the gross production of the country, the share of its capacity inside the perimeter of
    the series, the published output over the gross production inside that perimeter, the gap
    log(solar ratio / wind ratio), and the self-consumed share s of solar output, with the gap it
    gives to a series that leaves it out, log(1 - s)."""

    series: Series[str] = pa.Field(isin=list(SERIES))
    year: Series[int]
    published_solar_gwh: Series[float] = pa.Field(ge=0)
    published_wind_gwh: Series[float] = pa.Field(ge=0)
    gross_solar_gwh: Series[float] = pa.Field(gt=0)
    gross_wind_gwh: Series[float] = pa.Field(gt=0)
    inside_solar: Series[float] = pa.Field(gt=0, le=1)
    inside_wind: Series[float] = pa.Field(gt=0, le=1)
    solar_ratio: Series[float]
    wind_ratio: Series[float]
    gap: Series[float]
    self_consumption_share: Series[float] = pa.Field(ge=0, lt=1)
    gap_if_excluded: Series[float] = pa.Field(le=0)

    class Config:
        strict = True
        unique: ClassVar[list[str]] = ["series", "year"]


def annual_energy(hourly: pd.DataFrame) -> "pd.Series[float]":
    """Energy in GWh of the complete hours of a grid.Hourly frame, or of a frame that extends it,
    in each Europe/Paris calendar year, the years of Eurostat; Berlin time has the same offsets."""
    local = pd.DatetimeIndex(hourly.index).tz_convert(TIMEZONE)
    energy = hourly["energy_mwh"].groupby(local.year).sum() / 1000
    return energy.rename("gwh").rename_axis("year")


def year_bounds(first: int, last: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    """UTC hour starts of the Europe/Paris calendar years first to last."""
    start = pd.Timestamp(f"{first}-01-01", tz=TIMEZONE).tz_convert("UTC")
    end = pd.Timestamp(f"{last + 1}-01-01", tz=TIMEZONE).tz_convert("UTC")
    return start, end


def inside_shares(panel: pd.DataFrame, totals: pd.DataFrame) -> "pd.Series[float]":
    """Share of the capacity of France in mainland France without Corsica, the perimeter of the
    éCO2mix national series, as the mean over the quarter ends of each year. `panel` and `totals`
    are an SDES workbook read by sdes.read_capacity and sdes.read_totals, cut to the technologies
    of one series."""
    mainland = sdes.metropolitan_without_corsica(panel).groupby("quarter_end")["capacity_mw"].sum()
    national = totals[totals["total"] == "Total France"]
    france = national.groupby("quarter_end")["capacity_mw"].sum()
    share = (mainland / france).dropna()
    years = pd.DatetimeIndex(share.index).year
    return share.groupby(years).mean().rename("inside").rename_axis("year")


def levels(
    published: pd.DataFrame,
    gross: pd.DataFrame,
    inside: pd.DataFrame,
    self_consumption: pd.DataFrame,
) -> DataFrame[Levels]:
    """`Levels` of each series and year found in all the inputs: `published` and `inside` with the
    columns series, year, solar and wind; `gross` with country, year, solar and wind in GWh; and
    `self_consumption` with country, year and share."""
    table = (
        published.merge(inside, on=["series", "year"], suffixes=("", "_inside"))
        .assign(country=lambda frame: frame["series"].map(SERIES))
        .merge(gross, on=["country", "year"], suffixes=("", "_gross"))
        .merge(self_consumption, on=["country", "year"])
    )
    solar_ratio = table["solar"] / (table["solar_gross"] * table["solar_inside"])
    wind_ratio = table["wind"] / (table["wind_gross"] * table["wind_inside"])
    frame = pd.DataFrame(
        {
            "series": table["series"],
            "year": table["year"].astype("int64"),
            "published_solar_gwh": table["solar"],
            "published_wind_gwh": table["wind"],
            "gross_solar_gwh": table["solar_gross"],
            "gross_wind_gwh": table["wind_gross"],
            "inside_solar": table["solar_inside"],
            "inside_wind": table["wind_inside"],
            "solar_ratio": solar_ratio,
            "wind_ratio": wind_ratio,
            "gap": np.log(solar_ratio / wind_ratio),
            "self_consumption_share": table["share"],
            "gap_if_excluded": np.log1p(-table["share"]),
        }
    )
    ordered = frame.sort_values(["series", "year"]).reset_index(drop=True)
    return Levels.validate(ordered)


def gap_errors(table: DataFrame[Levels]) -> "pd.Series[float]":
    """Gap of each year of the validation series minus the gap of its known content, indexed by
    series and year."""
    rows = table[table["series"].isin(list(VALIDATION))].set_index(["series", "year"])
    excluded = rows.index.get_level_values("series").map(VALIDATION) == "excluded"
    expected = rows["gap_if_excluded"].where(excluded, 0.0)
    return (rows["gap"] - expected).rename("error")


def error_rms(errors: "pd.Series[float]") -> float:
    """Root mean square of the errors of the gap."""
    return float(np.sqrt(np.mean(np.square(errors))))


def detection_threshold(error: float) -> float:
    """Smallest self-consumed share s whose gap log(1 - s) a one-sided test at 5 % detects with
    80 % power, the gap having a normal error of standard deviation `error`."""
    return float(-np.expm1(-Z_SUM * error))


def decide(table: DataFrame[Levels], threshold: float) -> "pd.Series[str]":
    """Content of the solar output of each row of `table`: inconclusive below `threshold`,
    otherwise the content whose gap is closer to the observed one."""
    closer_to_excluded = table["gap"] < table["gap_if_excluded"] / 2
    decisions = np.where(
        table["self_consumption_share"] < threshold,
        "inconclusive",
        np.where(closer_to_excluded, "excluded", "included"),
    )
    return pd.Series(decisions, index=table.index, name="decision", dtype="str")


def verdict(decisions: "pd.Series[str]") -> str:
    """Content of a series from the decisions of its years: the one its decided years agree on,
    ambiguous when they disagree, inconclusive when no year is decided."""
    decided = set(decisions[decisions != "inconclusive"])
    if not decided:
        return "inconclusive"
    if len(decided) > 1:
        return "ambiguous"
    (content,) = decided
    return content
