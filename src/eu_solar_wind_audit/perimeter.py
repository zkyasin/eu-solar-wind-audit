"""Axis 3 of the evaluation plan, what each published series contains, from its yearly level.

Eurostat gross production counts all the output, including the output that producers consume on
site. Each published series is set against it, solar against wind: wind has no self-consumption,
so its ratio carries what the two technologies share, such as a smaller territory or missing
grids. The gap is the log of the solar ratio over the wind ratio. A series that leaves out a share
s of solar output, the self-consumed share, has a gap of log(1 - s); one that includes it has a
gap of zero. A series that leaves out the distribution grids, where a share d of wind capacity is
connected, has a wind ratio of about 1 - d; one that covers them has a wind ratio of one.

The error of each statistic is measured on the validation series, whose content is known: the
root mean square, over their years, of the statistic minus its value for their content. A year is
decided only when s, or d, reaches the detection threshold, the share that a one-sided test at 5 %
detects with 80 % power given that error, as for the minimal detectable effect of axis 4. It then
goes to the content whose value is closer to the observed one; below the threshold it is
inconclusive.

A series takes the content that its decided years agree on. A dimension of the matrix is decided
when this content agrees with the statement of the producer, the second type of evidence; the
cell of the series combines both dimensions (evaluation plan, axis 3, Decision rules). A
validation series is checked on the dimensions whose content is known.
"""

from typing import ClassVar, Literal

import numpy as np
import pandas as pd
import pandera.pandas as pa
from pandera.typing import DataFrame, Series

from eu_solar_wind_audit.detectability import Z_SUM
from eu_solar_wind_audit.sources import sdes

# Published national series and their countries. Energy-Charts total power is the SMARD feed-in
# times a monthly factor plus the self-consumption estimated by Fraunhofer ISE
# (sources/energy_charts.py).
SERIES = {"eco2mix": "FR", "smard": "DE", "energy_charts_total": "DE"}
DIMENSIONS = ("self_consumption", "distribution")
# The cell of each content of the two dimensions (evaluation plan, axis 3).
CELLS = {
    ("included", "covered"): "complete",
    ("included", "not_covered"): "partial",
    ("excluded", "covered"): "feed_in_only",
    ("excluded", "not_covered"): "partial_feed_in",
}
# Expected content of the validation series on each dimension they check (evaluation plan, axis
# 3, and Deviations).
EXPECTED: dict[str, dict[str, str]] = {
    "eco2mix": {"self_consumption": "included", "distribution": "covered"},
    "smard": {"self_consumption": "excluded", "distribution": "covered"},
    "energy_charts_total": {"self_consumption": "included", "distribution": "covered"},
}
# The series whose years measure the errors: those of Energy-Charts total power are SMARD's.
ERROR_SERIES = ("eco2mix", "smard")
Technology = Literal["solar", "wind"]
TECHNOLOGIES: tuple[Technology, ...] = ("solar", "wind")
TIMEZONE = "Europe/Paris"


class Levels(pa.DataFrameModel):
    """Yearly level of a published series against Eurostat gross production: the published output
    in GWh, the gross production of the country, the share of its capacity inside the perimeter of
    the series, the published output over the gross production inside that perimeter, the gap
    log(solar ratio / wind ratio), the self-consumed share s of solar output, with the gap it gives
    to a series that leaves it out, log(1 - s), and the share d of wind capacity connected to the
    distribution grids, with the log wind ratio of a series that leaves them out, log(1 - d). s is
    null where no estimate is published."""

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
    self_consumption_share: Series[float] = pa.Field(ge=0, lt=1, nullable=True)
    gap_if_excluded: Series[float] = pa.Field(le=0, nullable=True)
    distribution_share: Series[float] = pa.Field(ge=0, lt=1)
    log_wind_ratio_if_not_covered: Series[float] = pa.Field(le=0)

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


def listed_share(capacity: "pd.Series[float]", units: pd.DataFrame) -> float:
    """Share of the wind capacity of a country, in MW indexed by technology as
    sources.entsoe.fetch_capacity gives it, held by the wind farms of `units`, a
    sources.entsoe.UnitCapacity table: the list of units of 100 MW or more that the Transparency
    Platform publishes."""
    wind = ["wind_onshore", "wind_offshore"]
    listed = units.loc[units["technology"].isin(wind), "capacity_mw"].sum()
    return float(listed / capacity.reindex(wind).sum())


def levels(
    published: pd.DataFrame,
    gross: pd.DataFrame,
    inside: pd.DataFrame,
    self_consumption: pd.DataFrame,
    distribution: "pd.Series[float]",
) -> DataFrame[Levels]:
    """`Levels` of each series and year found in all the inputs: `published` and `inside` with the
    columns series, year, solar and wind; `gross` with country, year, solar and wind in GWh;
    `self_consumption` with country, year and share, for the countries that publish one; and
    `distribution`, the distribution share of wind capacity, indexed by country."""
    table = (
        published.merge(inside, on=["series", "year"], suffixes=("", "_inside"))
        .assign(country=lambda frame: frame["series"].map(SERIES))
        .merge(gross, on=["country", "year"], suffixes=("", "_gross"))
        .merge(self_consumption, on=["country", "year"], how="left")
        .merge(distribution.rename("distribution"), left_on="country", right_index=True)
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
            "distribution_share": table["distribution"],
            "log_wind_ratio_if_not_covered": np.log1p(-table["distribution"]),
        }
    )
    ordered = frame.sort_values(["series", "year"]).reset_index(drop=True)
    return Levels.validate(ordered)


def gap_errors(table: DataFrame[Levels]) -> "pd.Series[float]":
    """Gap of each year of the error series minus the gap of its expected content, indexed by
    series and year."""
    rows = _error_rows(table)
    excluded = _expected(rows, "self_consumption") == "excluded"
    expected = rows["gap_if_excluded"].where(excluded, 0.0)
    return (rows["gap"] - expected).rename("error")


def wind_errors(table: DataFrame[Levels]) -> "pd.Series[float]":
    """Log wind ratio of each year of the error series minus that of its expected content,
    indexed by series and year."""
    rows = _error_rows(table)
    not_covered = _expected(rows, "distribution") == "not_covered"
    expected = rows["log_wind_ratio_if_not_covered"].where(not_covered, 0.0)
    log_ratio: pd.Series[float] = np.log(rows["wind_ratio"])
    return (log_ratio - expected).rename("error")


def error_rms(errors: "pd.Series[float]") -> float:
    """Root mean square of the errors of a statistic."""
    return float(np.sqrt(np.mean(np.square(errors))))


def detection_threshold(error: float) -> float:
    """Smallest share s left out of a series whose statistic log(1 - s) a one-sided test at 5 %
    detects with 80 % power, the statistic having a normal error of standard deviation `error`."""
    return float(-np.expm1(-Z_SUM * error))


def decide_self_consumption(table: DataFrame[Levels], threshold: float) -> "pd.Series[str]":
    """Whether the solar output of each row of `table` includes self-consumption: inconclusive
    when the self-consumed share is unknown or below `threshold`, otherwise the content whose gap
    is closer to the observed one."""
    share = table["self_consumption_share"]
    return _decide(
        table["gap"],
        table["gap_if_excluded"],
        share.isna() | (share < threshold),
        ("included", "excluded"),
    )


def decide_distribution(table: DataFrame[Levels], threshold: float) -> "pd.Series[str]":
    """Whether each row of `table` covers the distribution grids: inconclusive when their share
    of wind capacity is below `threshold`, otherwise the content whose log wind ratio is closer to
    the observed one."""
    return _decide(
        np.log(table["wind_ratio"]),
        table["log_wind_ratio_if_not_covered"],
        table["distribution_share"] < threshold,
        ("covered", "not_covered"),
    )


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


def combine(level: str, documented: str | None) -> str:
    """Content of one dimension of a series from its two types of evidence: decided when the
    level is decided and the producer states the same content, ambiguous when they differ."""
    if level == "inconclusive" or documented is None:
        return "inconclusive"
    if level == "ambiguous" or level != documented:
        return "ambiguous"
    return level


def cells(decided: pd.DataFrame, documentation: pd.DataFrame) -> pd.DataFrame:
    """Matrix of axis 3, one row per series of `decided`, the Levels with the decision of each
    year in a column per dimension: for each dimension, the content from the level, the content
    that the producer states (references.load_documentation) and their combination; then the cell
    of the series, its expected cells, and the validation, pass, fail or inconclusive."""
    stated = documentation.set_index(["series", "dimension"])["content"].to_dict()
    rows = []
    for series, years in decided.groupby("series"):
        name = str(series)
        row: dict[str, str | None] = {"series": name}
        combined = []
        for dimension in DIMENSIONS:
            level = verdict(years[dimension])
            documented = stated.get((name, dimension))
            content = combine(level, documented)
            combined.append(content)
            row |= {
                f"{dimension}_level": level,
                f"{dimension}_documented": documented,
                dimension: content,
            }
        self_consumption, distribution = combined
        cell = CELLS.get((self_consumption, distribution)) or (
            "ambiguous" if "ambiguous" in combined else "inconclusive"
        )
        expected = EXPECTED.get(name)
        row |= {
            "cell": cell,
            "expected_cell": None if expected is None else " or ".join(_cells_of(expected)),
            "validation": _validation(dict(zip(DIMENSIONS, combined, strict=True)), expected),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _cells_of(expected: dict[str, str]) -> list[str]:
    """The cells whose contents are those `expected` on the dimensions it gives."""
    return [
        cell
        for contents, cell in CELLS.items()
        if all(
            expected.get(dimension, content) == content
            for dimension, content in zip(DIMENSIONS, contents, strict=True)
        )
    ]


def _validation(contents: dict[str, str], expected: dict[str, str] | None) -> str | None:
    """Fail when a checked dimension is ambiguous or takes another content, inconclusive when
    one is inconclusive, pass otherwise."""
    if expected is None:
        return None
    checked = [contents[dimension] for dimension in expected]
    if "ambiguous" in checked:
        return "fail"
    if "inconclusive" in checked:
        return "inconclusive"
    agree = all(contents[dimension] == content for dimension, content in expected.items())
    return "pass" if agree else "fail"


def _error_rows(table: DataFrame[Levels]) -> pd.DataFrame:
    return table[table["series"].isin(ERROR_SERIES)].set_index(["series", "year"])


def _expected(rows: pd.DataFrame, dimension: str) -> "pd.Series[str]":
    """Expected content of `dimension` for each row of the error series."""
    series = rows.index.get_level_values("series").to_series(index=rows.index)
    return series.map(lambda name: EXPECTED[name][dimension])


def _decide(
    observed: "pd.Series[float]",
    if_left_out: "pd.Series[float]",
    below_threshold: "pd.Series[bool]",
    names: tuple[str, str],
) -> "pd.Series[str]":
    """The first of `names` where `observed` is closer to zero than to `if_left_out`, the second
    otherwise, and inconclusive where `below_threshold`."""
    closer_to_left_out = observed < if_left_out / 2
    decisions = np.where(
        below_threshold, "inconclusive", np.where(closer_to_left_out, names[1], names[0])
    )
    return pd.Series(decisions, index=observed.index, dtype="str")
