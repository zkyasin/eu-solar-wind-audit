"""Output modulated during negative prices in mainland France (evaluation plan, axis 4).

The estimate is M-hat = sum over N hours of (p* - y), with y the published output and p* the frozen
calibrated potential (fits) times the ratio of its stratum and the annual level of its year,
estimated on the eligible hours of that year (strata.annual_levels). N hours have the negative
price class and meet every other rule of eligibility.Eligibility (eligibility.negative_price_hours),
so they never enter the fit, the strata or the level. Sums are signed, never truncated hour by hour,
and given by Europe/Paris month, the months of the RTE series.
"""

from collections.abc import Sequence

import pandas as pd
import pandera.pandas as pa
from pandera.typing import DataFrame, Index, Series

from eu_solar_wind_audit import calibration, eligibility, fits, grid, strata

TIMEZONE = "Europe/Paris"
# The plan reports the share of M-hat from strata with fewer than 50 P hours.
THIN_STRATUM_HOURS = 50
MWH_PER_GWH = 1000.0


class NegativeHours(pa.DataFrameModel):
    """An N hour: its published output, its potential p* and their gap, the modulation, in MWh,
    and whether its stratum had fewer than THIN_STRATUM_HOURS training hours in the frozen fit."""

    time: Index[grid.UtcTime] = pa.Field(unique=True, check_name=True)
    published_mwh: Series[float]
    potential_mwh: Series[float]
    modulation_mwh: Series[float]
    thin_stratum: Series[bool]

    class Config:
        strict = True


class Months(pa.DataFrameModel):
    """M-hat of a Europe/Paris month, `YYYY-MM`, and what it rests on: its N hours, the potential
    and the published output over them, and the part of M-hat from thin strata, in GWh."""

    month: Index[str] = pa.Field(unique=True, check_name=True)
    negative_hours: Series[int] = pa.Field(gt=0)
    potential_gwh: Series[float]
    published_gwh: Series[float]
    modulation_gwh: Series[float]
    thin_stratum_gwh: Series[float]

    class Config:
        strict = True


def calibrated_potential(inputs: eligibility.YearInputs, fit: fits.Fit) -> "pd.Series[float]":
    """p* on every hour of the yearly `inputs`: the potential of the frozen `fit` times the ratio
    of its stratum and the annual level of its year, estimated on the eligible hours of the year.
    NaN at an hour without a stratum.

    Raises ValueError if the year has no eligible hour to give its annual level.
    """
    final = fit.final
    published = inputs.published["energy_mwh"]
    potential = calibration.potential_mwh(inputs.arrays, final.calibration)
    with_strata = strata.apply_ratios(final.ratios, potential, inputs.arrays.covariates)
    levels = strata.annual_levels(published, with_strata, inputs.eligibility["eligible"])
    year = pd.DatetimeIndex(published.index)[0].year
    if not levels.get(year, 0.0) > 0:
        raise ValueError(f"{year} has no eligible hour to give its annual level")
    p_star: pd.Series[float] = (with_strata * levels[year]).rename("potential_mwh")
    return p_star


def negative_hours(
    years: Sequence[eligibility.YearInputs], fit: fits.Fit
) -> DataFrame[NegativeHours]:
    """The N hours of the yearly inputs `years`, with the potential of the frozen `fit`.

    Raises ValueError if a year has N hours but no eligible hour to give its annual level.
    """
    final = fit.final
    table = final.ratios.table
    thin_strata = table.loc[table["training_hours"] < THIN_STRATUM_HOURS, strata.STRATUM_KEY]
    frames = []
    for inputs in years:
        negative = eligibility.negative_price_hours(inputs)
        if not negative.any():
            continue
        published = inputs.published["energy_mwh"]
        p_star = calibrated_potential(inputs, fit)
        potential = calibration.potential_mwh(inputs.arrays, final.calibration)
        hour_strata = strata.classify_hours(
            potential, inputs.arrays.covariates, final.ratios.decile_edges
        )
        thin = (
            hour_strata.reset_index()
            .merge(thin_strata, on=strata.STRATUM_KEY, how="left", indicator=True)["_merge"]
            .eq("both")
        )
        frames.append(
            pd.DataFrame(
                {
                    "published_mwh": published,
                    "potential_mwh": p_star,
                    "modulation_mwh": p_star - published,
                    "thin_stratum": thin.to_numpy(),
                },
                index=published.index,
            )[negative.to_numpy()]
        )
    if not frames:
        return NegativeHours.empty()
    return NegativeHours.validate(pd.concat(frames))


def months(hours: pd.DataFrame) -> DataFrame[Months]:
    """`Months` of the N hours `hours`, a NegativeHours frame."""
    table = NegativeHours.validate(hours)
    local = pd.DatetimeIndex(table.index).tz_convert(TIMEZONE)
    month = pd.Index(local.strftime("%Y-%m"), name="month")
    by_month = table.groupby(month)
    thin = table["modulation_mwh"].where(table["thin_stratum"], 0.0).groupby(month).sum()
    summary = pd.DataFrame(
        {
            "negative_hours": by_month.size(),
            "potential_gwh": by_month["potential_mwh"].sum() / MWH_PER_GWH,
            "published_gwh": by_month["published_mwh"].sum() / MWH_PER_GWH,
            "modulation_gwh": by_month["modulation_mwh"].sum() / MWH_PER_GWH,
            "thin_stratum_gwh": thin / MWH_PER_GWH,
        }
    )
    return Months.validate(summary)
