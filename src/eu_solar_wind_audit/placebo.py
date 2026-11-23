"""Contemporaneous placebo and injection-recovery of the modulation estimate (evaluation plan,
axis 4, and Deviations).

The candidates of a year are its eligible P hours with a mean day-ahead price in PLACEBO_PRICES.
Each N hour, in time order, is matched without replacement to the candidate of the same year with
the same Europe/Paris month, hour and day type, nearest in the capacity factor of p*, the
calibrated potential of the frozen fit, then in the national consumption, each over its standard
deviation among the candidates of the year. An N hour without a candidate in its cell stays
unmatched. The matched hours, treated as N hours by bootstrap.replicate, should show no gap: the
placebo passes when the 90 % interval of their gap lies within PLACEBO_SHARE of their potential.

Injection lowers the published output of the matched hours by INJECTED_SHARE. These hours never
enter the fit, so the estimate moves by exactly the injected volume.
"""

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from eu_solar_wind_audit import days, eligibility, fits, modulation

PLACEBO_PRICES = (5.0, 15.0)
PLACEBO_SHARE = 0.02
INJECTED_SHARE = 0.3


@dataclass(frozen=True)
class Matching:
    """The matched hours, flagged on the hours of the yearly inputs; the placebo hour of each
    matched N hour; the p* of each hour; and the number of N hours of each year with and without
    a match."""

    targets: pd.Series
    pairs: pd.Series
    potential_mwh: pd.Series
    counts: pd.DataFrame


def match(
    years: Sequence[eligibility.YearInputs], fit: fits.Fit, consumption_mwh: pd.Series
) -> Matching:
    """The placebo hours of the module docstring for the yearly inputs `years`, with the frozen
    `fit` and the national consumption `consumption_mwh` in MWh on their hours. Years without N
    hours get no placebo hour."""
    flags, potentials, counts = [], [], []
    pairs: dict[pd.Timestamp, pd.Timestamp] = {}
    for inputs in years:
        hours = pd.DatetimeIndex(inputs.published.index)
        negative = eligibility.negative_price_hours(inputs)
        if not negative.any():
            flags.append(pd.Series(False, index=hours))
            potentials.append(pd.Series(np.nan, index=hours))
            continue
        p_star = modulation.calibrated_potential(inputs, fit)
        price = inputs.price_classes["mean_price"]
        low, high = PLACEBO_PRICES
        candidate = inputs.eligibility["eligible"] & (price >= low) & (price < high)
        local = hours.tz_convert(modulation.TIMEZONE)
        features = pd.DataFrame(
            {
                "month": local.month,
                "hour": local.hour,
                "day_type": days.day_types(hours).to_numpy(),
                "factor": (p_star / inputs.arrays.covariates["capacity_mw"]).to_numpy(),
                "consumption": consumption_mwh.reindex(hours).to_numpy(),
            },
            index=hours,
        )
        measured = features[["factor", "consumption"]].notna().all(axis="columns")
        pool = features[candidate.to_numpy() & measured.to_numpy()]
        scale = pool[["factor", "consumption"]].std()
        picked: list[pd.Timestamp] = []
        unmatched = 0
        for hour in hours[negative.to_numpy()]:
            wanted = features.loc[hour]
            cell = pool[
                (pool["month"] == wanted["month"])
                & (pool["hour"] == wanted["hour"])
                & (pool["day_type"] == wanted["day_type"])
                & ~pool.index.isin(picked)
            ]
            # NaN when the N hour lacks a capacity factor or a consumption.
            distance = (
                ((cell[["factor", "consumption"]] - wanted[["factor", "consumption"]]) / scale) ** 2
            ).sum(axis="columns", min_count=2)
            if distance.dropna().empty:
                unmatched += 1
                continue
            picked.append(pd.DatetimeIndex(distance.index)[int(np.nanargmin(distance))])
            pairs[hour] = picked[-1]
        chosen = pd.Series(hours.isin(picked), index=hours)
        flags.append(chosen)
        potentials.append(p_star)
        counts.append({"year": hours[0].year, "matched": int(chosen.sum()), "unmatched": unmatched})
    table = pd.DataFrame(counts, columns=["year", "matched", "unmatched"]).set_index("year")
    paired = pd.Series(list(pairs.values()), index=pd.DatetimeIndex(list(pairs), name="time"))
    return Matching(pd.concat(flags), paired.rename("placebo_hour"), pd.concat(potentials), table)


def inject(
    years: Sequence[eligibility.YearInputs], targets: pd.Series
) -> list[eligibility.YearInputs]:
    """`years` with the published output of the hours of `targets` lowered by INJECTED_SHARE."""
    lowered = []
    for inputs in years:
        published = inputs.published.copy()
        flagged = targets.reindex(published.index, fill_value=False).to_numpy()
        published.loc[flagged, "energy_mwh"] *= 1 - INJECTED_SHARE
        lowered.append(dataclasses.replace(inputs, published=published))
    return lowered
