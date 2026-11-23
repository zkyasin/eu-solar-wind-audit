"""The axis-4 estimate of mainland France against the monthly modulation volumes that RTE
publishes (evaluation plan, axis 4, "France against RTE").

This is the only module that reads the reference series: the calibration and the estimate never
do (tests/test_calibration_isolation.py). The RTE series is itself an estimate, so agreement is a
replication, not a proof of accuracy. Its perimeter includes Corsica, which the éCO2mix series
leave out. The plan's check of the wind perimeter passes: RTE describes its wind series as the
modulation of onshore wind farms (page read on 2026-11-23; references/rte/manifest.json), the
perimeter of the wind_onshore estimate.

The tests of the plan:
- annual ratio: M-hat / RTE within RATIO_RANGE in each year of RATIO_YEARS;
- monthly detail: on months with more than MONTH_FROM_GWH at RTE and at least MONTH_FROM_HOURS N
  hours, the mean absolute log error of M-hat is lower than that of the prorata baseline, the
  potential at N hours times a rate estimated on the other years (leave-one-year-out);
- monthly rates: on the same months, the Spearman correlation of the estimated and RTE rates
  (volume over the potential at N hours) reaches SPEARMAN_FROM.

The verdict, checked in this order (evaluation plan and Deviations):
1. inconclusive: in a year of RATIO_YEARS, the half-width of the 90 % interval exceeds
   INCONCLUSIVE_HALF_WIDTH of the estimate, or the estimate is below the minimal detectable
   effect;
2. discordant: the annual ratio test fails;
3. concordant in volume only: the annual ratio test passes and a monthly test fails;
4. concordant: all three tests pass.
A year whose minimal detectable effect exceeds LIKELY_INCONCLUSIVE_SHARE of the RTE volume is
announced as likely inconclusive.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from eu_solar_wind_audit import modulation, physics

RATIO_RANGE = (0.8, 1.25)
RATIO_YEARS = (2024, 2025)
MONTH_FROM_GWH = 20.0
MONTH_FROM_HOURS = 30
SPEARMAN_FROM = 0.6
INCONCLUSIVE_HALF_WIDTH = 0.25
LIKELY_INCONCLUSIVE_SHARE = 0.2
# The names of the technologies in the RTE files (references.RTE_FILES).
RTE_TECHNOLOGY: dict[physics.Technology, str] = {"solar": "solar", "wind_onshore": "wind"}


@dataclass(frozen=True)
class Comparison:
    """The months of both series, with the baseline and whether each month is tested; the years,
    with the annual ratio; the mean absolute log errors of M-hat and of the baseline on the
    tested months, the Spearman correlation of their rates, and the outcome of each test."""

    months: pd.DataFrame
    years: pd.DataFrame
    log_error: float
    baseline_log_error: float
    spearman: float
    tests: dict[str, bool]


def compare(
    estimated: pd.DataFrame, reference: pd.DataFrame, technology: physics.Technology
) -> Comparison:
    """Compare `estimated`, a modulation.Months frame, with `reference`, the RTE volumes as
    references.load_rte_modulation gives them, for `technology`, on the months of both."""
    months = modulation.Months.validate(estimated)
    rte = reference[
        (reference["technology"] == RTE_TECHNOLOGY[technology]) & (reference["block"] == "total")
    ]
    rte_gwh = pd.Series(rte["gwh"].to_numpy(), index=rte["month"].astype(str), name="rte_gwh")
    table = months.join(rte_gwh, how="inner")
    year = table.index.str[:4].astype(int)
    by_year = table.groupby(year)[["rte_gwh", "potential_gwh"]].sum()
    # The rate of each year from the other years: the sums of all years less its own.
    others = by_year.sum() - by_year
    rate = (others["rte_gwh"] / others["potential_gwh"]).reindex(year).to_numpy()
    table["baseline_gwh"] = rate * table["potential_gwh"]
    table["tested"] = (table["rte_gwh"] > MONTH_FROM_GWH) & (
        table["negative_hours"] >= MONTH_FROM_HOURS
    )

    years = table.groupby(year)[["modulation_gwh", "rte_gwh", "baseline_gwh"]].sum()
    years["ratio"] = years["modulation_gwh"] / years["rte_gwh"]
    low, high = RATIO_RANGE
    years["within"] = years["ratio"].between(low, high)

    tested = table[table["tested"]]
    log_error = _mean_log_error(tested["modulation_gwh"], tested["rte_gwh"])
    baseline_log_error = _mean_log_error(tested["baseline_gwh"], tested["rte_gwh"])
    rates = tested[["modulation_gwh", "rte_gwh"]].div(tested["potential_gwh"], axis="index")
    spearman = (
        float(spearmanr(rates["modulation_gwh"], rates["rte_gwh"]).statistic)
        if len(rates) > 1
        else float("nan")
    )
    ratio_years = years.reindex(list(RATIO_YEARS))
    tests = {
        "annual_ratio": bool(ratio_years["within"].fillna(False).astype(bool).all()),
        "monthly_detail": bool(log_error < baseline_log_error),
        "monthly_rates": bool(spearman >= SPEARMAN_FROM),
    }
    return Comparison(table, years, log_error, baseline_log_error, spearman, tests)


def _mean_log_error(estimate: pd.Series, reference: pd.Series) -> float:
    """Mean absolute log ratio of `estimate` to `reference`; infinite if an estimate is not
    positive, since it has no log ratio."""
    if not (estimate > 0).all():
        return float("inf")
    return float(np.abs(np.log(estimate / reference)).mean())


@dataclass(frozen=True)
class Verdict:
    """The verdict of a technology, and for each year of RATIO_YEARS the checks of its first rule:
    the estimate, its 90 % interval and the share of the estimate that its half-width makes, the
    minimal detectable effect, the RTE volume, whether the year is announced as likely
    inconclusive, and whether it is inconclusive."""

    years: pd.DataFrame
    verdict: str


def verdict(found: Comparison, intervals: pd.DataFrame, mde_gwh: pd.Series) -> Verdict:
    """The verdict of the module docstring for the comparison `found`, with `intervals`, the
    90 % interval and the estimate of each period as estimate-modulation writes them, and the
    minimal detectable effect of each year, `mde_gwh`; both are indexed by period, `YYYY` for a
    year. A year without an estimate, interval or minimal detectable effect is inconclusive."""
    years = [str(year) for year in RATIO_YEARS]
    table = intervals.reindex(years)[["estimate_gwh", "low_gwh", "high_gwh"]].copy()
    table["half_width_share"] = (table["high_gwh"] - table["low_gwh"]) / 2 / table["estimate_gwh"]
    table["mde_gwh"] = mde_gwh.reindex(years)
    table["rte_gwh"] = found.years["rte_gwh"].reindex(list(RATIO_YEARS)).to_numpy()
    table["likely_inconclusive"] = ~(
        table["mde_gwh"] <= LIKELY_INCONCLUSIVE_SHARE * table["rte_gwh"]
    )
    table["inconclusive"] = ~(table["half_width_share"] <= INCONCLUSIVE_HALF_WIDTH) | ~(
        table["estimate_gwh"] >= table["mde_gwh"]
    )
    table.index.name = "year"
    if table["inconclusive"].any():
        outcome = "inconclusive"
    elif not found.tests["annual_ratio"]:
        outcome = "discordant"
    elif not (found.tests["monthly_detail"] and found.tests["monthly_rates"]):
        outcome = "concordant in volume only"
    else:
        outcome = "concordant"
    return Verdict(table, outcome)
