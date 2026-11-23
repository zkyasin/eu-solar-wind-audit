"""The axis-4 estimate of mainland France against the monthly modulation volumes that RTE
publishes (evaluation plan, axis 4, "France against RTE").

This is the only module that reads the reference series: the calibration and the estimate never
do (tests/test_calibration_isolation.py). The RTE series is itself an estimate, so agreement is a
replication, not a proof of accuracy. Its perimeter includes Corsica, which the éCO2mix series
leave out.

The tests of the plan, without the verdict, which needs the bootstrap interval and the minimal
detectable effect:
- annual ratio: M-hat / RTE within RATIO_RANGE in each year of RATIO_YEARS;
- monthly detail: on months with more than MONTH_FROM_GWH at RTE and at least MONTH_FROM_HOURS N
  hours, the mean absolute log error of M-hat is lower than that of the prorata baseline, the
  potential at N hours times a rate estimated on the other years (leave-one-year-out);
- monthly rates: on the same months, the Spearman correlation of the estimated and RTE rates
  (volume over the potential at N hours) reaches SPEARMAN_FROM.
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
