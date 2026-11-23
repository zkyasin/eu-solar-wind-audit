"""Synthetic yearly calibration inputs for the tests of folds and of the calibration command:
eligibility.YearInputs built from random precompute arrays and a known truth."""

import numpy as np
import pandas as pd
from pandera.typing import DataFrame

from eu_solar_wind_audit import days, eligibility, physics, precompute, regimes
from eu_solar_wind_audit.config import WindGeneration
from eu_solar_wind_audit.pv import PVParameters, System
from eu_solar_wind_audit.sources import odre

CAPACITY_MW = 100.0
PV = PVParameters(fleet=(System(35.0, 180.0, 0.25), System(15.0, 110.0, 0.75)), dc_ac_ratio=1.2)
# Level, horizontal share and seasonal amplitude of the published solar output.
SOLAR_TRUTH = (0.85, 0.55, 0.04)
GENERATIONS = (
    WindGeneration("VestasV82_1.65MW_82", 80.0, 2012),
    WindGeneration("IEA_Reference_3.4MW_130", 100.0),
)
WIND_TRUTH = 0.88


def utc_hours(year: int, step: int = 5) -> pd.DatetimeIndex:
    """Every `step`-th hour of the UTC year `year`, counted back from its last hour, so that the
    hours go through every hour of the day in every month."""
    hours = pd.date_range(
        f"{year}-01-01", f"{year + 1}-01-01", freq="h", inclusive="left", tz="UTC", unit="ns"
    )
    return hours[::-step][::-1].rename("time")


def covariates(
    hours: pd.DatetimeIndex, rng: np.random.Generator
) -> DataFrame[precompute.Covariates]:
    """100 MW on `hours`, with the sun above the horizon from 07:00 to 17:00 UTC and a fleet
    clear-sky index drawn for each Europe/Paris day."""
    _, day = np.unique(days.local_days(hours), return_inverse=True)
    frame = pd.DataFrame(
        {
            "complete": True,
            "capacity_mw": CAPACITY_MW,
            "capacity_provisional": False,
            "era5_source": "ERA5",
            "sun_elevation_deg": 60.0 - 10.0 * np.abs(hours.hour.to_numpy() - 12),
            "ghi_w_m2": rng.uniform(40.0, 110.0, day.max() + 1)[day],
            "clear_sky_ghi_w_m2": 100.0,
            "air_temperature_c": 20.0,
        },
        index=hours,
    )
    return precompute.Covariates.validate(frame)


def year_inputs(
    arrays: precompute.SolarArrays | precompute.WindArrays,
    truth_mwh: pd.Series,
    rng: np.random.Generator,
) -> eligibility.YearInputs:
    """Inputs on the hours of `arrays`: the published output is `truth_mwh` with a noise of 0.01
    in capacity factor, definitive, except where odre.onshore_hourly has no onshore wind, and one
    hour in 50 has a negative price."""
    hours = pd.DatetimeIndex(truth_mwh.index)
    published = pd.DataFrame(
        {
            "energy_mwh": truth_mwh + rng.normal(0.0, 0.01 * CAPACITY_MW, hours.size),
            "complete": True,
            "status": "definitive",
        },
        index=hours,
    )
    if isinstance(arrays, precompute.WindArrays):
        offshore = (hours >= odre.FIRST_OFFSHORE_POWER) & (hours < odre.WIND_SPLIT_START)
        published.loc[offshore, "energy_mwh"] = np.nan
        published.loc[offshore, "complete"] = False
        published.loc[offshore, "status"] = None
    prices = pd.Series(50.0, index=pd.date_range(hours[0], hours[-1], freq="h"))
    prices[hours[::50]] = -5.0
    classes = regimes.classify_hours(prices).reindex(hours)
    technology: physics.Technology = (
        "solar" if isinstance(arrays, precompute.SolarArrays) else "wind_onshore"
    )
    rules = eligibility.eligible_hours(technology, published, arrays.covariates, classes)
    return eligibility.YearInputs(odre.Hourly.validate(published), classes, arrays, rules)


def solar_year(year: int, step: int = 5) -> eligibility.YearInputs:
    rng = np.random.default_rng(year)
    hours = utc_hours(year, step)
    arrays = precompute.SolarArrays(
        fleet_mwh=rng.uniform(0.0, 80.0, (hours.size, 2, precompute.DC_AC_GRID.size)),
        horizontal_mwh=rng.uniform(0.0, 80.0, (hours.size, precompute.DC_AC_GRID.size)),
        shares=np.array([system.share for system in PV.fleet]),
        covariates=covariates(hours, rng),
        centroid=(46.0, 2.0),
    )
    return year_inputs(arrays, precompute.solar_energy(arrays, PV.dc_ac_ratio, *SOLAR_TRUTH), rng)


def wind_year(year: int) -> eligibility.YearInputs:
    """Each generation holds 50 MW, spread over 20 cells whose hub-height speeds scatter around
    a mean speed of the hour."""
    rng = np.random.default_rng(year)
    hours = utc_hours(year)
    cells = 20
    mean_speed = rng.uniform(2.0, 14.0, hours.size)
    capacity_mw = np.zeros((len(GENERATIONS), hours.size, precompute.SPEED_CENTRES.size))
    for generation in capacity_mw:
        speeds = mean_speed[:, np.newaxis] * rng.lognormal(0.0, 0.25, (hours.size, cells))
        bins = np.digitize(np.minimum(speeds, 39.0), precompute.SPEED_EDGES) - 1
        rows = np.repeat(np.arange(hours.size), cells)
        np.add.at(generation, (rows, bins.ravel()), CAPACITY_MW / len(GENERATIONS) / cells)
    arrays = precompute.WindArrays(
        capacity_mw=capacity_mw, generations=GENERATIONS, covariates=covariates(hours, rng)
    )
    return year_inputs(arrays, precompute.wind_energy(arrays, WIND_TRUTH), rng)
