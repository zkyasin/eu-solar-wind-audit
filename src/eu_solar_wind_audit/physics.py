"""Uncalibrated physics model of the solar and onshore wind output of mainland France, from ERA5,
the SDES capacity per department and the PV and wind modules.

Output is energy in MWh for each hour [t, t + 1 h), labelled t in UTC: the sum over ERA5 cells of
capacity factor times capacity in MW.
"""

import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from eu_solar_wind_audit import pv, spatial, wind
from eu_solar_wind_audit.config import Country, WindGeneration
from eu_solar_wind_audit.sources import era5, natural_earth, odre, sdes

Technology = Literal["solar", "wind_onshore"]

ERA5_CACHE_DIR = Path("data/raw/era5")
# ERA5T is released about six days after real time (evaluation plan, scope).
ERA5T_LATENCY = pd.Timedelta(days=6)
INPUTS = {
    "solar": ["ssrd", "fdir", "t2m", "u10", "v10"],
    "wind_onshore": ["u10", "v10", "u100", "v100"],
}
WEATHER = ("ssrd", "fdir", "t2m", "u10", "v10", "u100", "v100")
SDES_MEDIA = {"solar": sdes.SOLAR_2026_Q2, "wind_onshore": sdes.WIND_2026_Q2}
HOUR = pd.Timedelta(hours=1)
DAY = pd.Timedelta(days=1)


def era5_available_until() -> pd.Timestamp:
    """First hour, in naive UTC, that ERA5T is not expected to hold yet."""
    return pd.Timestamp.now(tz="UTC").tz_localize(None).floor("D") - ERA5T_LATENCY


def read_weather(
    country: str,
    box: era5.Box,
    variables: Sequence[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    cache_dir: Path = ERA5_CACHE_DIR,
) -> xr.Dataset:
    """ERA5 `variables` (some of `WEATHER`) for hours [start, end) in naive UTC, as
    era5.read_era5 gives them.

    A calendar year that ERA5T already covers is read once, with every field of `WEATHER`, and
    kept in `cache_dir`/<country>/<year>.zarr. The last, incomplete year is read again on every
    call.
    """
    available = era5_available_until()
    parts = []
    for year in range(start.year, (end - HOUR).year + 1):
        year_start, year_end = pd.Timestamp(year, 1, 1), pd.Timestamp(year + 1, 1, 1)
        first, stop = max(start, year_start), min(end, year_end)
        if year_end <= available:
            weather = _cached_year(cache_dir / country.lower() / f"{year}.zarr", box, year)
        else:
            weather = era5.read_era5(variables, box, first, stop)
        parts.append(weather[list(variables)].sel(time=slice(first, stop - HOUR)))
    # A single year stays lazy; concat would load it.
    return parts[0] if len(parts) == 1 else xr.concat(parts, dim="time")


def _cached_year(path: Path, box: era5.Box, year: int) -> xr.Dataset:
    if not path.exists():
        downloaded = era5.read_era5(
            WEATHER, box, pd.Timestamp(year, 1, 1), pd.Timestamp(year + 1, 1, 1)
        )
        partial = path.with_name(path.name + ".part")
        shutil.rmtree(partial, ignore_errors=True)
        downloaded.assign_attrs(box=list(box)).to_zarr(partial, mode="w", consolidated=False)
        partial.rename(path)
    weather: xr.Dataset = xr.open_zarr(path, chunks=None, consolidated=False)
    if tuple(weather.attrs["box"]) != tuple(box):
        raise ValueError(f"{path} holds the box {weather.attrs['box']}, not {list(box)}")
    return weather


def split_by_commissioning(
    capacity: pd.DataFrame, last_years: Sequence[int | None]
) -> list[pd.DataFrame]:
    """Month-end capacity in MW (one column per region) split into the capacity commissioned in
    each window, the windows ending on 31 December of `last_years` (None: no end).

    Capacity up to the level reached at the end of the first window belongs to it, the next layer
    to the second window, and so on, so the parts add up to the capacity. A net decrease removes
    the newest layer first.
    """
    level = pd.Series(0.0, index=capacity.columns)
    parts = []
    for year in last_years:
        boundary = pd.Timestamp(year, 12, 31) if year is not None else None
        if boundary is not None and boundary < capacity.index[0]:
            raise ValueError(f"the capacity panel starts after the end of {year}")
        reached = boundary is None or boundary > capacity.index[-1]
        top = capacity.max() if reached else capacity.loc[:boundary].iloc[-1]
        top = top.clip(lower=level)
        parts.append(capacity.clip(lower=level, upper=top, axis=1) - level)
        level = top
    return parts


def hourly_capacity(per_cell: xr.DataArray, hours: pd.DatetimeIndex) -> xr.DataArray:
    """Capacity at the start of each hour, linear in time between month ends."""
    # A month-end figure is the capacity at the end of that day. xarray interpolates wrongly
    # between datetimes of different units, so both are set to nanoseconds.
    instants = pd.DatetimeIndex(per_cell.indexes["month_end"] + DAY).as_unit("ns")
    by_instant = per_cell.assign_coords(month_end=instants).rename(month_end="time")
    capacity = by_instant.interp(time=hours.as_unit("ns"))
    if capacity.isnull().any():
        raise ValueError(f"no capacity for some hours from {hours[0]} to {hours[-1]}")
    return capacity


def generation_parameters(generation: WindGeneration) -> wind.WindParameters:
    """Wind model of one generation, with the uncalibrated defaults of wind.WindParameters."""
    member = wind.FleetTurbine(generation.turbine, generation.hub_height_m, 1.0)
    return wind.WindParameters(fleet=(member,))


def model(
    technology: Technology,
    start: pd.Timestamp,
    end: pd.Timestamp,
    parameters: Country,
    monthly: pd.DataFrame,
    regions: gpd.GeoDataFrame,
    cache_dir: Path = ERA5_CACHE_DIR,
) -> pd.DataFrame:
    """Hourly modelled output for hours [start, end) in naive UTC.

    `monthly` is a month-end capacity panel as sdes.monthly_capacity returns it and `regions` the
    department polygons of natural_earth.load_admin1. The result, indexed by UTC hour start, has
    the energy in MWh, whether the hour is complete, the modelled capacity in MW, whether that
    capacity uses provisional SDES months, and the ERA5 source of the hour.
    """
    # The only perimeter of Country so far.
    mainland = sdes.metropolitan_without_corsica(monthly)
    mainland = mainland[mainland["technology"] == technology]
    capacity = mainland.pivot(index="month_end", columns="department_code", values="capacity_mw")
    capacity = capacity.rename_axis(index="month_end", columns="region")

    box = parameters.era5_box
    weather = read_weather(parameters.country, box, INPUTS[technology], start, end, cache_dir)
    hours = pd.DatetimeIndex(weather.indexes["time"])
    polygons = regions[regions["region"].isin(capacity.columns)]
    fractions = spatial.region_fractions(polygons, weather["latitude"], weather["longitude"])

    if technology == "solar":
        parts = [capacity]
    else:
        last_years = [generation.last_commissioning_year for generation in parameters.wind_onshore]
        parts = split_by_commissioning(capacity, last_years)
    per_cell = [spatial.capacity_weights(fractions, xr.DataArray(part)) for part in parts]

    # Capacity factors are computed only on cells that ever hold capacity.
    occupied = (xr.concat(per_cell, dim="part") > 0).any(["part", "month_end"]).to_numpy()
    rows, columns = (xr.DataArray(index, dims="cell") for index in np.nonzero(occupied))
    capacities = [
        hourly_capacity(part.isel(latitude=rows, longitude=columns), hours) for part in per_cell
    ]
    cells = weather.load().isel(latitude=rows, longitude=columns)

    energy = xr.zeros_like(cells["time"], dtype=float)
    for index, capacity_mw in enumerate(capacities):
        if technology == "solar":
            factor = pv.capacity_factor(cells, parameters.solar)
        else:
            turbine = generation_parameters(parameters.wind_onshore[index])
            factor = wind.capacity_factor(cells, turbine)
        energy = energy + (factor * capacity_mw).sum("cell", skipna=False)
    total_capacity = sum(capacity_mw.sum("cell").to_numpy() for capacity_mw in capacities)

    last_final = mainland.loc[~mainland["provisional"], "month_end"].max() + DAY
    return pd.DataFrame(
        {
            "energy_mwh": energy.to_numpy(),
            "complete": energy.notnull().to_numpy(),
            "capacity_mw": total_capacity,
            "capacity_provisional": ~(hours <= last_final),
            "era5_source": weather["source"].to_numpy().astype(str),
        },
        index=hours.tz_localize("UTC").rename("time"),
    )


def load_monthly_capacity(technology: Technology, until: pd.Timestamp) -> pd.DataFrame:
    """Month-end SDES capacity per department up to the month end at or after `until`."""
    with odre.make_client() as client:
        workbook = sdes.download_dashboard(client, SDES_MEDIA[technology])
    month_end = (until + pd.offsets.MonthEnd(0)).normalize()
    return sdes.monthly_capacity(sdes.read_capacity(workbook), until=month_end)


def model_france(
    technology: Technology,
    start: pd.Timestamp,
    end: pd.Timestamp,
    parameters: Country,
    cache_dir: Path = ERA5_CACHE_DIR,
) -> pd.DataFrame:
    """Hourly modelled output of mainland France without Corsica for hours [start, end), naive
    UTC; see `model`."""
    monthly = load_monthly_capacity(technology, end - HOUR)
    regions = natural_earth.load_admin1("FR")
    return model(technology, start, end, parameters, monthly, regions, cache_dir)
