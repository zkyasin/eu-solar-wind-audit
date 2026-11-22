"""Uncalibrated physics model of the solar and onshore wind output of mainland France, from ERA5,
the SDES capacity per department and the PV and wind modules.

Output is energy in MWh for each hour [t, t + 1 h), labelled t in UTC: the sum over ERA5 cells of
capacity factor times capacity in MW.
"""

import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, NamedTuple

import geopandas as gpd
import numpy as np
import numpy.typing as npt
import pandas as pd
import pandera.pandas as pa
import xarray as xr
from pandera.typing import DataFrame, Series

from eu_solar_wind_audit import grid, pv, spatial, wind
from eu_solar_wind_audit.config import Country, WindGeneration
from eu_solar_wind_audit.sources import era5, natural_earth, odre, sdes

Technology = Literal["solar", "wind_onshore"]

ERA5_CACHE_DIR = Path("data/raw/era5")
INPUTS = {
    "solar": ["ssrd", "fdir", "t2m", "u10", "v10"],
    "wind_onshore": ["u10", "v10", "u100", "v100"],
}
WEATHER = ("ssrd", "fdir", "t2m", "u10", "v10", "u100", "v100")
SDES_MEDIA = {"solar": sdes.SOLAR_2026_Q2, "wind_onshore": sdes.WIND_2026_Q2}
DAY = pd.Timedelta(days=1)


class Modelled(grid.Hourly):
    """Modelled hour, with the capacity in MW at its start, whether that capacity uses provisional
    SDES months, and the ERA5 release its weather comes from."""

    capacity_mw: Series[float] = pa.Field(ge=0)
    capacity_provisional: Series[bool]
    era5_source: Series[str] = pa.Field(isin=["ERA5", "ERA5T"])


def era5_final_until(today: pd.Timestamp) -> pd.Timestamp:
    """Start of the first month, in naive UTC, that final ERA5 may not cover yet on `today`."""
    # Final ERA5 replaces ERA5T about two months after the month in question (ECMWF, "ERA5: data
    # documentation", confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation).
    return (today - pd.DateOffset(months=2)).to_period("M").to_timestamp()


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

    A calendar year that final ERA5 already covers is read once, with every field of `WEATHER`,
    and kept in `cache_dir`/<country>/<year>.zarr. A later year, which still holds or will hold
    ERA5T hours, is read again on every call, so that the cache never freezes ERA5T values.
    """
    final_until = era5_final_until(pd.Timestamp.now(tz="UTC").tz_localize(None))
    parts = []
    for year in range(start.year, (end - grid.HOUR).year + 1):
        year_start, year_end = pd.Timestamp(year, 1, 1), pd.Timestamp(year + 1, 1, 1)
        first, stop = max(start, year_start), min(end, year_end)
        if year_end <= final_until:
            weather = _cached_year(cache_dir / country.lower() / f"{year}.zarr", box, year)
        else:
            weather = era5.read_era5(variables, box, first, stop)
        parts.append(weather[list(variables)].sel(time=slice(first, stop - grid.HOUR)))
    # A single year stays lazy; concat would load it.
    return parts[0] if len(parts) == 1 else xr.concat(parts, dim="time")


def _cached_year(path: Path, box: era5.Box, year: int) -> xr.Dataset:
    if not path.exists():
        downloaded = era5.read_era5(
            WEATHER, box, pd.Timestamp(year, 1, 1), pd.Timestamp(year + 1, 1, 1)
        )
        # The final release can come later than usual.
        if (downloaded["source"] != "ERA5").any():
            return downloaded
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
    """Capacity at the start of each of `hours`, in naive UTC as ERA5 gives them, linear in time
    between month ends."""
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


class Fleet(NamedTuple):
    """Weather on the cells that hold capacity at some month end of the panel, with dims (time,
    cell), and the capacity in MW at the start of each hour on these cells for each part of the
    fleet: the whole fleet for solar, one commissioning window per wind generation.
    `capacity_provisional` tells, for each hour, whether that capacity uses provisional SDES
    months."""

    cells: xr.Dataset
    capacities: list[xr.DataArray]
    capacity_provisional: npt.NDArray[np.bool_]


def fleet_cells(
    technology: Technology,
    weather: xr.Dataset,
    parameters: Country,
    monthly: pd.DataFrame,
    regions: gpd.GeoDataFrame,
) -> Fleet:
    """Cells of `weather`, as read_weather gives it, where the mainland fleet of `technology`
    stands, with its hourly capacity; `monthly` and `regions` as for `model`."""
    # The only perimeter of Country so far.
    mainland = sdes.metropolitan_without_corsica(monthly)
    mainland = mainland[mainland["technology"] == technology]
    capacity = mainland.pivot(index="month_end", columns="department_code", values="capacity_mw")
    capacity = capacity.rename_axis(index="month_end", columns="region")
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
    hours = pd.DatetimeIndex(weather.indexes["time"])
    capacities = [
        hourly_capacity(part.isel(latitude=rows, longitude=columns), hours) for part in per_cell
    ]
    # Selecting cells on the lazy zarr store peaks at about four times the memory of loading first.
    cells = weather.load().isel(latitude=rows, longitude=columns)
    last_final = mainland.loc[~mainland["provisional"], "month_end"].max() + DAY
    return Fleet(cells, capacities, ~(hours <= last_final))


def model(
    technology: Technology,
    start: pd.Timestamp,
    end: pd.Timestamp,
    parameters: Country,
    monthly: pd.DataFrame,
    regions: gpd.GeoDataFrame,
    cache_dir: Path = ERA5_CACHE_DIR,
) -> DataFrame[Modelled]:
    """Hourly modelled output for hours [start, end), both hour starts in UTC.

    `monthly` is a month-end capacity panel as sdes.monthly_capacity returns it and `regions` the
    department polygons of natural_earth.load_admin1.
    """
    grid.check_period(start, end)
    # ERA5 times are naive UTC.
    weather = read_weather(
        parameters.country,
        parameters.era5_box,
        INPUTS[technology],
        start.tz_convert(None),
        end.tz_convert(None),
        cache_dir,
    )
    cells, capacities, capacity_provisional = fleet_cells(
        technology, weather, parameters, monthly, regions
    )

    energy = xr.zeros_like(cells["time"], dtype=float)
    for index, capacity_mw in enumerate(capacities):
        if technology == "solar":
            factor = pv.capacity_factor(cells, parameters.solar)
        else:
            turbine = generation_parameters(parameters.wind_onshore[index])
            factor = wind.capacity_factor(cells, turbine)
        energy = energy + (factor * capacity_mw).sum("cell", skipna=False)
    total_capacity = sum(capacity_mw.sum("cell").to_numpy() for capacity_mw in capacities)

    hours = pd.DatetimeIndex(cells.indexes["time"])
    modelled = pd.DataFrame(
        {
            "energy_mwh": energy.to_numpy(),
            "complete": energy.notnull().to_numpy(),
            "capacity_mw": total_capacity,
            "capacity_provisional": capacity_provisional,
            "era5_source": cells["source"].to_numpy().astype(str),
        },
        # ERA5 times read outside the zarr cache are not always in nanoseconds.
        index=hours.tz_localize("UTC").as_unit("ns").rename("time"),
    )
    return Modelled.validate(modelled)


def load_monthly_capacity(technology: Technology, until: pd.Timestamp) -> pd.DataFrame:
    """Month-end SDES capacity per department up to the month end at or after `until`, in UTC."""
    with odre.make_client() as client:
        workbook = sdes.download_dashboard(client, SDES_MEDIA[technology])
    # SDES month ends are naive dates.
    month_end = (until.tz_convert(None) + pd.offsets.MonthEnd(0)).normalize()
    return sdes.monthly_capacity(sdes.read_capacity(workbook), until=month_end)


def model_france(
    technology: Technology,
    start: pd.Timestamp,
    end: pd.Timestamp,
    parameters: Country,
    cache_dir: Path = ERA5_CACHE_DIR,
) -> DataFrame[Modelled]:
    """Hourly modelled output of mainland France without Corsica for hours [start, end), both
    hour starts in UTC, computed one calendar year at a time; see `model`."""
    grid.check_period(start, end)
    monthly = load_monthly_capacity(technology, end - grid.HOUR)
    regions = natural_earth.load_admin1("FR")
    years = []
    for year in range(start.year, (end - grid.HOUR).year + 1):
        first = max(start, pd.Timestamp(year, 1, 1, tz="UTC"))
        stop = min(end, pd.Timestamp(year + 1, 1, 1, tz="UTC"))
        years.append(model(technology, first, stop, parameters, monthly, regions, cache_dir))
    return Modelled.validate(pd.concat(years))
