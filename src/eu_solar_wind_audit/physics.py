"""Uncalibrated physics model of the solar and onshore wind output of mainland France and of its
12 éCO2mix regions, and of Germany, from ERA5, the PV and wind modules and the capacity of each
area: the SDES capacity per department in France, the MaStR capacity per district in Germany.

Output is energy in MWh for each hour [t, t + 1 h), labelled t in UTC: the sum over ERA5 cells of
capacity factor times capacity in MW.
"""

import json
import shutil
from collections.abc import Iterable, Iterator, Sequence
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
from eu_solar_wind_audit.sources import bkg, era5, mastr, natural_earth, odre, sdes

Technology = Literal["solar", "wind_onshore"]

ERA5_CACHE_DIR = Path("data/raw/era5")
INPUTS = {
    "solar": ["ssrd", "fdir", "t2m", "u10", "v10"],
    "wind_onshore": ["u10", "v10", "u100", "v100"],
}
WEATHER = ("ssrd", "fdir", "t2m", "u10", "v10", "u100", "v100")
SDES_MEDIA = {"solar": sdes.SOLAR_2026_Q2, "wind_onshore": sdes.WIND_2026_Q2}
DAY = pd.Timedelta(days=1)
# Units must be registered in MaStR within a month of their commissioning (MaStRV section 5(5)),
# and some are registered later, so the last months of an export are provisional.
MASTR_PROVISIONAL_MONTHS = 3
# The German panel starts a year before the evaluation period, for the margins of its runs.
MASTR_PANEL_START = pd.Timestamp("2018-01-31")


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
    """Cells of `weather`, as read_weather gives it, where the fleet of `technology` in the
    perimeter of `parameters` stands, with its hourly capacity; `monthly` and `regions` as for
    `model`."""
    mainland = _in_perimeter(monthly, parameters)
    mainland = mainland[mainland["technology"] == technology]
    by_area = mainland.groupby(["month_end", "department_code"])["capacity_mw"].sum()
    capacity = by_area.unstack().rename_axis(index="month_end", columns="region")
    polygons = regions[regions["region"].isin(capacity.columns)]
    fractions = spatial.region_fractions(polygons, weather["latitude"], weather["longitude"])

    if technology == "solar":
        parts = [capacity]
    elif "generation" in mainland.columns:
        # Units of known commissioning dates give the capacity of each generation.
        parts = [
            mainland[mainland["generation"] == generation]
            .pivot(index="month_end", columns="department_code", values="capacity_mw")
            .reindex(index=capacity.index, columns=capacity.columns, fill_value=0.0)
            .rename_axis(index="month_end", columns="region")
            for generation in range(len(parameters.wind_onshore))
        ]
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

    `monthly` is a month-end capacity panel as sdes.monthly_capacity returns it, or as
    load_germany_capacity gives it, and `regions` the polygons of its areas: the departments of
    natural_earth.load_admin1 or the districts of bkg.load_districts.
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
    fleet = fleet_cells(technology, weather, parameters, monthly, regions)
    return _modelled(fleet, _capacity_factors(technology, fleet.cells, parameters))


def model_regions(
    technology: Technology,
    start: pd.Timestamp,
    end: pd.Timestamp,
    parameters: Country,
    monthly: pd.DataFrame,
    regions: gpd.GeoDataFrame,
    cache_dir: Path = ERA5_CACHE_DIR,
) -> dict[str, DataFrame[Modelled]]:
    """Hourly modelled output of each éCO2mix region for hours [start, end), both hour starts in
    UTC, keyed by the codes of odre.REGIONS in its order: the output of `model` for the capacity
    of the departments of the region. `monthly` and `regions` are as for `model`; the regions of
    `monthly` are named as in odre.REGIONS.

    The weather is read and the capacity factors of the mainland cells are computed once: regions
    differ only in their capacity, and they add up to the output of `model`. A region without
    capacity has zero output at every hour.
    """
    grid.check_period(start, end)
    unknown = set(sdes.metropolitan_without_corsica(monthly)["region"]) - set(odre.REGIONS.values())
    if unknown:
        # Their capacity would be in the mainland and in no region.
        raise ValueError(f"mainland regions not in odre.REGIONS: {sorted(unknown)}")
    # ERA5 times are naive UTC. Loaded once, as every region selects its cells from it.
    weather = read_weather(
        parameters.country,
        parameters.era5_box,
        INPUTS[technology],
        start.tz_convert(None),
        end.tz_convert(None),
        cache_dir,
    ).load()
    cells = fleet_cells(technology, weather, parameters, monthly, regions).cells
    factors = list(_capacity_factors(technology, cells, parameters))
    mainland = _cell_index(cells)
    by_region = {}
    for code, name in odre.REGIONS.items():
        # Other regions keep their rows at zero capacity, so that a region needs no rows of its
        # own to get a fleet.
        capacity_mw = monthly["capacity_mw"].where(monthly["region"] == name, 0.0)
        fleet = fleet_cells(
            technology, weather, parameters, monthly.assign(capacity_mw=capacity_mw), regions
        )
        # A cell with capacity in the region has capacity in the mainland too.
        positions = mainland.get_indexer(_cell_index(fleet.cells))
        by_region[code] = _modelled(fleet, [factor.isel(cell=positions) for factor in factors])
    return by_region


def _cell_index(cells: xr.Dataset) -> pd.MultiIndex:
    """Latitude and longitude of each cell of a Fleet."""
    return pd.MultiIndex.from_arrays([cells["latitude"].to_numpy(), cells["longitude"].to_numpy()])


def _capacity_factors(
    technology: Technology, cells: xr.Dataset, parameters: Country
) -> Iterator[xr.DataArray]:
    """Capacity factor on `cells` of each part of the fleet, in the order of Fleet.capacities."""
    # Yielded one by one, so that `model` never holds the three wind generations at once, which
    # raises its peak memory over a year by about 70 MB.
    if technology == "solar":
        yield pv.capacity_factor(cells, parameters.solar)
    else:
        for generation in parameters.wind_onshore:
            yield wind.capacity_factor(cells, generation_parameters(generation))


def _modelled(fleet: Fleet, factors: Iterable[xr.DataArray]) -> DataFrame[Modelled]:
    """Output of `fleet`, given the capacity factor of each of its parts on its cells."""
    energy = xr.zeros_like(fleet.cells["time"], dtype=float)
    for factor, capacity_mw in zip(factors, fleet.capacities, strict=True):
        energy = energy + (factor * capacity_mw).sum("cell", skipna=False)
    total_capacity = sum(capacity_mw.sum("cell").to_numpy() for capacity_mw in fleet.capacities)

    hours = pd.DatetimeIndex(fleet.cells.indexes["time"])
    modelled = pd.DataFrame(
        {
            "energy_mwh": energy.to_numpy(),
            "complete": energy.notnull().to_numpy(),
            "capacity_mw": total_capacity,
            "capacity_provisional": fleet.capacity_provisional,
            "era5_source": fleet.cells["source"].to_numpy().astype(str),
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


def load_germany_capacity(
    technology: Technology,
    parameters: Country,
    units_path: Path | None = None,
    control_area: str | None = None,
) -> pd.DataFrame:
    """Month-end MaStR capacity per district in the columns of sdes.monthly_capacity, the district
    code standing for the department code and the code of its Land for the region, with the
    `generation` of the capacity: for onshore wind, the window of parameters.wind_onshore in which
    its units were commissioned, 0 for solar. The capacity is the net power, limited by the
    inverters, as SDES gives AC capacity. The last MASTR_PROVISIONAL_MONTHS months before the
    export are provisional.

    `units_path` is the unit table that mastr.write_capacity writes, extracted from the pinned
    export when it is missing. With a `control_area`, the capacity is that of its units only, on
    the rows of Germany.
    """
    path = units_path or mastr.OUTPUT_DIR / "mastr_units.parquet"
    if not path.exists():
        mastr.extract_capacity(output_dir=path.parent)
    export_date = pd.Timestamp(json.loads(path.with_suffix(".json").read_text())["export_date"])
    last = export_date - pd.offsets.MonthEnd(1)
    units = pd.read_parquet(path)
    units = units[units["technology"].eq(technology)]
    if control_area is not None:
        # Units of other control areas count with no power, so that the panel keeps the rows of
        # Germany and a control area without units has no capacity.
        others = units["control_area"].ne(control_area)
        units = units.assign(
            net_kw=units["net_kw"].mask(others, 0.0), gross_kw=units["gross_kw"].mask(others, 0.0)
        )
    year = units["commissioned"].dt.year.to_numpy()
    windows = (
        [None]
        if technology == "solar"
        else [generation.last_commissioning_year for generation in parameters.wind_onshore]
    )
    ends = [np.inf if window is None else float(window) for window in windows]
    frames = []
    for generation, (low, high) in enumerate(zip([-np.inf, *ends], ends, strict=False)):
        window = units[(year > low) & (year <= high)]
        panel = mastr.monthly_capacity(window, MASTR_PANEL_START, last)
        frames.append(panel[panel["district"].notna()].assign(generation=generation))
    rows = pd.concat(frames, ignore_index=True)
    return pd.DataFrame(
        {
            "department_code": rows["district"],
            "department": rows["district"],
            "region": rows["district"].str[:2],
            "technology": technology,
            "month_end": rows["month_end"],
            "capacity_mw": rows["capacity_mw_ac"],
            "provisional": rows["month_end"] > last - pd.offsets.MonthEnd(MASTR_PROVISIONAL_MONTHS),
            "generation": rows["generation"],
        }
    )


def model_germany(
    technology: Technology,
    start: pd.Timestamp,
    end: pd.Timestamp,
    parameters: Country,
    cache_dir: Path = ERA5_CACHE_DIR,
    units_path: Path | None = None,
    control_area: str | None = None,
) -> DataFrame[Modelled]:
    """Hourly modelled output of the onshore fleet of Germany, or of its units in `control_area`,
    for hours [start, end), both hour starts in UTC, computed one calendar year at a time; see
    `model`."""
    grid.check_period(start, end)
    monthly = load_germany_capacity(technology, parameters, units_path, control_area)
    districts = bkg.load_districts()
    years = [
        model(technology, first, stop, parameters, monthly, districts, cache_dir)
        for first, stop in _calendar_years(start, end)
    ]
    return Modelled.validate(pd.concat(years))


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
    years = [
        model(technology, first, stop, parameters, monthly, regions, cache_dir)
        for first, stop in _calendar_years(start, end)
    ]
    return Modelled.validate(pd.concat(years))


def model_france_regions(
    technology: Technology,
    start: pd.Timestamp,
    end: pd.Timestamp,
    parameters: Country,
    cache_dir: Path = ERA5_CACHE_DIR,
) -> dict[str, DataFrame[Modelled]]:
    """Hourly modelled output of each éCO2mix region for hours [start, end), both hour starts in
    UTC, keyed by the codes of odre.REGIONS, computed one calendar year at a time; see
    `model_regions` and `model_france`."""
    grid.check_period(start, end)
    monthly = load_monthly_capacity(technology, end - grid.HOUR)
    regions = natural_earth.load_admin1("FR")
    years = [
        model_regions(technology, first, stop, parameters, monthly, regions, cache_dir)
        for first, stop in _calendar_years(start, end)
    ]
    return {
        code: Modelled.validate(pd.concat(year[code] for year in years)) for code in odre.REGIONS
    }


def _in_perimeter(monthly: pd.DataFrame, parameters: Country) -> pd.DataFrame:
    """Rows of `monthly` in the perimeter of `parameters`: mainland France without Corsica, or
    every district of Germany."""
    if parameters.perimeter == "metropolitan_without_corsica":
        return sdes.metropolitan_without_corsica(monthly)
    return monthly


def _calendar_years(
    start: pd.Timestamp, end: pd.Timestamp
) -> Iterator[tuple[pd.Timestamp, pd.Timestamp]]:
    """Periods [first, stop) that cut the hours [start, end), both hour starts in UTC, at each new
    year."""
    for year in range(start.year, (end - grid.HOUR).year + 1):
        first = max(start, pd.Timestamp(year, 1, 1, tz="UTC"))
        stop = min(end, pd.Timestamp(year + 1, 1, 1, tz="UTC"))
        yield first, stop
