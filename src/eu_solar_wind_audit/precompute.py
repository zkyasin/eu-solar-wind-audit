"""Physics of mainland France computed once for a period, so that calibration can evaluate the
solar output for many parameter values without running physics.model again. The arrays are kept
in memory only, so they always match the code and inputs that produced them.

Solar: national AC energy of the whole fleet mounted as each configured system, and as horizontal
modules, for every DC/AC ratio of DC_AC_GRID. The calibration parameters act on AC output, after
the inverter, so evaluating them costs a few operations on these sums.
"""

from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import numpy.typing as npt
import pandas as pd
import pandera.pandas as pa
import pvlib
from pandera.typing import DataFrame, Index, Series

from eu_solar_wind_audit import grid, physics, pv
from eu_solar_wind_audit.config import Country

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]

DC_AC_GRID = np.round(np.linspace(0.90, 1.80, 91), 2)
HORIZONTAL = pv.System(tilt=0.0, azimuth=180.0, share=1.0)
# Day of year of the June solstice in a common year, where the seasonal term of solar is zero.
SOLSTICE_DAY = 172
YEAR_DAYS = 365.25


class Covariates(pa.DataFrameModel):
    """Hourly values that calibration needs besides the output, for the hour starting at `time`:
    whether every weather input of the hour is present, the capacity in MW at its start and
    whether it uses provisional SDES months, the ERA5 release, and the apparent sun elevation and
    the global and clear-sky horizontal irradiance in W m-2 seen by the fleet at mid-hour."""

    time: Index[grid.UtcTime] = pa.Field(unique=True, check_name=True)
    complete: Series[bool]
    capacity_mw: Series[float] = pa.Field(ge=0)
    capacity_provisional: Series[bool]
    era5_source: Series[str] = pa.Field(isin=["ERA5", "ERA5T"])
    sun_elevation_deg: Series[float] = pa.Field(nullable=True)
    ghi_w_m2: Series[float] = pa.Field(nullable=True)
    clear_sky_ghi_w_m2: Series[float] = pa.Field(nullable=True)

    class Config:
        strict = True


@dataclass(frozen=True)
class SolarArrays:
    """National AC energy in MWh over each hour of `covariates`, for each DC/AC ratio of
    DC_AC_GRID: `fleet_mwh`, shape (time, system, ratio), of the whole fleet mounted as each
    configured system, whose shares are `shares`; `horizontal_mwh`, shape (time, ratio), of the
    whole fleet as horizontal modules."""

    fleet_mwh: FloatArray
    horizontal_mwh: FloatArray
    shares: FloatArray
    covariates: DataFrame[Covariates]


def solar(
    start: pd.Timestamp,
    end: pd.Timestamp,
    parameters: Country,
    monthly: pd.DataFrame,
    regions: gpd.GeoDataFrame,
    cache_dir: Path = physics.ERA5_CACHE_DIR,
) -> SolarArrays:
    """Solar arrays for hours [start, end), both hour starts in UTC, from the inputs of
    physics.model. The covariates are capacity-weighted means over the cells, with the clear-sky
    irradiance of pv.clear_sky_ghi."""
    grid.check_period(start, end)
    # ERA5 times are naive UTC.
    weather = physics.read_weather(
        parameters.country,
        parameters.era5_box,
        physics.INPUTS["solar"],
        start.tz_convert(None),
        end.tz_convert(None),
        cache_dir,
    )
    fleet = physics.fleet_cells("solar", weather, parameters, monthly, regions)
    (capacity,) = fleet.capacities
    capacity_mw = capacity.transpose("time", "cell").to_numpy()
    fields = {
        name: fleet.cells[name].transpose("time", "cell").to_numpy()
        for name in physics.INPUTS["solar"]
    }
    latitude = fleet.cells["latitude"].to_numpy()
    longitude = fleet.cells["longitude"].to_numpy()
    # Radiation is the mean over the hour, so the sun is placed at its middle.
    times = pd.DatetimeIndex(fleet.cells.indexes["time"]).tz_localize("UTC") + pv.HALF_HOUR
    dni_extra = pvlib.irradiance.get_extra_radiation(times).to_numpy()[:, np.newaxis]
    params = parameters.solar
    systems = [*params.fleet, HORIZONTAL]

    energy = np.zeros((times.size, len(systems), DC_AC_GRID.size))
    # Capacity-weighted sums of the sun elevation, the global and the clear-sky irradiance.
    sky = np.zeros((3, times.size))
    # Blocks of cells as in pv.capacity_factor, with the same conversions, so that each cell
    # gets the values of physics.model.
    for first in range(0, latitude.size, pv.CELLS_PER_BLOCK):
        block = slice(first, first + pv.CELLS_PER_BLOCK)
        field = {name: values[:, block].astype(np.float64) for name, values in fields.items()}
        ghi, temp_air = field["ssrd"], field["t2m"] - pv.KELVIN
        wind_speed = np.hypot(field["u10"], field["v10"])
        zenith, apparent_zenith, azimuth = pv.solar_position(
            times, latitude[block], longitude[block], temp_air
        )
        dni, dhi = pv.split_irradiance(ghi, field["fdir"], zenith)
        # A contiguous copy makes the products with the cell values several times faster.
        weights = np.ascontiguousarray(capacity_mw[:, block])
        for index, system in enumerate(systems):
            poa = pv.plane_of_array(
                system, apparent_zenith, azimuth, dni, ghi, dhi, dni_extra, params.albedo
            )
            dc = pv.dc_per_unit(poa, temp_air, wind_speed, params.gamma_pdc)
            for column, ratio in enumerate(DC_AC_GRID):
                ac = pv.ac_from_dc(dc, ratio, params.dc_losses)
                energy[:, index, column] += np.vecdot(ac, weights)
        clear = pv.clear_sky_ghi(
            times, latitude[block], longitude[block], apparent_zenith, dni_extra
        )
        for row, values in enumerate([90.0 - apparent_zenith, ghi, clear]):
            sky[row] += np.vecdot(values, weights)

    elevation, mean_ghi, mean_clear = sky / capacity.sum("cell").to_numpy()
    complete = np.isfinite(energy).all(axis=(1, 2))
    return SolarArrays(
        fleet_mwh=energy[:, :-1],
        horizontal_mwh=energy[:, -1],
        shares=np.array([system.share for system in params.fleet]),
        covariates=_covariates(fleet, complete, elevation, mean_ghi, mean_clear),
    )


def solar_energy(
    arrays: SolarArrays,
    dc_ac_ratio: float,
    level: float = 1.0,
    horizontal_share: float = 0.0,
    seasonal_amplitude: float = 0.0,
) -> "pd.Series[float]":
    """National solar output in MWh per hour, NaN where the hour is incomplete:

        level * (1 + seasonal_amplitude * sin(2 pi (d - 172) / 365.25))
              * ((1 - horizontal_share) * fleet + horizontal_share * horizontal)

    with d the UTC day of year of the hour, fleet the configured systems weighted by their shares
    and horizontal the horizontal modules, both at `dc_ac_ratio`, a value of DC_AC_GRID. The
    parameters act on AC output, after the inverter.
    """
    matches = np.flatnonzero(np.isclose(DC_AC_GRID, dc_ac_ratio, rtol=0.0, atol=1e-9))
    if matches.size != 1:
        raise ValueError(f"DC/AC ratio {dc_ac_ratio} is not on the grid")
    fleet = arrays.fleet_mwh[:, :, matches[0]] @ arrays.shares
    horizontal = arrays.horizontal_mwh[:, matches[0]]
    hours = pd.DatetimeIndex(arrays.covariates.index)
    day = hours.day_of_year.to_numpy()
    seasonal = 1 + seasonal_amplitude * np.sin(2 * np.pi * (day - SOLSTICE_DAY) / YEAR_DAYS)
    energy = level * seasonal * ((1 - horizontal_share) * fleet + horizontal_share * horizontal)
    return pd.Series(energy, index=hours, name="energy_mwh")


def _covariates(
    fleet: physics.Fleet,
    complete: BoolArray,
    sun_elevation_deg: FloatArray,
    ghi_w_m2: FloatArray,
    clear_sky_ghi_w_m2: FloatArray,
) -> DataFrame[Covariates]:
    hours = pd.DatetimeIndex(fleet.cells.indexes["time"])
    covariates = pd.DataFrame(
        {
            "complete": complete,
            "capacity_mw": sum(capacity.sum("cell").to_numpy() for capacity in fleet.capacities),
            "capacity_provisional": fleet.capacity_provisional,
            "era5_source": fleet.cells["source"].to_numpy().astype(str),
            "sun_elevation_deg": sun_elevation_deg,
            "ghi_w_m2": ghi_w_m2,
            "clear_sky_ghi_w_m2": clear_sky_ghi_w_m2,
        },
        # ERA5 times read outside the zarr cache are not always in nanoseconds.
        index=hours.tz_localize("UTC").as_unit("ns").rename("time"),
    )
    return Covariates.validate(covariates)
