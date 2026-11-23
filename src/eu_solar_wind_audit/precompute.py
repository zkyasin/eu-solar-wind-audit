"""Physics of mainland France computed once for a period, so that calibration can evaluate the
solar and onshore wind output for many parameter values without running physics.model again. The
arrays are kept in memory only, so they always match the code and inputs that produced them.

Solar: national AC energy of the whole fleet mounted as each configured system, and as horizontal
modules, for every DC/AC ratio of DC_AC_GRID. The calibration parameters act on AC output, after
the inverter, so evaluating them costs a few operations on these sums.

Onshore wind: capacity of each commissioning window in each 0.1 m/s bin of hub-height speed, hour
by hour. Evaluating a speed scale costs one product of these histograms with the smoothed power
curves; taking each speed at the centre of its bin is the only approximation.
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

from eu_solar_wind_audit import grid, physics, pv, wind
from eu_solar_wind_audit.config import Country, WindGeneration

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]

DC_AC_GRID = np.round(np.linspace(0.90, 1.80, 91), 2)
HORIZONTAL = pv.System(tilt=0.0, azimuth=180.0, share=1.0)
# Day of year of the June solstice in a common year, where the seasonal term of solar is zero.
SOLSTICE_DAY = 172
YEAR_DAYS = 365.25
# 0.1 m/s bins over the speeds of the smoothed power curves.
SPEED_EDGES = np.linspace(0.0, wind.MAX_SPEED, 401)
SPEED_CENTRES = (SPEED_EDGES[:-1] + SPEED_EDGES[1:]) / 2


class Covariates(pa.DataFrameModel):
    """Hourly values that calibration needs besides the output, for the hour starting at `time`:
    whether the output of the hour is complete, as in physics.model, the capacity in MW at its
    start and whether it uses provisional SDES months, the ERA5 release, the apparent sun
    elevation and the global and clear-sky horizontal irradiance in W m-2 seen by the fleet at
    mid-hour, and the air temperature at 2 m in degC over the hour."""

    time: Index[grid.UtcTime] = pa.Field(unique=True, check_name=True)
    complete: Series[bool]
    capacity_mw: Series[float] = pa.Field(ge=0)
    capacity_provisional: Series[bool]
    era5_source: Series[str] = pa.Field(isin=["ERA5", "ERA5T"])
    sun_elevation_deg: Series[float] = pa.Field(nullable=True)
    ghi_w_m2: Series[float] = pa.Field(nullable=True)
    clear_sky_ghi_w_m2: Series[float] = pa.Field(nullable=True)
    air_temperature_c: Series[float] = pa.Field(nullable=True)

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


@dataclass(frozen=True)
class WindArrays:
    """Capacity in MW of each commissioning window of `generations` whose hub-height speed lies in
    each bin of SPEED_EDGES, over each hour of `covariates`: shape (generation, time, bin)."""

    capacity_mw: FloatArray
    generations: tuple[WindGeneration, ...]
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
    # Capacity-weighted sums of the sun elevation, the global and the clear-sky irradiance and
    # the air temperature.
    weighted = np.zeros((4, times.size))
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
        for row, values in enumerate([90.0 - apparent_zenith, ghi, clear, temp_air]):
            weighted[row] += np.vecdot(values, weights)

    elevation, mean_ghi, mean_clear, mean_temperature = weighted / capacity.sum("cell").to_numpy()
    complete = np.isfinite(energy).all(axis=(1, 2))
    return SolarArrays(
        fleet_mwh=energy[:, :-1],
        horizontal_mwh=energy[:, -1],
        shares=np.array([system.share for system in params.fleet]),
        covariates=_covariates(fleet, complete, elevation, mean_ghi, mean_clear, mean_temperature),
    )


def solar_energy(
    arrays: SolarArrays,
    dc_ac_ratio: float,
    level: float = 1.0,
    horizontal_share: float = 0.0,
    seasonal_amplitude: float = 0.0,
) -> "pd.Series[float]":
    """National solar output in MWh per hour, NaN where the hour is incomplete: the
    `solar_potential` of the fleet and of the horizontal modules at `dc_ac_ratio`, a value of
    DC_AC_GRID."""
    matches = np.flatnonzero(np.isclose(DC_AC_GRID, dc_ac_ratio, rtol=0.0, atol=1e-9))
    if matches.size != 1:
        raise ValueError(f"DC/AC ratio {dc_ac_ratio} is not on the grid")
    hours = pd.DatetimeIndex(arrays.covariates.index)
    energy = solar_potential(
        arrays.fleet_mwh[:, :, matches[0]] @ arrays.shares,
        arrays.horizontal_mwh[:, matches[0]],
        seasonal_sine(hours),
        level,
        horizontal_share,
        seasonal_amplitude,
    )
    return pd.Series(energy, index=hours, name="energy_mwh")


def solar_potential(
    fleet: FloatArray,
    horizontal: FloatArray,
    sine: FloatArray,
    level: float,
    horizontal_share: float,
    seasonal_amplitude: float,
) -> FloatArray:
    """Solar output from that of the whole fleet mounted as the configured systems weighted by
    their shares, `fleet`, and as horizontal modules, `horizontal`, both at one DC/AC ratio, and
    from the `seasonal_sine` of the hours:

        level * (1 + seasonal_amplitude * sine)
              * ((1 - horizontal_share) * fleet + horizontal_share * horizontal)

    The parameters act on AC output, after the inverter. The output is linear in `fleet` and
    `horizontal`, which can be energies or capacity factors.
    """
    seasonal = 1 + seasonal_amplitude * sine
    output: FloatArray = (
        level * seasonal * ((1 - horizontal_share) * fleet + horizontal_share * horizontal)
    )
    return output


def seasonal_sine(hours: pd.DatetimeIndex) -> FloatArray:
    """sin(2 pi (d - 172) / 365.25) with d the UTC day of year of each of `hours`: zero on the
    June solstice of a common year, negative in spring and positive in autumn."""
    day = hours.day_of_year.to_numpy()
    sine: FloatArray = np.sin(2 * np.pi * (day - SOLSTICE_DAY) / YEAR_DAYS)
    return sine


def wind_onshore(
    start: pd.Timestamp,
    end: pd.Timestamp,
    parameters: Country,
    monthly: pd.DataFrame,
    regions: gpd.GeoDataFrame,
    cache_dir: Path = physics.ERA5_CACHE_DIR,
) -> WindArrays:
    """Onshore wind arrays for hours [start, end), both hour starts in UTC, from the inputs of
    physics.model, with hub-height speeds as wind.capacity_factor computes them.

    The covariates refer to the wind fleet: the sun elevation at its capacity centroid over the
    period, for the default air temperature and pressure of pvlib, and the global and clear-sky
    irradiance and the air temperature as capacity-weighted means over its cells, as for solar.
    """
    grid.check_period(start, end)
    # ERA5 times are naive UTC.
    weather = physics.read_weather(
        parameters.country,
        parameters.era5_box,
        [*physics.INPUTS["wind_onshore"], "ssrd", "t2m"],
        start.tz_convert(None),
        end.tz_convert(None),
        cache_dir,
    )
    fleet = physics.fleet_cells("wind_onshore", weather, parameters, monthly, regions)
    capacities = [capacity.transpose("time", "cell").to_numpy() for capacity in fleet.capacities]
    total = sum(capacities, np.zeros_like(capacities[0]))
    n_hours, n_cells = total.shape
    hour_index = np.arange(n_hours)[:, np.newaxis]
    hour_edges = np.arange(n_hours + 1)
    latitude = fleet.cells["latitude"].to_numpy()
    longitude = fleet.cells["longitude"].to_numpy()
    # The sun at mid-hour, as for solar.
    times = pd.DatetimeIndex(fleet.cells.indexes["time"]).tz_localize("UTC") + pv.HALF_HOUR
    dni_extra = pvlib.irradiance.get_extra_radiation(times).to_numpy()[:, np.newaxis]

    histograms = np.zeros((len(capacities), n_hours, SPEED_CENTRES.size))
    complete = np.ones(n_hours, dtype=bool)
    # Capacity-weighted sums of the global and the clear-sky irradiance and the air temperature.
    weighted = np.zeros((3, n_hours))
    for first in range(0, n_cells, pv.CELLS_PER_BLOCK):
        block = slice(first, first + pv.CELLS_PER_BLOCK)
        cells = fleet.cells.isel(cell=block)
        speed_10m = wind.wind_speed(cells["u10"], cells["v10"])
        speed_100m = wind.wind_speed(cells["u100"], cells["v100"])
        for index, generation in enumerate(parameters.wind_onshore):
            hub = wind.hub_height_speed(speed_10m, speed_100m, generation.hub_height_m)
            speed = hub.transpose("time", "cell").to_numpy()
            # Such speeds would be left out of the bins, although a speed scale below 1 brings
            # them back onto the power curves.
            if (speed >= wind.MAX_SPEED).any():
                raise ValueError(f"hub-height speeds of {wind.MAX_SPEED} m/s or more")
            present = np.isfinite(speed)
            complete &= present.all(axis=1)
            counts, _, _ = np.histogram2d(
                np.broadcast_to(hour_index, speed.shape)[present],
                speed[present],
                bins=(hour_edges, SPEED_EDGES),
                weights=capacities[index][:, block][present],
            )
            histograms[index] += counts
        field = {
            name: cells[name].transpose("time", "cell").to_numpy().astype(np.float64)
            for name in ("ssrd", "t2m")
        }
        temp_air = field["t2m"] - pv.KELVIN
        _, apparent_zenith, _ = pv.solar_position(
            times, latitude[block], longitude[block], temp_air
        )
        clear = pv.clear_sky_ghi(
            times, latitude[block], longitude[block], apparent_zenith, dni_extra
        )
        weights = np.ascontiguousarray(total[:, block])
        for row, values in enumerate([field["ssrd"], clear, temp_air]):
            weighted[row] += np.vecdot(values, weights)

    mean_ghi, mean_clear, mean_temperature = weighted / total.sum(axis=1)
    per_cell = total.sum(axis=0)
    centroid = np.average(latitude, weights=per_cell), np.average(longitude, weights=per_cell)
    sun = pvlib.solarposition.get_solarposition(times, *centroid)
    return WindArrays(
        capacity_mw=histograms,
        generations=parameters.wind_onshore,
        covariates=_covariates(
            fleet,
            complete,
            sun["apparent_elevation"].to_numpy(),
            mean_ghi,
            mean_clear,
            mean_temperature,
        ),
    )


def wind_energy(arrays: WindArrays, speed_scale: float = 1.0) -> "pd.Series[float]":
    """National onshore wind output in MWh per hour, NaN where the hour is incomplete, with
    hub-height speeds multiplied by `speed_scale` and taken at the centre of their bin. The other
    parameters of each generation are those of physics.generation_parameters."""
    energy = np.zeros(len(arrays.covariates))
    for histogram, generation in zip(arrays.capacity_mw, arrays.generations, strict=True):
        parameters = physics.generation_parameters(generation)
        speeds, fraction = wind.smoothed_curve(
            generation.turbine, parameters.smoothing_base_m_s, parameters.smoothing_slope
        )
        corrected = speed_scale * SPEED_CENTRES + parameters.speed_offset_m_s
        energy += histogram @ np.interp(corrected, speeds, fraction, left=0.0, right=0.0)
    complete = arrays.covariates["complete"].to_numpy()
    return pd.Series(
        np.where(complete, energy, np.nan), index=arrays.covariates.index, name="energy_mwh"
    )


def _covariates(
    fleet: physics.Fleet,
    complete: BoolArray,
    sun_elevation_deg: FloatArray,
    ghi_w_m2: FloatArray,
    clear_sky_ghi_w_m2: FloatArray,
    air_temperature_c: FloatArray,
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
            "air_temperature_c": air_temperature_c,
        },
        # ERA5 times read outside the zarr cache are not always in nanoseconds.
        index=hours.tz_localize("UTC").as_unit("ns").rename("time"),
    )
    return Covariates.validate(covariates)
