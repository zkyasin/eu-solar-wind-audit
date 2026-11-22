"""AC capacity factor of a PV fleet from hourly ERA5 fields, with pvlib.

Defaults come from the PVWatts V5 manual (A. P. Dobos, NREL/TP-6A20-62641, 2014), cited by section.
"""

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd
import pvlib
import xarray as xr

Array = npt.NDArray[np.float64]

HALF_HOUR = pd.Timedelta(minutes=30)
KELVIN = 273.15
# One-axis tracker: rotation limit (section 6) and ground coverage ratio (section 3).
TRACKER_MAX_ANGLE = 45.0
TRACKER_GCR = 0.4
# Nominal inverter efficiency (section 12), also the pvlib.inverter.pvwatts default.
INVERTER_EFFICIENCY = 0.96
# Cells computed together: a year of hours then needs a few hundred MB.
CELLS_PER_BLOCK = 100


@dataclass(frozen=True)
class System:
    """One mounting type and its share of the fleet's DC capacity. Angles in degrees, azimuth in
    the pvlib convention (180 = south); for a single-axis tracker they are those of the axis."""

    tilt: float
    azimuth: float
    share: float
    tracker: bool = False


@dataclass(frozen=True)
class PVParameters:
    fleet: tuple[System, ...]
    dc_ac_ratio: float = 1.1  # section 3
    # Section 11 default system losses, 14.08 %, as computed by pvlib.
    dc_losses: float = pvlib.pvsystem.pvwatts_losses() / 100
    gamma_pdc: float = -0.0047  # 1/degC, standard module, section 3
    albedo: float = 0.2  # section 7

    def __post_init__(self) -> None:
        if not math.isclose(sum(system.share for system in self.fleet), 1.0):
            raise ValueError("fleet shares must sum to 1")


def solar_position(
    times: pd.DatetimeIndex, latitude: Array, longitude: Array, temp_air: Array
) -> tuple[Array, Array, Array]:
    """True zenith, apparent zenith and azimuth in degrees, shape (time, cell), from SPA.

    Refraction uses the air temperature in degC, as pvlib's ModelChain does.
    """
    zenith, apparent_zenith, azimuth = (np.empty_like(temp_air) for _ in range(3))
    for cell, (lat, lon) in enumerate(zip(latitude, longitude, strict=True)):
        position = pvlib.solarposition.get_solarposition(
            times, lat, lon, temperature=temp_air[:, cell]
        )
        zenith[:, cell] = position["zenith"].to_numpy()
        apparent_zenith[:, cell] = position["apparent_zenith"].to_numpy()
        azimuth[:, cell] = position["azimuth"].to_numpy()
    return zenith, apparent_zenith, azimuth


def split_irradiance(ghi: Array, direct_horizontal: Array, zenith: Array) -> tuple[Array, Array]:
    """Direct normal and diffuse horizontal irradiance from global and direct horizontal."""
    # ERA5 packing can leave the direct part slightly above the global one.
    dhi = np.clip(ghi - direct_horizontal, 0.0, ghi)
    dni = pvlib.irradiance.dni(ghi, dhi, zenith)
    # pvlib sets DNI to NaN when the sun is too low to trust it; that light is kept as diffuse.
    rejected = np.isnan(dni) & ~np.isnan(dhi)
    return np.where(rejected, 0.0, dni), np.where(rejected, ghi, dhi)


def plane_of_array(
    system: System,
    apparent_zenith: Array,
    azimuth: Array,
    dni: Array,
    ghi: Array,
    dhi: Array,
    dni_extra: Array,
    albedo: float,
) -> Array:
    """Global irradiance on the modules in W m-2, Hay and Davies sky diffuse model."""
    tilt: float | Array = system.tilt
    surface_azimuth: float | Array = system.azimuth
    if system.tracker:
        tracking = pvlib.tracking.singleaxis(
            apparent_zenith.ravel(),
            azimuth.ravel(),
            axis_tilt=system.tilt,
            axis_azimuth=system.azimuth,
            max_angle=TRACKER_MAX_ANGLE,
            backtrack=True,
            gcr=TRACKER_GCR,
        )
        # Below the horizon the tracker rests along its axis, as in pvlib's ModelChain.
        tilt = np.nan_to_num(tracking["surface_tilt"], nan=system.tilt).reshape(ghi.shape)
        surface_azimuth = np.nan_to_num(tracking["surface_azimuth"], nan=system.azimuth).reshape(
            ghi.shape
        )
    irradiance = pvlib.irradiance.get_total_irradiance(
        tilt,
        surface_azimuth,
        apparent_zenith,
        azimuth,
        dni,
        ghi,
        dhi,
        dni_extra=dni_extra,
        albedo=albedo,
        model="haydavies",
    )
    poa: Array = irradiance["poa_global"]
    return poa


def dc_per_unit(poa: Array, temp_air: Array, wind_speed: Array, gamma_pdc: float) -> Array:
    """DC output per unit of DC capacity before system losses, with temperatures in degC and wind
    speed in m/s."""
    temp_cell = pvlib.temperature.faiman(poa, temp_air, wind_speed)
    dc: Array = pvlib.pvsystem.pvwatts_dc(poa, temp_cell, 1.0, gamma_pdc)
    return dc


def ac_from_dc(dc: Array, dc_ac_ratio: float, dc_losses: float) -> Array:
    """AC output per unit of AC capacity from the DC output per unit of DC capacity of
    `dc_per_unit`."""
    # An inverter DC rating of 1 / efficiency gives an AC nameplate of 1, where pvwatts clips.
    ac: Array = pvlib.inverter.pvwatts(
        dc * dc_ac_ratio * (1 - dc_losses),
        1 / INVERTER_EFFICIENCY,
        eta_inv_nom=INVERTER_EFFICIENCY,
    )
    return ac


def ac_capacity_factor(
    poa: Array, temp_air: Array, wind_speed: Array, params: PVParameters
) -> Array:
    """AC output per unit of AC capacity, with temperatures in degC and wind speed in m/s."""
    dc = dc_per_unit(poa, temp_air, wind_speed, params.gamma_pdc)
    return ac_from_dc(dc, params.dc_ac_ratio, params.dc_losses)


def fleet_capacity_factor(
    times: pd.DatetimeIndex,
    latitude: Array,
    longitude: Array,
    ghi: Array,
    direct_horizontal: Array,
    temp_air: Array,
    wind_speed: Array,
    params: PVParameters,
) -> Array:
    """AC capacity factor on arrays of shape (time, cell), with `times` the UTC instants at which
    the sun is placed, irradiance in W m-2, temperature in degC and wind speed in m/s."""
    zenith, apparent_zenith, azimuth = solar_position(times, latitude, longitude, temp_air)
    dni, dhi = split_irradiance(ghi, direct_horizontal, zenith)
    dni_extra = pvlib.irradiance.get_extra_radiation(times).to_numpy()[:, np.newaxis]
    total = np.zeros_like(ghi)
    for system in params.fleet:
        poa = plane_of_array(
            system, apparent_zenith, azimuth, dni, ghi, dhi, dni_extra, params.albedo
        )
        total += system.share * ac_capacity_factor(poa, temp_air, wind_speed, params)
    return total


def capacity_factor(weather: xr.Dataset, params: PVParameters) -> xr.DataArray:
    """AC capacity factor of the fleet for each hour [t, t + 1 h) labelled t, in naive UTC.

    `weather` holds ssrd and fdir as mean W m-2 over the hour, t2m in K and u10, v10 in m/s, on
    any spatial dims that carry latitude and longitude coordinates.
    """
    ssrd = weather["ssrd"]
    space = [dim for dim in ssrd.dims if dim != "time"]
    layout = ssrd.transpose("time", *space)
    n_times = layout.sizes["time"]
    fields = {
        name: weather[name].transpose("time", *space).to_numpy().reshape(n_times, -1)
        for name in ("ssrd", "fdir", "t2m", "u10", "v10")
    }
    cells = layout.isel(time=0, drop=True)
    latitude, longitude = (
        weather[name].broadcast_like(cells).transpose(*space).to_numpy().ravel()
        for name in ("latitude", "longitude")
    )
    # Radiation is the mean over the hour, so the sun is placed at its middle.
    times = pd.DatetimeIndex(weather.indexes["time"]).tz_localize("UTC") + HALF_HOUR

    total = np.empty(fields["ssrd"].shape)
    for start in range(0, latitude.size, CELLS_PER_BLOCK):
        block = slice(start, start + CELLS_PER_BLOCK)
        field = {name: values[:, block].astype(np.float64) for name, values in fields.items()}
        total[:, block] = fleet_capacity_factor(
            times,
            latitude[block],
            longitude[block],
            field["ssrd"],
            field["fdir"],
            field["t2m"] - KELVIN,
            np.hypot(field["u10"], field["v10"]),
            params,
        )

    result = xr.DataArray(
        total.reshape(layout.shape), dims=layout.dims, coords=layout.coords, name="capacity_factor"
    )
    return result.transpose(*ssrd.dims)
