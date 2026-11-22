"""Capacity factor of an onshore or offshore wind fleet on the ERA5 grid.

Power curves come from the NREL turbine-models archive (BSD 3-Clause, licence in `data/`). They are
smoothed for the spread of speeds within an hour and across a fleet as in Staffell and Pfenninger
(2016, Energy 114, 1224-1239). Air density is not corrected: the tabulated curves are used as they
are at every cell and hour.
"""

from dataclasses import dataclass
from functools import cache
from importlib.resources import files
from typing import NamedTuple

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr

FloatArray = npt.NDArray[np.float64]

# Range of the Hellmann exponents tabulated by Kaltschmitt, Streicher and Wiese (2007, p. 55),
# from unstable air over open water to stable air over inhabited areas.
MIN_EXPONENT = 0.06
MAX_EXPONENT = 0.60
# Default minimum speed of the Windographer shear analysis
# (https://support.ul-renewables.com/windographer/dlg-analyze-wind-shear.htm).
MIN_SPEED_FOR_EXPONENT = 4.0  # m/s
# The 1/7 law, default Hellman exponent of windpowerlib (wind_speed.hellman, citing Hau 2008).
FALLBACK_EXPONENT = 1 / 7
# Range and resolution of the smoothed farm curves of the Virtual Wind Farm model
# (convoluteFarmCurve in github.com/renewables-ninja/vwf, lib/VWF.EXTRAS.R).
SPEED_STEP = 0.01  # m/s
MAX_SPEED = 40.0  # m/s
# The smoothed curve is stored every fifth grid point to keep the kernel matrix small.
CURVE_STRIDE = 5


class FleetTurbine(NamedTuple):
    turbine: str
    hub_height_m: float
    share: float


@dataclass(frozen=True)
class WindParameters:
    """A fleet and the parameters to be learned.

    `fleet` lists turbines of `turbines()` with their hub height and share of fleet capacity. The
    smoothing kernel has a standard deviation of `smoothing_base_m_s + smoothing_slope * v` at speed
    v; both at 0 turn smoothing off.
    """

    fleet: tuple[FleetTurbine, ...]
    speed_scale: float = 1.0
    speed_offset_m_s: float = 0.0
    # Staffell and Pfenninger (2016), equation 1: sigma = 0.6 + 0.2 w m/s.
    smoothing_base_m_s: float = 0.6
    smoothing_slope: float = 0.2

    def __post_init__(self) -> None:
        shares = np.array([member.share for member in self.fleet], dtype=float)
        if shares.size == 0 or (shares < 0).any() or not np.isclose(shares.sum(), 1.0):
            raise ValueError("fleet shares must be non-negative and sum to 1")
        smoothing_off = self.smoothing_base_m_s == 0 and self.smoothing_slope == 0
        # The kernel is integrated on the 0.01 m/s grid, so it must be several steps wide.
        too_narrow = self.smoothing_base_m_s < 10 * SPEED_STEP or self.smoothing_slope < 0
        if not smoothing_off and too_narrow:
            raise ValueError("smoothing must be off or have a base of at least 0.1 m/s")


def _read_data(name: str) -> pd.DataFrame:
    with (files("eu_solar_wind_audit") / "data" / name).open("rb") as f:
        return pd.read_csv(f)


def turbines() -> pd.DataFrame:
    """Turbines with a power curve, indexed by name: technology, rated power (kW), rotor diameter
    (m) and specific power (W m-2)."""
    return _read_data("wind_turbines.csv").set_index("turbine")


def power_curve(turbine: str) -> tuple[FloatArray, FloatArray]:
    """Tabulated speeds (m/s) and output as a fraction of rated power."""
    rated_kw = float(turbines()["rated_power_kw"][turbine])
    curves = _read_data("wind_power_curves.csv")
    curve = curves[curves["turbine"] == turbine]
    speeds: FloatArray = curve["wind_speed_m_s"].to_numpy(np.float64)
    # Some tables overshoot rated power by a few parts in 10^5.
    fraction: FloatArray = np.clip(curve["power_kw"].to_numpy(np.float64) / rated_kw, 0.0, 1.0)
    return speeds, fraction


@cache
def smoothed_curve(turbine: str, base_m_s: float, slope: float) -> tuple[FloatArray, FloatArray]:
    """Speeds (m/s) and output fraction of the curve convolved with a Gaussian of standard
    deviation `base_m_s + slope * v` centred on each speed v; the raw curve if both are 0."""
    speeds, fraction = power_curve(turbine)
    if base_m_s != 0 or slope != 0:
        grid = np.arange(0.0, MAX_SPEED + SPEED_STEP / 2, SPEED_STEP)
        raw = np.interp(grid, speeds, fraction, left=0.0, right=0.0)
        speeds = grid[::CURVE_STRIDE]
        sigma = (base_m_s + slope * speeds)[:, np.newaxis]
        z = (grid[np.newaxis, :] - speeds[:, np.newaxis]) / sigma
        # The density is not renormalized on the grid: its mass below 0 m/s produces no power.
        kernel = np.exp(-0.5 * z**2) / (sigma * np.sqrt(2 * np.pi)) * SPEED_STEP
        fraction = np.clip(kernel @ raw, 0.0, 1.0)
    # The arrays are cached and shared between calls.
    speeds.setflags(write=False)
    fraction.setflags(write=False)
    return speeds, fraction


def wind_speed(u: xr.DataArray, v: xr.DataArray) -> xr.DataArray:
    speed: xr.DataArray = xr.ufuncs.hypot(u, v)
    return speed


def shear_exponent(speed_10m: xr.DataArray, speed_100m: xr.DataArray) -> xr.DataArray:
    """Power-law exponent between 10 m and 100 m, clipped to the tabulated range; the 1/7 law
    where the 10 m speed is too low for the ratio to be meaningful."""
    with np.errstate(divide="ignore", invalid="ignore"):
        exponent: xr.DataArray = xr.ufuncs.log(speed_100m / speed_10m) / np.log(100.0 / 10.0)
    exponent = exponent.clip(MIN_EXPONENT, MAX_EXPONENT)
    exponent = exponent.where(speed_10m >= MIN_SPEED_FOR_EXPONENT, FALLBACK_EXPONENT)
    return exponent.where(speed_10m.notnull() & speed_100m.notnull())


def hub_height_speed(
    speed_10m: xr.DataArray, speed_100m: xr.DataArray, hub_height_m: float
) -> xr.DataArray:
    """Speed at hub height, extrapolated from 100 m with the 10-100 m exponent (Karkossa et al.
    2026, arXiv 2610.39859, equations S9 and S10)."""
    exponent = shear_exponent(speed_10m, speed_100m)
    # At 100 m the factor is 1.0 ** exponent, which is 1 even for a NaN exponent.
    return (speed_100m * (hub_height_m / 100.0) ** exponent).where(exponent.notnull())


def capacity_factor(weather: xr.Dataset, parameters: WindParameters) -> xr.DataArray:
    """Fleet capacity factor in [0, 1] from `u10`, `v10`, `u100` and `v100` (m/s).

    The result has the dims of the inputs and is NaN where any of them is NaN.
    """
    speed_10m = wind_speed(weather["u10"], weather["v10"])
    speed_100m = wind_speed(weather["u100"], weather["v100"])
    total = xr.zeros_like(speed_100m)
    for member in parameters.fleet:
        hub_speed = hub_height_speed(speed_10m, speed_100m, member.hub_height_m)
        corrected = parameters.speed_scale * hub_speed + parameters.speed_offset_m_s
        speeds, fraction = smoothed_curve(
            member.turbine, parameters.smoothing_base_m_s, parameters.smoothing_slope
        )
        output: xr.DataArray = xr.apply_ufunc(
            np.interp,
            corrected,
            kwargs={"xp": speeds, "fp": fraction, "left": 0.0, "right": 0.0},
            dask="parallelized",
            output_dtypes=[float],
        )
        total = total + member.share * output
    return total.rename("capacity_factor")
