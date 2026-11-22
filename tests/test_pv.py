import numpy as np
import pandas as pd
import pvlib
import pytest
import xarray as xr

from eu_solar_wind_audit import pv
from eu_solar_wind_audit.pv import PVParameters, System, capacity_factor

SOUTH_30 = System(tilt=30, azimuth=180, share=1)
HORIZONTAL = System(tilt=0, azimuth=180, share=1)
TRACKER = System(tilt=0, azimuth=180, share=1, tracker=True)
SITE = pvlib.location.Location(latitude=45.0, longitude=5.0, altitude=0)
SUMMER = "2025-06-21"
WINTER = "2025-12-21"
HALF_HOUR = pd.Timedelta(minutes=30)


def hours(day: str) -> pd.DatetimeIndex:
    return pd.date_range(day, periods=24, freq="h")


def mid_hours(day: str) -> pd.DatetimeIndex:
    return hours(day).tz_localize("UTC") + HALF_HOUR


def weather(
    day: str,
    ssrd: np.ndarray,
    fdir: np.ndarray,
    temp_c: float = 20.0,
    latitude: float = SITE.latitude,
    longitude: float = SITE.longitude,
) -> xr.Dataset:
    """One cell, 24 hours, wind of 5 m/s from the u and v components."""
    dims = ("time", "latitude", "longitude")
    shape = (24, 1, 1)

    def field(values: np.ndarray | float) -> tuple[tuple[str, ...], np.ndarray]:
        return dims, np.broadcast_to(np.asarray(values, dtype=float).reshape(-1, 1, 1), shape)

    return xr.Dataset(
        {
            "ssrd": field(ssrd),
            "fdir": field(fdir),
            "t2m": field(temp_c + 273.15),
            "u10": field(3.0),
            "v10": field(-4.0),
        },
        coords={"time": hours(day), "latitude": [latitude], "longitude": [longitude]},
    )


def clear_sky(day: str) -> pd.DataFrame:
    """Ineichen clear sky at the middle of each hour, standing in for the hourly means."""
    sky: pd.DataFrame = SITE.get_clearsky(mid_hours(day), model="ineichen", linke_turbidity=3.0)
    return sky


def clear_weather(day: str, temp_c: float = 20.0) -> xr.Dataset:
    sky = clear_sky(day)
    return weather(day, sky["ghi"].to_numpy(), (sky["ghi"] - sky["dhi"]).to_numpy(), temp_c)


def series(cf: xr.DataArray) -> np.ndarray:
    return cf.isel(latitude=0, longitude=0).to_numpy()


def run(ds: xr.Dataset, *fleet: System, dc_ac_ratio: float = 1.1) -> np.ndarray:
    return series(capacity_factor(ds, PVParameters(fleet, dc_ac_ratio=dc_ac_ratio)))


@pytest.mark.parametrize(("day", "temp_c"), [(SUMMER, 25.0), (WINTER, 2.0)])
@pytest.mark.parametrize("system", [SOUTH_30, TRACKER])
def test_matches_pvlib_modelchain(day: str, temp_c: float, system: System) -> None:
    params = PVParameters((system,))
    sky = clear_sky(day)
    times = mid_hours(day)
    zenith = SITE.get_solarposition(times, temperature=temp_c)["zenith"]
    # ModelChain needs DNI: the direct horizontal part over the cosine of the true zenith.
    chain_weather = pd.DataFrame(
        {
            "ghi": sky["ghi"],
            "dhi": sky["dhi"],
            "dni": (sky["ghi"] - sky["dhi"]) / np.cos(np.radians(zenith)),
            "temp_air": temp_c,
            "wind_speed": 5.0,
        }
    )
    mount = (
        pvlib.pvsystem.SingleAxisTrackerMount(
            axis_tilt=system.tilt,
            axis_azimuth=system.azimuth,
            max_angle=pv.TRACKER_MAX_ANGLE,
            backtrack=True,
            gcr=pv.TRACKER_GCR,
        )
        if system.tracker
        else pvlib.pvsystem.FixedMount(surface_tilt=system.tilt, surface_azimuth=system.azimuth)
    )
    array = pvlib.pvsystem.Array(
        mount,
        albedo=params.albedo,
        module_parameters={"pdc0": params.dc_ac_ratio, "gamma_pdc": params.gamma_pdc},
        temperature_model_parameters={"u0": 25.0, "u1": 6.84},
    )
    chain = pvlib.modelchain.ModelChain(
        pvlib.pvsystem.PVSystem(arrays=[array], inverter_parameters={"pdc0": 1 / 0.96}),
        SITE,
        transposition_model="haydavies",
        aoi_model="no_loss",
        spectral_model="no_loss",
        temperature_model="faiman",
        dc_model="pvwatts",
        ac_model="pvwatts",
        losses_model="pvwatts",
    )
    chain.run_model(chain_weather)

    cf = run(clear_weather(day, temp_c), system)

    # From 88 degrees pvlib rejects the DNI, which the model then counts as diffuse.
    trusted = (zenith < 88).to_numpy()
    assert trusted.sum() >= 8
    np.testing.assert_allclose(cf[trusted], chain.results.ac.to_numpy()[trusted], atol=1e-12)
    assert cf.max() > 0.3
