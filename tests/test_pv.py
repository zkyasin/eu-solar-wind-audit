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


def test_sun_is_placed_at_the_middle_of_the_hour() -> None:
    # At longitude 0 on 13 June solar noon is 12:00 UTC to within 10 s. The hours from 11:00 and
    # from 12:00 then mirror each other, and a south-facing panel sees both alike.
    ds = weather("2025-06-13", np.full(24, 700.0), np.full(24, 500.0), longitude=0.0)

    cf = run(ds, SOUTH_30)

    assert cf[11] == pytest.approx(cf[12], rel=1e-4)
    assert cf[10] == pytest.approx(cf[13], rel=1e-4)
    # With the sun at the start of each hour, hour 12 would be at noon and hour 11 an hour before.
    assert cf[12] != pytest.approx(cf[13], rel=1e-2)


def test_zero_at_night() -> None:
    ds = clear_weather(SUMMER)
    night = (SITE.get_solarposition(mid_hours(SUMMER))["zenith"] > 90).to_numpy()

    cf = run(ds, SOUTH_30, dc_ac_ratio=1.3)

    assert night.sum() >= 6
    assert (cf[night] == 0).all()
    assert (cf[~night] >= 0).all()


def test_output_clips_at_ac_capacity() -> None:
    ds = clear_weather(SUMMER, temp_c=0.0)

    cf = run(ds, TRACKER, dc_ac_ratio=2.0)

    assert cf.max() == 1.0
    assert (cf == 1.0).sum() >= 4
    assert run(ds, TRACKER, dc_ac_ratio=1.0).max() < 1.0


@pytest.mark.parametrize("name", ["ssrd", "fdir", "t2m", "u10", "v10"])
def test_missing_input_gives_missing_output(name: str) -> None:
    ds = clear_weather(SUMMER)
    clean = run(ds, SOUTH_30)
    gappy = ds.copy(deep=True)
    gappy[name][[2, 12], 0, 0] = np.nan

    cf = run(gappy, SOUTH_30)

    assert np.isnan(cf[[2, 12]]).all()
    others = np.ones(24, dtype=bool)
    others[[2, 12]] = False
    np.testing.assert_array_equal(cf[others], clean[others])


def test_direct_above_global_counts_as_all_direct() -> None:
    ssrd = clear_sky(SUMMER)["ghi"].to_numpy()

    above = run(weather(SUMMER, ssrd, 1.02 * ssrd), SOUTH_30)
    equal = run(weather(SUMMER, ssrd, ssrd), SOUTH_30)

    assert np.isfinite(above).all()
    np.testing.assert_array_equal(above, equal)


def test_direct_light_at_very_low_sun_counts_as_diffuse() -> None:
    zenith = SITE.get_solarposition(mid_hours(SUMMER))["zenith"].to_numpy()
    low = np.flatnonzero((zenith > 88) & (zenith < 95))
    ssrd = np.where(zenith < 95, 30.0, 0.0)

    with_direct = run(weather(SUMMER, ssrd, 0.6 * ssrd), SOUTH_30)
    diffuse_only = run(weather(SUMMER, ssrd, np.zeros(24)), SOUTH_30)

    assert low.size >= 2
    assert (with_direct[low] > 0).all()
    np.testing.assert_array_equal(with_direct[low], diffuse_only[low])
