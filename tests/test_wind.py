from importlib.resources import files

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from eu_solar_wind_audit import wind

V82 = "VestasV82_1.65MW_82"
IEA_3_4 = "IEA_Reference_3.4MW_130"


def speeds(values: list[float]) -> xr.DataArray:
    return xr.DataArray(np.array(values, dtype=float), dims="time")


def weather(speed_100m: list[float], speed_10m: list[float] | None = None) -> xr.Dataset:
    """Westerly wind on a time axis; the 10 m speed defaults to the 1/7 law."""
    at_100m = np.array(speed_100m, dtype=float)
    at_10m = at_100m / 10 ** (1 / 7) if speed_10m is None else np.array(speed_10m, dtype=float)
    zeros = np.zeros_like(at_100m)
    return xr.Dataset(
        {
            "u10": ("time", at_10m),
            "v10": ("time", zeros),
            "u100": ("time", at_100m),
            "v100": ("time", zeros),
        }
    )


def single(turbine: str, hub_height_m: float = 100.0, **kwargs: float) -> wind.WindParameters:
    return wind.WindParameters(fleet=(wind.FleetTurbine(turbine, hub_height_m, 1.0),), **kwargs)


def unsmoothed(turbine: str, **kwargs: float) -> wind.WindParameters:
    return single(turbine, smoothing_base_m_s=0.0, smoothing_slope=0.0, **kwargs)


def curves() -> pd.DataFrame:
    with (files("eu_solar_wind_audit") / "data" / "wind_power_curves.csv").open("rb") as f:
        return pd.read_csv(f)


def test_speed_is_the_norm_of_the_components() -> None:
    speed = wind.wind_speed(speeds([3.0, -6.0]), speeds([-4.0, 8.0]))

    np.testing.assert_allclose(speed, [5.0, 10.0])


def test_exponent_of_the_one_seventh_profile() -> None:
    exponent = wind.shear_exponent(speeds([5.0]), speeds([5.0 * 10 ** (1 / 7)]))

    np.testing.assert_allclose(exponent, [1 / 7])


def test_hub_height_speed_extrapolates_from_100_m() -> None:
    speed_100m = speeds([8.0])
    speed_10m = speed_100m / 10**0.2

    hub = wind.hub_height_speed(speed_10m, speed_100m, hub_height_m=150.0)

    np.testing.assert_allclose(hub, [8.0 * 1.5**0.2])


def test_clipped_and_fallback_exponents_still_start_from_100_m() -> None:
    # Where the exponent is not the measured one, 10 m and 100 m give different hub speeds.
    speed_10m = speeds([2.0, 6.0])
    speed_100m = speeds([6.0, 60.0])

    hub = wind.hub_height_speed(speed_10m, speed_100m, hub_height_m=150.0)

    np.testing.assert_allclose(hub, [6.0 * 1.5 ** (1 / 7), 60.0 * 1.5**0.60])


def test_exponent_is_clipped_to_the_tabulated_range() -> None:
    speed_10m = speeds([6.0, 6.0, 6.0])

    exponent = wind.shear_exponent(speed_10m, speeds([6.0, 3.0, 60.0]))

    np.testing.assert_allclose(exponent, [0.06, 0.06, 0.60])


def test_low_10_m_speed_falls_back_to_one_seventh() -> None:
    exponent = wind.shear_exponent(speeds([0.0, 2.0, 3.99, 4.0]), speeds([0.0, 6.0, 6.0, 6.0]))

    np.testing.assert_allclose(exponent[:3], [1 / 7] * 3)
    np.testing.assert_allclose(exponent[3], np.log10(6.0 / 4.0))


def test_exponent_is_nan_where_either_speed_is_nan() -> None:
    exponent = wind.shear_exponent(speeds([np.nan, 6.0]), speeds([6.0, np.nan]))

    assert exponent.isnull().all()


@pytest.mark.parametrize("turbine", list(wind.turbines().index))
def test_unsmoothed_curve_reproduces_the_table(turbine: str) -> None:
    table = curves().query("turbine == @turbine")
    rated_kw = wind.turbines().loc[turbine, "rated_power_kw"]

    cf = wind.capacity_factor(weather(list(table["wind_speed_m_s"])), unsmoothed(turbine))

    expected = np.clip(table["power_kw"].to_numpy() / rated_kw, 0, 1)
    np.testing.assert_allclose(cf, expected, atol=1e-12)


def test_unsmoothed_curve_is_zero_at_zero_speed_and_beyond_cut_out() -> None:
    cf = wind.capacity_factor(weather([0.0, 2.9, 20.0, 20.01, 30.0]), unsmoothed(V82))

    np.testing.assert_allclose(cf, [0.0, 0.0, 1.0, 0.0, 0.0])


@pytest.mark.parametrize("turbine", list(wind.turbines().index))
def test_smoothed_curve_stays_between_0_and_1(turbine: str) -> None:
    _, fraction = wind.smoothed_curve(turbine, 0.6, 0.2)

    assert fraction.min() >= 0.0
    assert fraction.max() <= 1.0


def test_smoothing_lowers_the_knee_at_rated_speed_and_raises_the_cut_in() -> None:
    hours = weather([3.5, 13.0])

    raw = wind.capacity_factor(hours, unsmoothed(V82))
    smoothed = wind.capacity_factor(hours, single(V82))

    assert smoothed[0] > raw[0] + 0.01
    assert smoothed[1] < raw[1] - 0.05


def test_smoothing_spreads_the_cut_out_over_higher_speeds() -> None:
    cf = wind.capacity_factor(weather([0.0, 22.0, 45.0]), single(V82))

    assert cf[0] < 1e-6
    assert 0.05 < cf[1] < 0.5
    assert cf[2] == 0.0


def test_smoothing_keeps_flat_parts_of_the_curve() -> None:
    below_cut_in = wind.capacity_factor(weather([1.0]), single(IEA_3_4))
    narrow = single(IEA_3_4, smoothing_base_m_s=0.3, smoothing_slope=0.0)
    on_the_plateau = wind.capacity_factor(weather([17.0]), narrow)

    np.testing.assert_allclose(below_cut_in, [0.0], atol=1e-3)
    np.testing.assert_allclose(on_the_plateau, [1.0], atol=1e-6)


def test_smoothing_width_grows_with_speed() -> None:
    # On a straight ramp a symmetric kernel changes nothing, so compare the kink at rated speed:
    # a wider kernel at higher speed takes more off the plateau.
    constant = single(IEA_3_4, smoothing_base_m_s=1.0, smoothing_slope=0.0)
    growing = single(IEA_3_4, smoothing_base_m_s=1.0, smoothing_slope=0.2)
    hours = weather([11.0])

    assert wind.capacity_factor(hours, growing)[0] < wind.capacity_factor(hours, constant)[0]


def test_capacity_factor_is_nan_where_any_component_is_nan() -> None:
    hours = weather([8.0, 8.0, 8.0])
    hours["u10"][0] = np.nan
    hours["v100"][1] = np.nan

    cf = wind.capacity_factor(hours, single(V82))

    assert cf[:2].isnull().all()
    assert 0.0 < cf[2] < 1.0


def test_capacity_factor_keeps_the_dims_of_the_weather() -> None:
    shape = (2, 3, 4)
    dims = ("time", "latitude", "longitude")
    rng = np.random.default_rng(0)
    hours = xr.Dataset(
        {name: (dims, rng.uniform(-15, 15, shape)) for name in ["u10", "v10", "u100", "v100"]}
    )

    cf = wind.capacity_factor(hours, single(V82))

    assert cf.dims == dims
    assert ((cf >= 0) & (cf <= 1)).all()


def test_fleet_capacity_factor_is_the_share_weighted_sum() -> None:
    hours = weather([4.0, 7.0, 10.0, 15.0], speed_10m=[3.0, 5.0, 6.0, 9.0])
    old = wind.FleetTurbine(V82, 80.0, 0.25)
    recent = wind.FleetTurbine(IEA_3_4, 120.0, 0.75)

    fleet = wind.capacity_factor(hours, wind.WindParameters(fleet=(old, recent)))

    expected = 0.25 * wind.capacity_factor(hours, single(V82, 80.0)) + 0.75 * wind.capacity_factor(
        hours, single(IEA_3_4, 120.0)
    )
    np.testing.assert_allclose(fleet, expected)


@pytest.mark.parametrize("shares", [(0.5, 0.4), (1.2, -0.2), ()])
def test_shares_must_be_non_negative_and_sum_to_1(shares: tuple[float, ...]) -> None:
    fleet = tuple(wind.FleetTurbine(V82, 80.0, share) for share in shares)

    with pytest.raises(ValueError, match="shares"):
        wind.WindParameters(fleet=fleet)


@pytest.mark.parametrize(("base", "slope"), [(0.01, 0.2), (0.0, 0.2), (0.6, -0.1)])
def test_smoothing_too_narrow_for_the_grid_is_rejected(base: float, slope: float) -> None:
    with pytest.raises(ValueError, match="smoothing"):
        single(V82, smoothing_base_m_s=base, smoothing_slope=slope)


def test_speed_correction_applies_at_hub_height() -> None:
    # A 150 m hub, so that correcting the 100 m speed before extrapolation would differ.
    speeds_100m = [5.0, 8.0, 11.0]
    corrected = [1.1 * speed * 1.5 ** (1 / 7) - 0.5 for speed in speeds_100m]

    cf = wind.capacity_factor(
        weather(speeds_100m), single(V82, 150.0, speed_scale=1.1, speed_offset_m_s=-0.5)
    )

    np.testing.assert_allclose(cf, wind.capacity_factor(weather(corrected), single(V82)))


def test_negative_corrected_speed_gives_no_output() -> None:
    cf = wind.capacity_factor(weather([1.0]), unsmoothed(V82, speed_offset_m_s=-3.0))

    np.testing.assert_allclose(cf, [0.0])


def test_every_turbine_has_a_curve_and_every_curve_a_turbine() -> None:
    assert set(curves()["turbine"]) == set(wind.turbines().index)


def test_specific_power_is_rated_power_over_swept_area() -> None:
    table = wind.turbines()
    swept_area = np.pi * table["rotor_diameter_m"] ** 2 / 4

    np.testing.assert_allclose(
        table["specific_power_w_m2"], table["rated_power_kw"] * 1000 / swept_area, atol=0.05
    )


@pytest.mark.parametrize("turbine", list(wind.turbines().index))
def test_curve_speeds_increase_and_output_peaks_at_rated_power(turbine: str) -> None:
    curve = curves().query("turbine == @turbine")
    rated_kw = float(wind.turbines()["rated_power_kw"][turbine])

    assert (np.diff(curve["wind_speed_m_s"]) > 0).all()
    assert curve["power_kw"].max() == pytest.approx(rated_kw, rel=1e-3)
    # A table cut before its cut-out speed would stop the fleet early: power_curve gives 0 beyond
    # the last speed.
    at_rated = curve["power_kw"] >= 0.99 * rated_kw
    held = curve["wind_speed_m_s"].between(curve.loc[at_rated, "wind_speed_m_s"].min(), 20.0)
    assert curve["wind_speed_m_s"].max() >= 20.0
    assert at_rated[held].all()


@pytest.mark.parametrize(
    ("technology", "lowest", "highest"), [("onshore", 150.0, 380.0), ("offshore", 300.0, 400.0)]
)
def test_specific_powers_span_the_fleet_range_in_steps_of_at_most_50_w_m2(
    technology: str, lowest: float, highest: float
) -> None:
    specific_power = np.sort(
        wind.turbines().query("technology == @technology")["specific_power_w_m2"].to_numpy()
    )

    assert specific_power[0] == pytest.approx(lowest, abs=5)
    assert specific_power[-1] == pytest.approx(highest, abs=5)
    assert np.diff(specific_power).max() <= 50
