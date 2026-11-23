import dataclasses
from collections.abc import Sequence
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pvlib
import pytest
import xarray as xr
from shapely.geometry import box

from eu_solar_wind_audit import physics, precompute, pv, wind
from eu_solar_wind_audit.config import Country, WindGeneration
from eu_solar_wind_audit.precompute import FloatArray
from eu_solar_wind_audit.pv import PVParameters, System
from eu_solar_wind_audit.sources import era5

STEP = 0.25
BOX = era5.Box(west=2.0, south=45.0, east=2.75, north=45.5)
# Every input is missing at this hour in the first cell of the box.
GAP = pd.Timestamp("2020-06-03 12:00")
# Hours from this stamp on are ERA5T in the synthetic store.
ERA5T_FROM = pd.Timestamp("2020-12-01")

COUNTRY = Country(
    country="FR",
    time_zone="Europe/Paris",
    perimeter="metropolitan_without_corsica",
    era5_box=BOX,
    solar=PVParameters(
        fleet=(
            System(tilt=35.0, azimuth=180.0, share=0.5),
            System(tilt=15.0, azimuth=110.0, share=0.3),
            System(tilt=0.0, azimuth=180.0, share=0.2, tracker=True),
        ),
        dc_ac_ratio=1.2,
    ),
    wind_onshore=(
        WindGeneration("VestasV82_1.65MW_82", 80.0, 2012),
        WindGeneration("IEA_Reference_3.4MW_130", 90.0, 2020),
        WindGeneration("IEA_Reference_3.4MW_130", 100.0),
    ),
)


class WeatherStore:
    """Stands in for era5.read_era5 with random but plausible weather: the same values for the
    same box and hours on every call, whatever the variables asked for."""

    def __call__(
        self, variables: Sequence[str], box: era5.Box, start: pd.Timestamp, end: pd.Timestamp
    ) -> xr.Dataset:
        time = pd.date_range(start, end, freq="h", inclusive="left")
        latitude = np.arange(box.south, box.north + STEP / 2, STEP)
        longitude = np.arange(box.west, box.east + STEP / 2, STEP)
        shape = (time.size, latitude.size, longitude.size)
        rng = np.random.default_rng(0)
        # Daylight from 05:00 to 19:00 UTC, so that some of it falls before sunrise or after
        # sunset, and a random cloud cover.
        daylight = np.sin(np.pi * (time.hour.to_numpy() + 0.5 - 5) / 14).clip(0)
        ssrd = 950 * daylight[:, None, None] * rng.uniform(0.1, 1.1, shape)
        u100 = rng.normal(2.0, 6.0, shape)
        v100 = rng.normal(1.0, 6.0, shape)
        slowdown = rng.uniform(0.55, 0.85, shape)
        fields = {
            "ssrd": ssrd,
            "fdir": ssrd * rng.uniform(0.0, 0.85, shape),
            "t2m": rng.uniform(270.0, 305.0, shape),
            "u10": slowdown * u100,
            "v10": slowdown * v100,
            "u100": u100,
            "v100": v100,
        }
        gap = time == GAP
        for values in fields.values():
            values[gap, 0, 0] = np.nan
        dims = ("time", "latitude", "longitude")
        return xr.Dataset(
            {name: (dims, fields[name].astype("float32")) for name in variables},
            coords={
                "time": time,
                "latitude": latitude,
                "longitude": longitude,
                "source": ("time", np.where(time < ERA5T_FROM, "ERA5", "ERA5T")),
            },
        )


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch) -> WeatherStore:
    fake = WeatherStore()
    monkeypatch.setattr(era5, "read_era5", fake)
    return fake


def cell(latitude: float, longitude: float) -> tuple[float, float, float, float]:
    half = STEP / 2
    return (longitude - half, latitude - half, longitude + half, latitude + half)


# "01" covers one cell, "02" two cells; "2A" (Corse-du-Sud) one cell.
REGIONS = gpd.GeoDataFrame(
    {"region": ["01", "02", "2A"]},
    geometry=[
        box(*cell(45.0, 2.0)),
        box(*cell(45.25, 2.25)[:3], cell(45.5, 2.25)[3]),
        box(*cell(45.5, 2.75)),
    ],
    crs="EPSG:4326",
)


def panel(technology: str, capacity: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Month-end panel as sdes.monthly_capacity returns it, from {code: {month end: MW}}."""
    rows = [
        {
            "department_code": code,
            "department": code,
            "region": "region",
            "technology": technology,
            "month_end": pd.Timestamp(month_end),
            "capacity_mw": mw,
            "provisional": False,
        }
        for code, values in capacity.items()
        for month_end, mw in values.items()
    ]
    return pd.DataFrame(rows)


def utc(stamp: str) -> pd.Timestamp:
    return pd.Timestamp(stamp, tz="UTC")


# Two regions growing at different paces; the December figures are provisional, so the hours
# after 1 July 2020 lean on provisional capacity.
SOLAR_PANEL = panel(
    "solar",
    {
        "01": {"2019-12-31": 100.0, "2020-06-30": 120.0, "2020-12-31": 150.0},
        "02": {"2019-12-31": 300.0, "2020-06-30": 330.0, "2020-12-31": 340.0},
        "2A": {"2019-12-31": 900.0, "2020-06-30": 900.0, "2020-12-31": 900.0},
    },
).assign(provisional=lambda frame: frame["month_end"] == pd.Timestamp("2020-12-31"))
YEAR_2020 = (utc("2020-01-01"), utc("2021-01-01"))


@pytest.mark.parametrize(
    ("fleet", "dc_ac_ratio", "horizontal_share"),
    [
        pytest.param(COUNTRY.solar.fleet, 1.2, 0.0, id="configured-1.2"),
        pytest.param(COUNTRY.solar.fleet, 0.9, 0.0, id="configured-0.9"),
        # Spelled out, so that precompute.HORIZONTAL is checked as well.
        pytest.param((System(tilt=0.0, azimuth=180.0, share=1.0),), 1.8, 1.0, id="horizontal-1.8"),
    ],
)
def test_solar_equals_the_model_at_every_hour(
    store: WeatherStore,
    tmp_path: Path,
    fleet: tuple[System, ...],
    dc_ac_ratio: float,
    horizontal_share: float,
) -> None:
    solar = dataclasses.replace(COUNTRY.solar, fleet=fleet, dc_ac_ratio=dc_ac_ratio)
    country = dataclasses.replace(COUNTRY, solar=solar)
    modelled = physics.model("solar", *YEAR_2020, country, SOLAR_PANEL, REGIONS, tmp_path)

    arrays = precompute.solar(*YEAR_2020, COUNTRY, SOLAR_PANEL, REGIONS, tmp_path)
    energy = precompute.solar_energy(arrays, dc_ac_ratio, horizontal_share=horizontal_share)

    assert energy.index.equals(modelled.index)
    # Only the order of the sums over cells and systems differs from the model.
    np.testing.assert_allclose(energy, modelled["energy_mwh"], rtol=1e-12)
    columns = ["complete", "capacity_mw", "capacity_provisional", "era5_source"]
    pd.testing.assert_frame_equal(arrays.covariates[columns], modelled[columns])
    assert not arrays.covariates.loc[GAP.tz_localize("UTC"), "complete"]
    assert arrays.covariates["capacity_provisional"].any()
    assert arrays.covariates["era5_source"].eq("ERA5T").any()


def mid_hours(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    return pd.date_range(start, end, freq="h", inclusive="left") + pd.Timedelta(minutes=30)


def sky_of_cells(cells: xr.Dataset, times: pd.DatetimeIndex) -> tuple[FloatArray, FloatArray]:
    """Apparent sun elevation and clear-sky irradiance of each cell at `times`, shape (time, cell),
    from pvlib with the air temperature of the cell and the sun at sea level, as
    pv.solar_position places it."""
    elevation, clear = [], []
    for index in range(cells.sizes["cell"]):
        site = pvlib.location.Location(
            float(cells["latitude"][index]), float(cells["longitude"][index])
        )
        temp_air = cells["t2m"].isel(cell=index).to_numpy().astype(float) - pv.KELVIN
        position = pvlib.solarposition.get_solarposition(
            times, site.latitude, site.longitude, temperature=temp_air
        )
        elevation.append(position["apparent_elevation"].to_numpy())
        clear.append(site.get_clearsky(times, solar_position=position)["ghi"].to_numpy())
    return np.column_stack(elevation), np.column_stack(clear)


def test_solar_covariates_are_capacity_weighted_means_at_mid_hour(
    store: WeatherStore, tmp_path: Path
) -> None:
    start, end = utc("2020-06-01"), utc("2020-06-03")

    covariates = precompute.solar(start, end, COUNTRY, SOLAR_PANEL, REGIONS, tmp_path).covariates

    weather = physics.read_weather(
        "FR", BOX, physics.INPUTS["solar"], start.tz_localize(None), end.tz_localize(None), tmp_path
    )
    cells, (capacity,), _ = physics.fleet_cells("solar", weather, COUNTRY, SOLAR_PANEL, REGIONS)
    weights = capacity.transpose("time", "cell").to_numpy()
    elevation, clear = sky_of_cells(cells, mid_hours(start, end))
    ghi = cells["ssrd"].transpose("time", "cell").to_numpy()
    temp_air = cells["t2m"].transpose("time", "cell").to_numpy().astype(float) - pv.KELVIN

    for column, values in [
        ("sun_elevation_deg", elevation),
        ("ghi_w_m2", ghi),
        ("clear_sky_ghi_w_m2", clear),
        ("air_temperature_c", temp_air),
    ]:
        expected = np.average(values, axis=1, weights=weights)
        np.testing.assert_allclose(covariates[column], expected, rtol=1e-12, atol=1e-12)
    assert covariates["clear_sky_ghi_w_m2"].max() > 500


def test_the_solar_centroid_weights_the_cells_by_their_capacity(
    store: WeatherStore, tmp_path: Path
) -> None:
    start, end = utc("2020-06-01"), utc("2020-06-03")

    centroid = precompute.solar(start, end, COUNTRY, SOLAR_PANEL, REGIONS, tmp_path).centroid

    weather = physics.read_weather(
        "FR", BOX, physics.INPUTS["solar"], start.tz_localize(None), end.tz_localize(None), tmp_path
    )
    cells, (capacity,), _ = physics.fleet_cells("solar", weather, COUNTRY, SOLAR_PANEL, REGIONS)
    per_cell = capacity.sum("time").to_numpy()
    expected = [np.average(cells[name], weights=per_cell) for name in ("latitude", "longitude")]
    assert centroid == pytest.approx(expected, rel=1e-12)
    # "02", with most of the capacity on its two cells, pulls the centroid north and east of the
    # plain mean of the three cells, 45.25 N and 2.17 E.
    assert centroid[0] > 45.25
    assert centroid[1] > 2.17


def solar_arrays(hours: pd.DatetimeIndex) -> precompute.SolarArrays:
    rng = np.random.default_rng(0)
    covariates = pd.DataFrame(
        {
            "complete": True,
            "capacity_mw": 100.0,
            "capacity_provisional": False,
            "era5_source": "ERA5",
            "sun_elevation_deg": 30.0,
            "ghi_w_m2": 500.0,
            "clear_sky_ghi_w_m2": 700.0,
            "air_temperature_c": 20.0,
        },
        index=hours.as_unit("ns").rename("time"),
    )
    return precompute.SolarArrays(
        fleet_mwh=rng.uniform(0.0, 100.0, (hours.size, 2, precompute.DC_AC_GRID.size)),
        horizontal_mwh=rng.uniform(0.0, 100.0, (hours.size, precompute.DC_AC_GRID.size)),
        shares=np.array([0.25, 0.75]),
        covariates=precompute.Covariates.validate(covariates),
        centroid=(46.0, 2.0),
    )


def test_solar_parameters_act_on_the_ac_output() -> None:
    # Spring equinox, June solstice and autumn equinox of 2025: days 79, 172 and 265.
    hours = pd.DatetimeIndex(["2025-03-20 12:00", "2025-06-21 12:00", "2025-09-22 12:00"], tz="UTC")
    arrays = solar_arrays(hours)
    column = int(np.flatnonzero(precompute.DC_AC_GRID == 1.3)[0])
    fleet = arrays.fleet_mwh[:, :, column] @ arrays.shares
    horizontal = arrays.horizontal_mwh[:, column]

    flat = precompute.solar_energy(arrays, 1.3, level=0.8, horizontal_share=0.4)
    seasonal = precompute.solar_energy(
        arrays, 1.3, level=0.8, horizontal_share=0.4, seasonal_amplitude=0.05
    )

    np.testing.assert_allclose(flat, 0.8 * (0.6 * fleet + 0.4 * horizontal), rtol=1e-14)
    ratio = (seasonal / flat).to_numpy()
    days = np.array([79, 172, 265])
    np.testing.assert_allclose(ratio, 1 + 0.05 * np.sin(2 * np.pi * (days - 172) / 365.25))
    # A positive amplitude lowers spring and raises autumn; the solstice is unchanged.
    assert ratio[0] < ratio[1] == 1.0 < ratio[2]
    assert seasonal.index.equals(hours)


def test_solar_ratio_must_be_on_the_grid() -> None:
    arrays = solar_arrays(pd.DatetimeIndex(["2025-06-21 12:00"], tz="UTC"))

    with pytest.raises(ValueError, match="not on the grid"):
        precompute.solar_energy(arrays, 1.205)


def test_dc_ac_grid_spans_0_90_to_1_80_in_steps_of_0_01() -> None:
    assert precompute.DC_AC_GRID.tolist() == [round(0.9 + 0.01 * step, 2) for step in range(91)]


# 40 by 25 cells, all in one region, so that binning errors average out over the cells as they do
# over the mainland fleet.
WIDE_BOX = era5.Box(west=0.0, south=45.0, east=9.75, north=51.0)
WIDE_REGIONS = gpd.GeoDataFrame(
    {"region": ["01"]},
    geometry=[box(*cell(45.0, 0.0)[:2], *cell(51.0, 9.75)[2:])],
    crs="EPSG:4326",
)
# Windows that end in 2012 and 2018, so that all three hold capacity in 2020.
WIND_COUNTRY = dataclasses.replace(
    COUNTRY,
    era5_box=WIDE_BOX,
    wind_onshore=(
        WindGeneration("VestasV82_1.65MW_82", 80.0, 2012),
        WindGeneration("IEA_Reference_3.4MW_130", 90.0, 2018),
        WindGeneration("IEA_Reference_3.4MW_130", 100.0),
    ),
)
WIND_PANEL = panel(
    "wind_onshore", {"01": {"2012-12-31": 1000.0, "2018-12-31": 2500.0, "2020-12-31": 3500.0}}
)
TWO_WEEKS = (utc("2020-06-01"), utc("2020-06-15"))


@pytest.fixture
def uncached(monkeypatch: pytest.MonkeyPatch) -> None:
    # Years are not final yet, so read_weather reads each period without writing a cache.
    monkeypatch.setattr(physics, "era5_final_until", lambda today: pd.Timestamp("2019-01-01"))


def per_cell_wind(speed_scale: float, tmp_path: Path) -> FloatArray:
    """Onshore wind output of WIND_COUNTRY over TWO_WEEKS computed cell by cell, as in
    physics.model, with hub-height speeds multiplied by `speed_scale`."""
    start, end = (stamp.tz_localize(None) for stamp in TWO_WEEKS)
    weather = physics.read_weather(
        "FR", WIDE_BOX, physics.INPUTS["wind_onshore"], start, end, tmp_path
    )
    cells, capacities, _ = physics.fleet_cells(
        "wind_onshore", weather, WIND_COUNTRY, WIND_PANEL, WIDE_REGIONS
    )
    energy = np.zeros(cells.sizes["time"])
    for generation, capacity in zip(WIND_COUNTRY.wind_onshore, capacities, strict=True):
        parameters = dataclasses.replace(
            physics.generation_parameters(generation), speed_scale=speed_scale
        )
        factor = wind.capacity_factor(cells, parameters)
        energy += (factor * capacity).sum("cell", skipna=False).to_numpy()
    return energy


def test_wind_matches_the_model_within_the_binning_error(
    store: WeatherStore, uncached: None, tmp_path: Path
) -> None:
    modelled = physics.model(
        "wind_onshore", *TWO_WEEKS, WIND_COUNTRY, WIND_PANEL, WIDE_REGIONS, tmp_path
    )

    arrays = precompute.wind_onshore(*TWO_WEEKS, WIND_COUNTRY, WIND_PANEL, WIDE_REGIONS, tmp_path)

    energy = precompute.wind_energy(arrays)
    assert energy.index.equals(modelled.index)
    np.testing.assert_array_equal(energy.isna(), modelled["energy_mwh"].isna())
    columns = ["complete", "capacity_mw", "capacity_provisional", "era5_source"]
    pd.testing.assert_frame_equal(arrays.covariates[columns], modelled[columns])
    assert not arrays.covariates.loc[GAP.tz_localize("UTC"), "complete"]
    # Bounds of the binning error measured on the mainland fleet over 2019-2022, at the model's
    # speeds and with the speeds scaled at hub height.
    for speed_scale in [1.0, 0.8, 1.2]:
        scaled = precompute.wind_energy(arrays, speed_scale).to_numpy()
        expected = per_cell_wind(speed_scale, tmp_path)
        if speed_scale == 1.0:
            np.testing.assert_array_equal(expected, modelled["energy_mwh"])
        assert np.nanmax(np.abs(scaled - expected)) <= 7e-4 * np.nanmax(expected)
        assert np.nansum(scaled) == pytest.approx(np.nansum(expected), rel=1e-5)


def test_wind_covariates_refer_to_the_wind_fleet(store: WeatherStore, tmp_path: Path) -> None:
    start, end = utc("2020-06-01"), utc("2020-06-03")
    # The second window adds capacity in "01" only, so that each window has its own weights.
    monthly = panel(
        "wind_onshore",
        {
            "01": {"2012-12-31": 100.0, "2020-12-31": 500.0},
            "02": {"2012-12-31": 300.0, "2020-12-31": 300.0},
        },
    )

    covariates = precompute.wind_onshore(start, end, COUNTRY, monthly, REGIONS, tmp_path).covariates

    weather = physics.read_weather(
        "FR", BOX, physics.WEATHER, start.tz_localize(None), end.tz_localize(None), tmp_path
    )
    cells, capacities, _ = physics.fleet_cells("wind_onshore", weather, COUNTRY, monthly, REGIONS)
    weights = np.sum([capacity.transpose("time", "cell") for capacity in capacities], axis=0)
    latitude = float(np.average(cells["latitude"], weights=weights.sum(axis=0)))
    longitude = float(np.average(cells["longitude"], weights=weights.sum(axis=0)))
    times = mid_hours(start, end)
    centroid = pvlib.solarposition.get_solarposition(times, latitude, longitude)
    _, clear = sky_of_cells(cells, times)
    ghi = cells["ssrd"].transpose("time", "cell").to_numpy()
    temp_air = cells["t2m"].transpose("time", "cell").to_numpy().astype(float) - pv.KELVIN

    np.testing.assert_allclose(
        covariates["sun_elevation_deg"], centroid["apparent_elevation"], rtol=1e-12, atol=1e-12
    )
    for column, values in [
        ("ghi_w_m2", ghi),
        ("clear_sky_ghi_w_m2", clear),
        ("air_temperature_c", temp_air),
    ]:
        expected = np.average(values, axis=1, weights=weights)
        np.testing.assert_allclose(covariates[column], expected, rtol=1e-12, atol=1e-12)
    # The plain mean latitude of the three cells is 45.25, and the first window alone puts the
    # centroid above it.
    assert 45.0 < latitude < 45.25


def test_wind_speeds_beyond_the_power_curves_are_refused(
    store: WeatherStore, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def storm(
        variables: Sequence[str], bounds: era5.Box, start: pd.Timestamp, end: pd.Timestamp
    ) -> xr.Dataset:
        weather = store(variables, bounds, start, end)
        return weather.assign(u100=10 * weather["u100"], u10=10 * weather["u10"])

    monkeypatch.setattr(era5, "read_era5", storm)
    monthly = panel("wind_onshore", {"01": {"2012-12-31": 100.0, "2020-12-31": 100.0}})

    with pytest.raises(ValueError, match="m/s or more"):
        precompute.wind_onshore(*TWO_WEEKS, COUNTRY, monthly, REGIONS, tmp_path)
