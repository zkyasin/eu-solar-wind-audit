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

from eu_solar_wind_audit import physics, precompute, pv
from eu_solar_wind_audit.config import Country, WindGeneration
from eu_solar_wind_audit.pv import PVParameters, System
from eu_solar_wind_audit.sources import era5

STEP = 0.25
BOX = era5.Box(west=2.0, south=45.0, east=2.75, north=45.5)
# Every input is missing at this hour in the first cell of the box.
GAP = pd.Timestamp("2020-06-03 12:00")

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
                "source": ("time", np.full(time.size, "ERA5")),
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
        pytest.param((precompute.HORIZONTAL,), 1.8, 1.0, id="horizontal-1.8"),
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
    times = pd.date_range(start, end, freq="h", inclusive="left") + pd.Timedelta(minutes=30)
    elevation, clear = [], []
    for index in range(cells.sizes["cell"]):
        site = pvlib.location.Location(
            float(cells["latitude"][index]), float(cells["longitude"][index])
        )
        temp_air = cells["t2m"].isel(cell=index).to_numpy().astype(float) - pv.KELVIN
        # At sea level, as pv.solar_position places the sun.
        position = pvlib.solarposition.get_solarposition(
            times, site.latitude, site.longitude, temperature=temp_air
        )
        elevation.append(position["apparent_elevation"].to_numpy())
        clear.append(site.get_clearsky(times, solar_position=position)["ghi"].to_numpy())
    ghi = cells["ssrd"].transpose("time", "cell").to_numpy()

    for column, values in [
        ("sun_elevation_deg", np.column_stack(elevation)),
        ("ghi_w_m2", ghi),
        ("clear_sky_ghi_w_m2", np.column_stack(clear)),
    ]:
        expected = np.average(values, axis=1, weights=weights)
        np.testing.assert_allclose(covariates[column], expected, rtol=1e-12, atol=1e-12)
    assert covariates["clear_sky_ghi_w_m2"].max() > 500


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
        },
        index=hours.as_unit("ns").rename("time"),
    )
    return precompute.SolarArrays(
        fleet_mwh=rng.uniform(0.0, 100.0, (hours.size, 2, precompute.DC_AC_GRID.size)),
        horizontal_mwh=rng.uniform(0.0, 100.0, (hours.size, precompute.DC_AC_GRID.size)),
        shares=np.array([0.25, 0.75]),
        covariates=precompute.Covariates.validate(covariates),
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
