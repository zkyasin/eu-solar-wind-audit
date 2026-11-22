from collections.abc import Sequence
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pandera.errors import SchemaError
from shapely.geometry import box

from eu_solar_wind_audit import physics, pv, wind
from eu_solar_wind_audit.config import Country, WindGeneration
from eu_solar_wind_audit.physics import read_weather, split_by_commissioning
from eu_solar_wind_audit.pv import PVParameters, System
from eu_solar_wind_audit.sources import era5, natural_earth, sdes

SDES_SOLAR = Path(__file__).parent / "fixtures" / "sdes" / "solaire-pv-2026-t2.xlsx"
STEP = 0.25
LATITUDE = np.array([45.0, 45.25, 45.5])
LONGITUDE = np.array([2.0, 2.25, 2.5, 2.75])
BOX = era5.Box(west=2.0, south=45.0, east=2.75, north=45.5)
# Hours from this stamp on are ERA5T in the synthetic store.
ERA5T_FROM = pd.Timestamp("2021-01-01")

COUNTRY = Country(
    country="FR",
    time_zone="Europe/Paris",
    perimeter="metropolitan_without_corsica",
    era5_box=BOX,
    solar=PVParameters(fleet=(System(tilt=30.0, azimuth=180.0, share=1.0),)),
    wind_onshore=(
        WindGeneration("VestasV82_1.65MW_82", 80.0, 2012),
        WindGeneration("IEA_Reference_3.4MW_130", 90.0, 2020),
        WindGeneration("IEA_Reference_3.4MW_130", 100.0),
    ),
)


class FakeStore:
    """Stands in for era5.read_era5 and records the periods it is asked for."""

    def __init__(self) -> None:
        self.calls: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    def __call__(
        self, variables: Sequence[str], box: era5.Box, start: pd.Timestamp, end: pd.Timestamp
    ) -> xr.Dataset:
        self.calls.append((start, end))
        time = pd.date_range(start, end, freq="h", inclusive="left")
        # Each field holds its hour count since 2019, so that slices can be told apart.
        values = np.broadcast_to(
            ((time - pd.Timestamp("2019-01-01")) / pd.Timedelta(hours=1)).to_numpy()[:, None, None],
            (time.size, LATITUDE.size, LONGITUDE.size),
        ).astype("float32")
        fields = {name: (("time", "latitude", "longitude"), values) for name in variables}
        source = np.where(time < ERA5T_FROM, "ERA5", "ERA5T")
        return xr.Dataset(
            fields,
            coords={
                "time": time,
                "latitude": LATITUDE,
                "longitude": LONGITUDE,
                "source": ("time", source),
            },
        )


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch) -> FakeStore:
    fake = FakeStore()
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


def constant(value: float) -> dict[str, float]:
    # Every month end from before the end of the first wind window, as in the SDES panel.
    return {str(day.date()): value for day in pd.date_range("2011-12-31", "2020-12-31", freq="ME")}


def utc(stamp: str) -> pd.Timestamp:
    return pd.Timestamp(stamp, tz="UTC")


def constant_factor(value: float) -> object:
    def factor(weather: xr.Dataset, parameters: object) -> xr.DataArray:
        return xr.full_like(weather["u10"], value, dtype=float)

    return factor


def run(
    technology: physics.Technology,
    monthly: pd.DataFrame,
    tmp_path: Path,
    start: str = "2020-06-01",
    end: str = "2020-06-02",
) -> pd.DataFrame:
    return physics.model(
        technology,
        utc(start),
        utc(end),
        COUNTRY,
        monthly,
        REGIONS,
        cache_dir=tmp_path,
    )


def modelled_frame() -> pd.DataFrame:
    time = pd.date_range("2025-06-01", periods=2, freq="h", tz="UTC", unit="ns", name="time")
    return pd.DataFrame(
        {
            "energy_mwh": [1.0, np.nan],
            "complete": [True, False],
            "capacity_mw": [10.0, 10.0],
            "capacity_provisional": [False, True],
            "era5_source": ["ERA5", "ERA5T"],
        },
        index=time,
    )


def test_modelled_accepts_a_valid_frame() -> None:
    physics.Modelled.validate(modelled_frame())


@pytest.mark.parametrize(
    ("frame", "error"),
    [
        pytest.param(
            modelled_frame().assign(capacity_mw=[10.0, -1.0]),
            "greater_than_or_equal_to",
            id="negative-capacity",
        ),
        pytest.param(
            modelled_frame().assign(capacity_mw=[10.0, np.nan]), "non-nullable", id="no-capacity"
        ),
        pytest.param(
            modelled_frame().assign(era5_source=["ERA5", "ERA6"]), "isin", id="unknown-era5-source"
        ),
    ],
)
def test_modelled_rejects(frame: pd.DataFrame, error: str) -> None:
    with pytest.raises(SchemaError, match=error):
        physics.Modelled.validate(frame)


def test_model_output_is_validated(
    store: FakeStore, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(pv, "capacity_factor", constant_factor(0.5))
    # Linear from 10 MW on 1 June to -10 MW on 1 July, so below zero after 16 June.
    capacity = {"2020-05-31": 10.0, "2020-06-30": -10.0}

    with pytest.raises(SchemaError, match="greater_than_or_equal_to"):
        run("solar", panel("solar", {"01": capacity}), tmp_path, "2020-06-20", "2020-06-21")


@pytest.mark.parametrize(("technology", "module"), [("solar", pv), ("wind_onshore", wind)])
def test_half_capacity_factor_on_100_mw_gives_50_mwh_per_hour(
    store: FakeStore,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    technology: physics.Technology,
    module: object,
) -> None:
    monkeypatch.setattr(module, "capacity_factor", constant_factor(0.5))
    monthly = panel(technology, {"01": constant(60.0), "02": constant(40.0)})

    result = run(technology, monthly, tmp_path)

    assert len(result) == 24
    assert result["energy_mwh"].to_numpy() == pytest.approx(50.0)
    assert result["capacity_mw"].to_numpy() == pytest.approx(100.0)
    assert result["complete"].all()
    assert result.index[0] == pd.Timestamp("2020-06-01", tz="UTC")


@pytest.mark.parametrize("final_until", ["2026-01-01", "2020-01-01"], ids=["cached", "not-cached"])
def test_hours_are_labelled_by_their_utc_start_in_nanoseconds(
    store: FakeStore, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, final_until: str
) -> None:
    monkeypatch.setattr(pv, "capacity_factor", constant_factor(0.5))
    monkeypatch.setattr(physics, "era5_final_until", lambda today: pd.Timestamp(final_until))

    result = run("solar", panel("solar", {"01": constant(100.0)}), tmp_path)

    assert result.index.equals(pd.date_range("2020-06-01", periods=24, freq="h", tz="UTC"))
    assert result.index.dtype == "datetime64[ns, UTC]"
    assert result.index.name == "time"


def test_capacity_factors_are_computed_on_cells_with_capacity_only(
    store: FakeStore, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: list[xr.Dataset] = []

    def factor(weather: xr.Dataset, parameters: PVParameters) -> xr.DataArray:
        seen.append(weather)
        return xr.full_like(weather["ssrd"], 1.0, dtype=float)

    monkeypatch.setattr(pv, "capacity_factor", factor)

    run("solar", panel("solar", {"02": constant(10.0)}), tmp_path)

    (weather,) = seen
    assert weather.sizes["cell"] == 2
    assert weather["latitude"].to_numpy().tolist() == [45.25, 45.5]
    assert weather["longitude"].to_numpy().tolist() == [2.25, 2.25]


def test_capacity_is_linear_in_time_between_month_ends(
    store: FakeStore, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(pv, "capacity_factor", constant_factor(1.0))
    # 696 MW added over the 696 hours of February 2020.
    capacity = {"2020-01-31": 0.0, "2020-02-29": 696.0, "2020-03-31": 696.0}

    result = run("solar", panel("solar", {"01": capacity}), tmp_path, "2020-02-01", "2020-03-02")

    february = result.loc["2020-02"]
    assert february["capacity_mw"].to_numpy() == pytest.approx(np.arange(696.0))
    assert february["energy_mwh"].to_numpy() == pytest.approx(np.arange(696.0))
    assert result["capacity_mw"].loc["2020-03-01"].to_numpy() == pytest.approx(696.0)


def test_hours_outside_the_capacity_panel_are_refused(store: FakeStore, tmp_path: Path) -> None:
    capacity = {"2020-01-31": 10.0, "2020-02-29": 10.0}

    with pytest.raises(ValueError, match="no capacity"):
        run("solar", panel("solar", {"01": capacity}), tmp_path, "2020-03-01", "2020-03-02")


def test_corsica_is_left_out(
    store: FakeStore, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(pv, "capacity_factor", constant_factor(0.5))
    monthly = panel(
        "solar",
        {"01": constant(100.0), "2A": constant(1000.0), "20": constant(1000.0)},
    )

    result = run("solar", monthly, tmp_path)

    assert result["capacity_mw"].to_numpy() == pytest.approx(100.0)
    assert result["energy_mwh"].to_numpy() == pytest.approx(50.0)


def test_capacity_is_split_by_the_window_it_was_added_in() -> None:
    month_ends = pd.DatetimeIndex(["2012-11-30", "2012-12-31", "2020-12-31", "2021-12-31"])
    capacity = pd.DataFrame(
        {"old": [80.0, 100.0, 250.0, 300.0], "new": [0.0, 0.0, 0.0, 50.0]}, index=month_ends
    )

    up_to_2012, up_to_2020, later = split_by_commissioning(capacity, [2012, 2020, None])

    assert up_to_2012["old"].tolist() == [80.0, 100.0, 100.0, 100.0]
    assert up_to_2020["old"].tolist() == [0.0, 0.0, 150.0, 150.0]
    assert later["old"].tolist() == [0.0, 0.0, 0.0, 50.0]
    assert later["new"].tolist() == [0.0, 0.0, 0.0, 50.0]
    assert (up_to_2012 + up_to_2020 + later).equals(capacity)


def test_a_decrease_removes_the_newest_capacity_first() -> None:
    month_ends = pd.DatetimeIndex(["2012-12-31", "2020-12-31", "2021-12-31"])
    capacity = pd.DataFrame({"a": [100.0, 250.0, 120.0]}, index=month_ends)

    up_to_2012, up_to_2020, later = split_by_commissioning(capacity, [2012, 2020, None])

    assert up_to_2012["a"].tolist() == [100.0, 100.0, 100.0]
    assert up_to_2020["a"].tolist() == [0.0, 150.0, 20.0]
    assert later["a"].tolist() == [0.0, 0.0, 0.0]


def test_a_window_with_a_net_decrease_gets_no_capacity() -> None:
    month_ends = pd.DatetimeIndex(["2012-12-31", "2020-12-31", "2021-12-31"])
    capacity = pd.DataFrame({"a": [100.0, 80.0, 150.0]}, index=month_ends)

    up_to_2012, up_to_2020, later = split_by_commissioning(capacity, [2012, 2020, None])

    assert up_to_2012["a"].tolist() == [100.0, 80.0, 100.0]
    assert up_to_2020["a"].tolist() == [0.0, 0.0, 0.0]
    assert later["a"].tolist() == [0.0, 0.0, 50.0]


def test_window_ending_before_the_panel_is_refused() -> None:
    capacity = pd.DataFrame({"a": [1.0]}, index=pd.DatetimeIndex(["2015-03-31"]))

    with pytest.raises(ValueError, match="starts after"):
        split_by_commissioning(capacity, [2012, None])


def test_each_generation_uses_its_own_turbine_and_capacity(
    store: FakeStore, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def factor(weather: xr.Dataset, parameters: wind.WindParameters) -> xr.DataArray:
        (member,) = parameters.fleet
        return xr.full_like(weather["u10"], member.hub_height_m / 1000, dtype=float)

    monkeypatch.setattr(wind, "capacity_factor", factor)
    capacity = {"2012-12-31": 100.0, "2020-12-31": 300.0, "2021-12-31": 600.0}

    result = run("wind_onshore", panel("wind_onshore", {"01": capacity}), tmp_path)

    # 100 MW of the first window (80 m); the second window (90 m) grows linearly from 0 to
    # 200 MW between the instants 2013-01-01 and 2021-01-01.
    hours = pd.DatetimeIndex(result.index).tz_localize(None)
    window = pd.Timestamp("2021-01-01") - pd.Timestamp("2013-01-01")
    added = 200.0 * ((hours - pd.Timestamp("2013-01-01")) / window).to_numpy()
    expected = 100.0 * 0.08 + added * 0.09
    assert result["energy_mwh"].to_numpy() == pytest.approx(expected)


def test_era5_source_and_missing_hours_are_carried(
    store: FakeStore, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def factor(weather: xr.Dataset, parameters: PVParameters) -> xr.DataArray:
        # NaN in one cell for the first hour only.
        values = xr.full_like(weather["ssrd"], 1.0, dtype=float)
        return values.where((values["time"] > values["time"][0]) | (values["cell"] > 0))

    monkeypatch.setattr(pv, "capacity_factor", factor)
    monkeypatch.setattr(physics, "era5_final_until", lambda today: pd.Timestamp("2026-01-01"))
    capacity = {"2020-11-30": 10.0, "2020-12-31": 10.0, "2021-01-31": 10.0}

    result = run("solar", panel("solar", {"02": capacity}), tmp_path, "2020-12-31", "2021-01-02")

    assert result["era5_source"].loc["2020-12-31"].eq("ERA5").all()
    assert result["era5_source"].loc["2021-01-01"].eq("ERA5T").all()
    assert not result["complete"].iloc[0]
    assert np.isnan(result["energy_mwh"].iloc[0])
    assert result["complete"].iloc[1:].all()


def test_complete_year_is_read_once_and_kept(store: FakeStore, tmp_path: Path) -> None:
    start, end = pd.Timestamp("2020-03-01"), pd.Timestamp("2020-03-04")

    first = read_weather("FR", BOX, physics.WEATHER, start, end, tmp_path)
    second = read_weather("FR", BOX, physics.WEATHER, start, end, tmp_path)

    assert store.calls == [(pd.Timestamp("2020-01-01"), pd.Timestamp("2021-01-01"))]
    assert (tmp_path / "fr" / "2020.zarr").is_dir()
    xr.testing.assert_identical(first.load(), second.load())
    assert first.indexes["time"].equals(pd.date_range(start, end, freq="h", inclusive="left"))
    hours_since_2019 = (start - pd.Timestamp("2019-01-01")) / pd.Timedelta(hours=1)
    assert float(first["u100"][0, 0, 0]) == hours_since_2019


def test_last_incomplete_year_is_read_again(
    store: FakeStore, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(physics, "era5_final_until", lambda today: pd.Timestamp("2020-06-01"))
    start, end = pd.Timestamp("2020-03-01"), pd.Timestamp("2020-03-04")

    read_weather("FR", BOX, physics.WEATHER, start, end, tmp_path)
    read_weather("FR", BOX, physics.WEATHER, start, end, tmp_path)

    assert store.calls == [(start, end), (start, end)]
    assert not (tmp_path / "fr").exists()


def test_year_holding_era5t_is_not_kept(
    store: FakeStore, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(physics, "era5_final_until", lambda today: pd.Timestamp("2026-01-01"))
    start, end = pd.Timestamp("2021-03-01"), pd.Timestamp("2021-03-02")

    read_weather("FR", BOX, physics.WEATHER, start, end, tmp_path)
    weather = read_weather("FR", BOX, physics.WEATHER, start, end, tmp_path)

    year = (pd.Timestamp("2021-01-01"), pd.Timestamp("2022-01-01"))
    assert store.calls == [year, year]
    assert not (tmp_path / "fr").exists()
    assert weather["source"].to_numpy().tolist() == ["ERA5T"] * 24


@pytest.mark.parametrize(
    ("today", "final_until"),
    [
        ("2027-02-28 23:00", "2026-12-01"),
        ("2027-03-01 00:00", "2027-01-01"),
        ("2026-11-22 12:00", "2026-09-01"),
    ],
)
def test_final_era5_is_expected_two_months_after_each_month(today: str, final_until: str) -> None:
    assert physics.era5_final_until(pd.Timestamp(today)) == pd.Timestamp(final_until)


def test_period_across_new_year_joins_both_years(store: FakeStore, tmp_path: Path) -> None:
    start, end = pd.Timestamp("2019-12-31 22:00"), pd.Timestamp("2020-01-01 02:00")

    weather = read_weather("FR", BOX, physics.WEATHER, start, end, tmp_path)

    assert weather.indexes["time"].equals(pd.date_range(start, end, freq="h", inclusive="left"))
    assert len(store.calls) == 2


def test_cache_of_another_box_is_refused(store: FakeStore, tmp_path: Path) -> None:
    start, end = pd.Timestamp("2020-03-01"), pd.Timestamp("2020-03-02")
    read_weather("FR", BOX, physics.WEATHER, start, end, tmp_path)

    with pytest.raises(ValueError, match="box"):
        read_weather("FR", BOX._replace(west=1.0), physics.WEATHER, start, end, tmp_path)


def test_model_france_runs_one_calendar_year_at_a_time(
    store: FakeStore, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(pv, "capacity_factor", constant_factor(0.5))
    monthly = panel("solar", {"01": constant(100.0)})
    monkeypatch.setattr(physics, "load_monthly_capacity", lambda technology, until: monthly)
    monkeypatch.setattr(natural_earth, "load_admin1", lambda country: REGIONS)
    periods: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    model = physics.model

    def spy(
        technology: physics.Technology, start: pd.Timestamp, end: pd.Timestamp, *args: Any
    ) -> pd.DataFrame:
        periods.append((start, end))
        return model(technology, start, end, *args)

    monkeypatch.setattr(physics, "model", spy)
    start, end = utc("2019-12-31 22:00"), utc("2020-01-01 02:00")

    result = physics.model_france("solar", start, end, COUNTRY, cache_dir=tmp_path)

    assert periods == [(start, utc("2020-01-01")), (utc("2020-01-01"), end)]
    expected = pd.date_range(start, end, freq="h", inclusive="left", name="time")
    assert result.index.equals(expected)
    assert result["energy_mwh"].to_numpy() == pytest.approx(50.0)


def test_capacity_is_loaded_up_to_the_month_of_the_last_utc_hour(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sdes, "download_dashboard", lambda client, media_id: SDES_SOLAR)

    monthly = physics.load_monthly_capacity("solar", utc("2026-07-31 23:00"))

    assert monthly["month_end"].max() == pd.Timestamp("2026-07-31")


WRONG_PERIODS = [
    pytest.param(utc("2025-01-01"), utc("2025-01-01"), "empty period", id="empty"),
    pytest.param(pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-02"), "hour in UTC", id="naive"),
    pytest.param(
        pd.Timestamp("2025-01-01", tz="Europe/Paris"), utc("2025-01-02"), "hour in UTC", id="paris"
    ),
    pytest.param(utc("2025-01-01 00:30"), utc("2025-01-02"), "hour in UTC", id="off-hour"),
]


@pytest.mark.parametrize(("start", "end", "error"), WRONG_PERIODS)
def test_wrong_period_is_refused_before_any_download(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
    error: str,
) -> None:
    def no_download(technology: physics.Technology, until: pd.Timestamp) -> pd.DataFrame:
        raise AssertionError("capacity was downloaded")

    monkeypatch.setattr(physics, "load_monthly_capacity", no_download)

    with pytest.raises(ValueError, match=error):
        physics.model_france("solar", start, end, COUNTRY, cache_dir=tmp_path)


@pytest.mark.parametrize(("start", "end", "error"), WRONG_PERIODS)
def test_wrong_period_is_refused_before_reading_era5(
    store: FakeStore, tmp_path: Path, start: pd.Timestamp, end: pd.Timestamp, error: str
) -> None:
    monthly = panel("solar", {"01": constant(1.0)})

    with pytest.raises(ValueError, match=error):
        physics.model("solar", start, end, COUNTRY, monthly, REGIONS, tmp_path)
    assert store.calls == []


def test_hours_after_the_last_final_month_end_use_provisional_capacity(
    store: FakeStore, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(pv, "capacity_factor", constant_factor(1.0))
    monthly = panel("solar", {"01": {"2019-12-31": 10.0, "2020-01-31": 10.0, "2020-02-29": 20.0}})
    monthly["provisional"] = monthly["month_end"] == pd.Timestamp("2020-02-29")

    result = run("solar", monthly, tmp_path, "2020-01-31 22:00", "2020-02-01 02:00")

    # The 31 January figure holds at 1 February 00:00; later hours lean on the February figure.
    assert result["capacity_provisional"].tolist() == [False, False, False, True]
