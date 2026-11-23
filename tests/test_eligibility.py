import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import NamedTuple

import geopandas as gpd
import httpx
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pandera.errors import SchemaError
from shapely.geometry import box

from eu_solar_wind_audit import eligibility, physics, precompute, regimes
from eu_solar_wind_audit.config import Country, WindGeneration
from eu_solar_wind_audit.pv import PVParameters, System
from eu_solar_wind_audit.sources import era5, natural_earth, odre, smard

SHARED_RULES = [
    "consolidated_or_definitive",
    "complete",
    "reference_price",
    "final_capacity",
    "final_era5",
    "training_year",
]


class Inputs(NamedTuple):
    published: pd.DataFrame
    covariates: pd.DataFrame
    # Hourly day-ahead prices in EUR/MWh.
    prices: pd.Series


def utc_hours(start: str | pd.Timestamp, periods: int = 4) -> pd.DatetimeIndex:
    return pd.date_range(start, periods=periods, freq="h", tz="UTC", unit="ns", name="time")


def inputs(hours: pd.DatetimeIndex) -> Inputs:
    """Inputs on `hours` that meet every rule."""
    published = pd.DataFrame(
        {"energy_mwh": 100.0, "complete": True, "status": "definitive"}, index=hours
    )
    covariates = pd.DataFrame(
        {
            "complete": True,
            "capacity_mw": 1000.0,
            "capacity_provisional": False,
            "era5_source": "ERA5",
            "sun_elevation_deg": 30.0,
            "ghi_w_m2": 500.0,
            "clear_sky_ghi_w_m2": 600.0,
            "air_temperature_c": 20.0,
        },
        index=hours,
    )
    return Inputs(published, covariates, pd.Series(50.0, index=hours))


def rules(technology: physics.Technology, given: Inputs) -> pd.DataFrame:
    classes = regimes.classify_hours(given.prices)
    return eligibility.eligible_hours(technology, given.published, given.covariates, classes)


HOURS = utc_hours("2021-06-01 10:00")
SECOND_HOUR = HOURS[1]


def realtime(given: Inputs) -> None:
    given.published.loc[SECOND_HOUR, "status"] = "realtime"


def unpublished(given: Inputs) -> None:
    given.published.loc[SECOND_HOUR, "energy_mwh"] = np.nan
    given.published.loc[SECOND_HOUR, "complete"] = False


def unmodelled(given: Inputs) -> None:
    given.covariates.loc[SECOND_HOUR, "complete"] = False


def near_zero_price(given: Inputs) -> None:
    # A near-zero hour leaves its neighbours in the reference class.
    given.prices.iloc[1] = 2.0


def provisional_capacity(given: Inputs) -> None:
    given.covariates.loc[SECOND_HOUR, "capacity_provisional"] = True


def era5t(given: Inputs) -> None:
    given.covariates.loc[SECOND_HOUR, "era5_source"] = "ERA5T"


@pytest.mark.parametrize("technology", ["solar", "wind_onshore"])
@pytest.mark.parametrize(
    ("rule", "change"),
    [
        pytest.param("consolidated_or_definitive", realtime, id="realtime"),
        pytest.param("complete", unpublished, id="unpublished"),
        pytest.param("complete", unmodelled, id="unmodelled"),
        pytest.param("reference_price", near_zero_price, id="near-zero-price"),
        pytest.param("final_capacity", provisional_capacity, id="provisional-capacity"),
        pytest.param("final_era5", era5t, id="era5t"),
    ],
)
def test_each_shared_rule_leaves_out_its_hours_alone(
    technology: physics.Technology, rule: str, change: Callable[[Inputs], None]
) -> None:
    given = inputs(HOURS)
    change(given)

    found = rules(technology, given)

    assert found[rule].tolist() == [True, False, True, True]
    assert found.drop(columns=[rule, "eligible"]).all(axis=None)
    assert found["eligible"].tolist() == [True, False, True, True]


def test_definitive_and_consolidated_hours_are_final() -> None:
    given = inputs(HOURS)
    given.published["status"] = ["definitive", "consolidated", "realtime", None]

    found = rules("solar", given)

    assert found["consolidated_or_definitive"].tolist() == [True, True, False, False]


@pytest.mark.parametrize(
    ("start", "expected"),
    [
        ("2018-12-31 22:00", [False, False, True, True]),
        ("2025-12-31 22:00", [True, True, False, False]),
    ],
)
def test_training_years_are_2019_to_2025(start: str, expected: list[bool]) -> None:
    found = rules("wind_onshore", inputs(utc_hours(start)))

    assert found["training_year"].tolist() == expected
    assert found["eligible"].tolist() == expected


def test_solar_hours_need_the_sun_above_the_horizon() -> None:
    given = inputs(HOURS)
    given.covariates["sun_elevation_deg"] = [-5.0, 0.0, 0.1, 30.0]

    found = rules("solar", given)

    assert list(found.columns) == [*SHARED_RULES, "sun_above_horizon", "eligible"]
    assert found["sun_above_horizon"].tolist() == [False, False, True, True]
    assert found["eligible"].tolist() == [False, False, True, True]


@pytest.mark.parametrize(
    ("start", "expected"),
    [
        (odre.FIRST_OFFSHORE_POWER - 2 * pd.Timedelta(hours=1), [True, True, False, False]),
        (odre.WIND_SPLIT_START - 2 * pd.Timedelta(hours=1), [False, False, True, True]),
    ],
    ids=["first-offshore-power", "split"],
)
def test_wind_hours_need_the_onshore_reference(start: pd.Timestamp, expected: list[bool]) -> None:
    given = inputs(utc_hours(start))
    # Below the horizon, which only solar hours need.
    given.covariates["sun_elevation_deg"] = -10.0

    found = rules("wind_onshore", given)

    assert list(found.columns) == [*SHARED_RULES, "onshore_reference", "eligible"]
    assert found["onshore_reference"].tolist() == expected
    assert found["eligible"].tolist() == expected


def test_covariates_on_other_hours_are_refused() -> None:
    given = inputs(HOURS)
    later = given._replace(covariates=given.covariates.set_axis(HOURS + pd.Timedelta(hours=1)))

    with pytest.raises(ValueError, match="covariates are not on the hours"):
        rules("solar", later)


def test_price_classes_on_other_hours_are_refused() -> None:
    given = inputs(HOURS)
    shorter = given._replace(prices=given.prices.iloc[:-1])

    with pytest.raises(ValueError, match="price classes are not on the hours"):
        rules("solar", shorter)


def test_eligible_must_meet_every_rule() -> None:
    found = rules("solar", inputs(HOURS))
    found.loc[SECOND_HOUR, "final_era5"] = False

    with pytest.raises(SchemaError, match="eligible_meets_every_rule"):
        eligibility.Eligibility.validate(found)


def smard_files(prices: pd.Series, zone: str = "FR") -> dict[str, bytes]:
    """SMARD chart files that publish `prices`, in EUR/MWh on UTC quarter-hours, as the day-ahead
    price of the bidding `zone` in a single file; NaN is published as null."""
    filter_id, region = smard.PRICES[zone]
    stamps = [int(stamp.timestamp() * 1000) for stamp in prices.index]
    rows = [
        [stamp, None if np.isnan(value) else value]
        for stamp, value in zip(stamps, prices, strict=True)
    ]
    return {
        "index_quarterhour.json": json.dumps({"timestamps": [stamps[0]]}).encode(),
        f"{filter_id}_{region}_quarterhour_{stamps[0]}.json": json.dumps({"series": rows}).encode(),
    }


def serving(files: dict[str, bytes], requests: list[str]) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request.url.path)
        name = request.url.path.rsplit("/", 1)[-1]
        return httpx.Response(200, content=files[name]) if name in files else httpx.Response(404)

    return httpx.Client(transport=httpx.MockTransport(handler))


def quarter_hours(start: str, end: str) -> pd.DatetimeIndex:
    return pd.date_range(start, end, freq="15min", inclusive="left", tz="UTC")


def classes_from(prices: pd.Series, tmp_path: Path, requests: list[str]) -> pd.DataFrame:
    start, end = (
        pd.Timestamp("2022-03-01 00:00", tz="UTC"),
        pd.Timestamp("2022-03-01 06:00", tz="UTC"),
    )
    with serving(smard_files(prices), requests) as client:
        return eligibility.price_classes(start, end, client=client, cache_dir=tmp_path)


def test_price_classes_take_the_hours_around_the_period_into_account(tmp_path: Path) -> None:
    requests: list[str] = []
    prices = pd.Series(50.0, index=quarter_hours("2022-02-28 23:00", "2022-03-01 07:00"))
    # A negative hour before the period and an hour without prices after it.
    prices[prices.index < pd.Timestamp("2022-03-01 00:00", tz="UTC")] = -5.0
    prices[prices.index >= pd.Timestamp("2022-03-01 06:00", tz="UTC")] = np.nan

    classes = classes_from(prices, tmp_path, requests)

    assert classes.index.equals(utc_hours("2022-03-01 00:00", periods=6))
    expected = ["buffer"] + ["reference"] * 4 + ["buffer"]
    assert classes["regime"].tolist() == expected
    assert requests[0] == "/app/chart_data/254/DE/index_quarterhour.json"


def test_a_quarter_hour_without_a_price_makes_its_hour_missing(tmp_path: Path) -> None:
    prices = pd.Series(50.0, index=quarter_hours("2022-02-28 23:00", "2022-03-01 07:00"))
    absent = pd.Timestamp("2022-03-01 02:15", tz="UTC")

    classes = classes_from(prices.drop([absent]), tmp_path, [])

    expected = ["reference", "buffer", "missing", "buffer", "reference", "reference"]
    assert classes["regime"].tolist() == expected


def test_price_classes_of_another_zone_come_from_its_own_prices(tmp_path: Path) -> None:
    requests: list[str] = []
    prices = pd.Series(50.0, index=quarter_hours("2022-02-28 23:00", "2022-03-01 07:00"))
    prices[prices.index >= pd.Timestamp("2022-03-01 03:00", tz="UTC")] = -5.0
    start, end = pd.Timestamp("2022-03-01", tz="UTC"), pd.Timestamp("2022-03-01 06:00", tz="UTC")

    with serving(smard_files(prices, "DE-LU"), requests) as client:
        classes = eligibility.price_classes(
            start, end, zone="DE-LU", client=client, cache_dir=tmp_path
        )

    expected = ["reference", "reference", "buffer", "negative", "negative", "negative"]
    assert classes["regime"].tolist() == expected
    week_ms = int(prices.index[0].timestamp() * 1000)
    assert requests == [
        "/app/chart_data/4169/DE-LU/index_quarterhour.json",
        f"/app/chart_data/4169/DE-LU/4169_DE-LU_quarterhour_{week_ms}.json",
    ]


def test_prices_that_smard_does_not_publish_are_missing(tmp_path: Path) -> None:
    # Nothing is published for the hour before the period or for its last hour.
    prices = pd.Series(50.0, index=quarter_hours("2022-03-01 00:00", "2022-03-01 05:00"))

    classes = classes_from(prices, tmp_path, [])

    assert classes.index.equals(utc_hours("2022-03-01 00:00", periods=6))
    expected = ["buffer", "reference", "reference", "reference", "buffer", "missing"]
    assert classes["regime"].tolist() == expected


# Four ERA5 cells of 0.25 degree, all in department "01".
BOX = era5.Box(west=2.0, south=45.0, east=2.25, north=45.25)
REGIONS = gpd.GeoDataFrame(
    {"region": ["01"]}, geometry=[box(1.875, 44.875, 2.375, 45.375)], crs="EPSG:4326"
)
COUNTRY = Country(
    country="FR",
    time_zone="Europe/Paris",
    perimeter="metropolitan_without_corsica",
    era5_box=BOX,
    solar=PVParameters(fleet=(System(tilt=30.0, azimuth=180.0, share=1.0),), dc_ac_ratio=1.2),
    wind_onshore=(
        WindGeneration("VestasV82_1.65MW_82", 80.0, 2012),
        WindGeneration("IEA_Reference_3.4MW_130", 90.0, 2020),
        WindGeneration("IEA_Reference_3.4MW_130", 100.0),
    ),
)
YEAR_HOURS = utc_hours("2022-01-01", periods=8760)


def weather(
    variables: Sequence[str], bounds: era5.Box, start: pd.Timestamp, end: pd.Timestamp
) -> xr.Dataset:
    """Stands in for era5.read_era5: the same sunny day and steady wind every day."""
    time = pd.date_range(start, end, freq="h", inclusive="left")
    latitude = np.arange(bounds.south, bounds.north + 0.125, 0.25)
    longitude = np.arange(bounds.west, bounds.east + 0.125, 0.25)
    shape = (time.size, latitude.size, longitude.size)
    daylight = np.sin(np.pi * (time.hour.to_numpy() + 0.5 - 5) / 14).clip(0)
    ssrd = np.broadcast_to(800 * daylight[:, None, None], shape)
    fields = {
        "ssrd": ssrd,
        "fdir": 0.6 * ssrd,
        "t2m": np.full(shape, 290.0),
        "u10": np.full(shape, 3.0),
        "v10": np.full(shape, 4.0),
        "u100": np.full(shape, 6.0),
        "v100": np.full(shape, 8.0),
    }
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


def capacity_panel(technology: str) -> pd.DataFrame:
    """Month-end capacity of department "01", as sdes.monthly_capacity gives it."""
    month_ends = pd.DatetimeIndex(["2012-12-31", "2021-12-31", "2022-12-31"])
    return pd.DataFrame(
        {
            "department_code": "01",
            "department": "Ain",
            "region": "Auvergne-Rhône-Alpes",
            "technology": technology,
            "month_end": month_ends,
            "capacity_mw": [100.0, 200.0, 300.0],
            "provisional": False,
        }
    )


def national_export() -> bytes:
    """Definitive national export of 2022: 1000 MW of solar and 2000 MW of wind."""
    stamps = pd.date_range("2022-01-01 00:30", "2023-01-01", freq="30min", tz="UTC", unit="ms")
    export = pd.DataFrame(
        {
            "date_heure": stamps,
            "nature": "Données définitives",
            "consommation": 50000.0,
            "solaire": 1000.0,
            "eolien": 2000.0,
        }
    )
    return export.to_parquet(index=False)


@pytest.fixture
def sources(monkeypatch: pytest.MonkeyPatch) -> list[pd.Timestamp]:
    """Capacity, regions and weather of the tests; returns the hours up to which capacity is
    loaded."""
    capacity_until: list[pd.Timestamp] = []

    def load_monthly_capacity(technology: str, until: pd.Timestamp) -> pd.DataFrame:
        capacity_until.append(until)
        return capacity_panel(technology)

    monkeypatch.setattr(physics, "load_monthly_capacity", load_monthly_capacity)
    monkeypatch.setattr(natural_earth, "load_admin1", lambda country: REGIONS)
    monkeypatch.setattr(era5, "read_era5", weather)
    return capacity_until


def year_2022(technology: physics.Technology, tmp_path: Path) -> eligibility.YearInputs:
    # The hour before 2022 has a negative price.
    prices = pd.Series(50.0, index=quarter_hours("2021-12-31 23:00", "2023-01-01 01:00"))
    prices.iloc[:4] = -5.0
    files = smard_files(prices)
    export = national_export()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "odre.opendatasoft.com":
            return httpx.Response(200, content=export)
        return httpx.Response(200, content=files[request.url.path.rsplit("/", 1)[-1]])

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        return eligibility.year_inputs(
            client,
            technology,
            2022,
            COUNTRY,
            odre_cache_dir=tmp_path / "odre",
            smard_cache_dir=tmp_path / "smard",
            era5_cache_dir=tmp_path / "era5",
        )


def test_year_inputs_of_solar(sources: list[pd.Timestamp], tmp_path: Path) -> None:
    inputs = year_2022("solar", tmp_path)

    assert sources == [pd.Timestamp("2022-12-31 23:00", tz="UTC")]
    assert isinstance(inputs.arrays, precompute.SolarArrays)
    for frame in (inputs.published, inputs.price_classes, inputs.eligibility):
        assert frame.index.equals(YEAR_HOURS)
    assert inputs.published["energy_mwh"].eq(1000.0).all()
    assert inputs.price_classes["regime"].iloc[0] == "buffer"
    sun_up = inputs.arrays.covariates["sun_elevation_deg"] > 0
    assert inputs.eligibility["eligible"].equals(sun_up.rename("eligible"))


def test_year_inputs_of_onshore_wind(sources: list[pd.Timestamp], tmp_path: Path) -> None:
    inputs = year_2022("wind_onshore", tmp_path)

    assert isinstance(inputs.arrays, precompute.WindArrays)
    assert inputs.eligibility.index.equals(YEAR_HOURS)
    onshore = YEAR_HOURS < odre.FIRST_OFFSHORE_POWER
    assert inputs.published.index[inputs.published["complete"]].equals(YEAR_HOURS[onshore])
    # The first hour is next to the negative hour before 2022.
    assert inputs.eligibility.index[inputs.eligibility["eligible"]].equals(YEAR_HOURS[onshore][1:])
