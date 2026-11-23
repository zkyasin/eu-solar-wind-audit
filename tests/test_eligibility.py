import json
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import httpx
import numpy as np
import pandas as pd
import pytest
from pandera.errors import SchemaError

from eu_solar_wind_audit import eligibility, physics, regimes
from eu_solar_wind_audit.sources import odre

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


def smard_files(prices: pd.Series) -> dict[str, bytes]:
    """SMARD chart files that publish `prices`, in EUR/MWh on UTC quarter-hours, as the French
    day-ahead price in a single file; NaN is published as null."""
    stamps = [int(stamp.timestamp() * 1000) for stamp in prices.index]
    rows = [
        [stamp, None if np.isnan(value) else value]
        for stamp, value in zip(stamps, prices, strict=True)
    ]
    return {
        "index_quarterhour.json": json.dumps({"timestamps": [stamps[0]]}).encode(),
        f"254_DE_quarterhour_{stamps[0]}.json": json.dumps({"series": rows}).encode(),
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
