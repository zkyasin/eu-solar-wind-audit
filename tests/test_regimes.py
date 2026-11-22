import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from eu_solar_wind_audit.regimes import classify_hours

# Fixture: https://www.smard.de/app/chart_data/254/DE/254_DE_quarterhour_1747000800000.json
# (French day-ahead price, week of 2025-05-12), retrieved 2026-11-22.
FRENCH_WEEK = Path(__file__).parent / "fixtures" / "smard" / "254_DE_quarterhour_1747000800000.json"
START = pd.Timestamp("2025-06-02 00:00", tz="UTC")


def quarter_hours(*hours: list[float]) -> pd.Series:
    values = [value for hour in hours for value in hour]
    return pd.Series(values, index=pd.date_range(START, periods=len(values), freq="15min"))


def hourly(*values: float) -> pd.Series:
    return pd.Series(values, index=pd.date_range(START, periods=len(values), freq="h"))


def french_week() -> pd.Series:
    rows = json.loads(FRENCH_WEEK.read_bytes())["series"]
    index = pd.to_datetime([ms for ms, _ in rows], unit="ms", utc=True)
    return pd.Series([value for _, value in rows], index=index, dtype=float)


@pytest.mark.parametrize(
    ("price", "regime", "near_zero_negative"),
    [
        (-0.02, "negative", False),
        (-0.01, "negative", True),
        (0.0, "near_zero", False),
        (4.99, "near_zero", False),
        (5.0, "reference", False),
    ],
)
def test_class_boundaries(price: float, regime: str, near_zero_negative: bool) -> None:
    hours = classify_hours(quarter_hours([price] * 4))

    assert hours["regime"].iloc[0] == regime
    assert hours["near_zero_negative"].iloc[0] == near_zero_negative


def test_boundaries_hold_despite_float_rounding() -> None:
    # Both hours average exactly 0 and -0.01, but the float sums come out just below.
    hours = classify_hours(quarter_hours([-0.6, -0.6, 0.06, 1.14], [-2.0, -2.0, 1.99, 1.97]))

    assert hours["mean_price"].to_list() == [0.0, -0.01]
    assert hours["regime"].to_list() == ["near_zero", "negative"]
    assert hours["near_zero_negative"].to_list() == [False, True]


def test_hour_is_classed_on_its_mean_not_on_its_quarters() -> None:
    hours = classify_hours(quarter_hours([-1.0, 20.0, 20.0, 20.0], [-30.0, 10.0, 10.0, 9.0]))

    assert hours["mean_price"].to_list() == [14.75, -0.25]
    assert hours["has_negative_quarter"].to_list() == [True, True]
    assert hours["regime"].iloc[0] == "buffer"
    assert hours["regime"].iloc[1] == "negative"


def test_a_negative_quarter_alone_keeps_an_hour_out_of_reference() -> None:
    hours = classify_hours(quarter_hours([50.0] * 4, [-1.0, 20.0, 20.0, 20.0], [50.0] * 4))

    assert hours["regime"].to_list() == ["reference", "buffer", "reference"]


def test_hours_next_to_a_negative_hour_are_buffer() -> None:
    hours = classify_hours(hourly(50, 50, 50, -5, 50, 50, 50))

    assert hours["regime"].to_list() == [
        "reference",
        "reference",
        "buffer",
        "negative",
        "buffer",
        "reference",
        "reference",
    ]


def test_near_zero_hours_do_not_create_a_buffer() -> None:
    hours = classify_hours(hourly(50, 0, 50))

    assert hours["regime"].to_list() == ["reference", "near_zero", "reference"]


def test_an_hour_missing_a_quarter_is_missing() -> None:
    prices = quarter_hours([-10.0] * 4, [-10.0, np.nan, -10.0, -10.0], [50.0] * 4, [50.0] * 4)
    prices = prices.drop(prices.index[9])

    hours = classify_hours(prices)

    assert hours["regime"].to_list() == ["negative", "missing", "missing", "reference"]
    assert hours["mean_price"].isna().to_list() == [False, True, True, False]


def test_hours_absent_from_the_index_are_missing() -> None:
    prices = hourly(50, 50, 50, 50)

    hours = classify_hours(prices.drop(prices.index[1:3]))

    assert hours.index.equals(prices.index)
    assert hours["regime"].to_list() == ["reference", "missing", "missing", "reference"]


def test_hourly_and_quarter_hour_inputs_agree_on_hourly_prices() -> None:
    # Before October 2025 the quarter-hour file repeats each hourly price four times.
    quarters = french_week()
    hourly_prices = quarters.resample("1h").first()

    from_quarters = classify_hours(quarters)
    from_hours = classify_hours(hourly_prices)

    pd.testing.assert_frame_equal(from_quarters, from_hours)
    assert len(from_quarters) == 168
    assert (from_quarters["regime"] == "negative").sum() == 44
    first_negative = pd.Timestamp("2025-05-12 09:00", tz="UTC")
    assert from_quarters.loc[first_negative, "near_zero_negative"]
    assert from_quarters.loc[first_negative - pd.Timedelta(hours=1), "regime"] == "near_zero"
    assert from_quarters.loc[first_negative + pd.Timedelta(hours=6), "regime"] == "buffer"


@pytest.mark.parametrize(
    "index",
    [
        pd.date_range(START, periods=4, freq="10min"),
        pd.DatetimeIndex([START, START, START + pd.Timedelta(minutes=15)]),
    ],
)
def test_timestamps_off_the_quarter_hour_or_duplicated_are_rejected(
    index: pd.DatetimeIndex,
) -> None:
    with pytest.raises(ValueError, match="unique timestamps on the quarter-hour"):
        classify_hours(pd.Series(1.0, index=index))
