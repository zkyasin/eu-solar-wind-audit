"""Hourly price classes of the evaluation plan, section Price classes."""

import numpy as np
import pandas as pd

REGIMES = ["negative", "near_zero", "reference", "buffer", "missing"]
NEAR_ZERO_NEGATIVE_FROM = -0.01  # EUR/MWh
REFERENCE_FROM = 5.0  # EUR/MWh


def classify_hours(prices: pd.Series) -> pd.DataFrame:
    """Price class of each UTC hour from day-ahead prices in EUR/MWh.

    `prices` has a UTC index labelled at the start of each interval. It is read as quarter-hourly if
    any timestamp falls after the start of an hour, else as hourly. The result covers every hour
    from the first to the last price.
    """
    index = pd.DatetimeIndex(prices.index)
    if index.has_duplicates or (index != index.floor("15min")).any():
        raise ValueError("expected unique timestamps on the quarter-hour")
    expected = 4 if (index.minute != 0).any() else 1

    hours = prices.resample("1h")
    # Prices have two decimals, so rounding only removes float noise at the class boundaries.
    mean = hours.mean().round(6)
    missing = hours.count() < expected
    has_negative_quarter = hours.min() < 0

    negative = ~missing & (mean < 0)
    # A missing hour may have been negative, so its neighbours are kept out of reference too.
    unsure = negative | missing
    next_to_unsure = unsure.shift(1, fill_value=False) | unsure.shift(-1, fill_value=False)
    regime = np.select(
        [missing, negative, mean < REFERENCE_FROM, has_negative_quarter | next_to_unsure],
        ["missing", "negative", "near_zero", "buffer"],
        default="reference",
    )
    return pd.DataFrame(
        {
            "mean_price": mean.where(~missing),
            "has_negative_quarter": has_negative_quarter,
            "near_zero_negative": negative & (mean >= NEAR_ZERO_NEGATIVE_FROM),
            "regime": pd.Categorical(regime, categories=REGIMES),
        }
    )
