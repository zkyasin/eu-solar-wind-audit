"""Conversion of published series to the common grid: hourly energy in MWh, in UTC, labelled at
the start of the hour.

An hour is complete only when all its values are present. Incomplete hours keep NaN energy: they
are never partially summed or interpolated.
"""

import pandas as pd

HOUR = pd.Timedelta(hours=1)


def hourly_from_period_start_energy(energy_mwh: pd.Series, step: str = "15min") -> pd.DataFrame:
    """Energy in MWh per period labelled at its start (SMARD generation)."""
    length = pd.Timedelta(step)
    energy = _cleaned(energy_mwh, length)
    return _hourly(energy, length, energy_mwh.index.min(), energy_mwh.index.max())


def _cleaned(series: pd.Series, step: pd.Timedelta) -> pd.Series:
    """Sorted float copy without missing values and without stamps that appear more than once."""
    index = series.index
    if not isinstance(index, pd.DatetimeIndex) or str(index.tz) != "UTC":
        raise ValueError("expected a DatetimeIndex in UTC")
    if HOUR % step:
        raise ValueError(f"{step} does not divide an hour")
    if not (index == index.floor(step)).all():
        raise ValueError(f"timestamps are not on the {step} grid")
    series = series.sort_index().astype(float).dropna()
    return series[~series.index.duplicated(keep=False)]


def _hourly(
    energy: pd.Series, step: pd.Timedelta, first: pd.Timestamp, last: pd.Timestamp
) -> pd.DataFrame:
    """Sum energies of periods labelled at their start into the hours from `first` to `last`."""
    hours = pd.date_range(first.floor("h"), last.floor("h"), freq="h")
    by_hour = energy.resample("h")
    complete = by_hour.count().reindex(hours, fill_value=0) == HOUR // step
    total = by_hour.sum().reindex(hours)
    return pd.DataFrame({"energy_mwh": total.where(complete), "complete": complete})
