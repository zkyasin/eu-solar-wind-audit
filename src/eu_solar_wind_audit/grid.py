"""Conversion of published series to the common grid: hourly energy in MWh, in UTC, labelled at
the start of the hour.

An hour is complete only when all its values are present. Incomplete hours keep NaN energy: they
are never partially summed or interpolated.
"""

import pandas as pd

HOUR = pd.Timedelta(hours=1)


def hourly_from_period_end_means(power_mw: pd.Series, period: str = "30min") -> pd.DataFrame:
    """Average power in MW over periods labelled at their end (éCO2mix consolidated and
    definitive data). Stamps repeated on spring clock-change days are dropped, not resolved."""
    length = pd.Timedelta(period)
    energy = _cleaned(power_mw, length) * (length / HOUR)
    return _hourly(energy.shift(-1, freq=length), length)


def hourly_from_instantaneous(power_mw: pd.Series, step: str = "15min") -> pd.DataFrame:
    """Instantaneous power in MW every `step` (éCO2mix real-time data). Each step contributes the
    trapezoid between its end points, so an hour needs both its boundary points."""
    length = pd.Timedelta(step)
    power = _cleaned(power_mw, length).asfreq(step)
    energy = (power + power.shift(-1)) / 2 * (length / HOUR)
    return _hourly(energy.iloc[:-1], length)


def hourly_from_period_start_energy(energy_mwh: pd.Series, step: str = "15min") -> pd.DataFrame:
    """Energy in MWh per period labelled at its start (SMARD generation)."""
    length = pd.Timedelta(step)
    return _hourly(_cleaned(energy_mwh, length), length)


def _cleaned(series: pd.Series, step: pd.Timedelta) -> pd.Series:
    """Sorted float copy on the `step` grid, without the stamps that appear more than once."""
    index = series.index
    if not isinstance(index, pd.DatetimeIndex) or str(index.tz) != "UTC":
        raise ValueError("expected a DatetimeIndex in UTC")
    if HOUR % step:
        raise ValueError(f"{step} does not divide an hour")
    on_grid = index == index.floor(step)
    if series[~on_grid].notna().any():
        raise ValueError(f"values off the {step} grid")
    return series[on_grid & ~index.duplicated(keep=False)].sort_index().astype(float)


def _hourly(energy: pd.Series, step: pd.Timedelta) -> pd.DataFrame:
    """Sum energies of periods labelled at their start into every hour of their span."""
    hours = pd.date_range(energy.index.min().floor("h"), energy.index.max().floor("h"), freq="h")
    by_hour = energy.resample("h")
    complete = by_hour.count().reindex(hours, fill_value=0) == HOUR // step
    total = by_hour.sum().reindex(hours)
    return pd.DataFrame({"energy_mwh": total.where(complete), "complete": complete})
