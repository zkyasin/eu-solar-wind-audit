from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from eu_solar_wind_audit import figures

BERLIN = "Europe/Berlin"


def test_daily_gap_sums_each_local_day_and_leaves_out_an_incomplete_one() -> None:
    hours = pd.date_range("2025-01-01", periods=48, freq="h", tz=BERLIN).tz_convert("UTC")
    total = pd.Series(10.0, index=hours)
    parts = pd.DataFrame({"north": 4.0, "south": 5.0}, index=hours)
    parts.iloc[30, 1] = np.nan

    gap = figures.daily_gap_gwh(total, parts, BERLIN)

    assert list(gap.index) == list(pd.date_range("2025-01-01", periods=2, freq="D", tz=BERLIN))
    assert gap.iloc[0] == pytest.approx(0.024)
    assert np.isnan(gap.iloc[1])


def write_each(directory: Path) -> list[Path]:
    hours = pd.date_range("2025-06-02", periods=48, freq="h", tz="UTC", name="time")
    hourly = pd.DataFrame(
        {"published_mwh": np.linspace(0, 5000, 48), "potential_mwh": np.linspace(0, 6000, 48)},
        index=hours,
    )
    negative = pd.Series(hours.hour.isin([11, 12, 13]), index=hours)
    years = pd.DataFrame(
        {
            "estimate_gwh": [20.0, 700.0],
            "low_gwh": [10.0, 600.0],
            "high_gwh": [30.0, 800.0],
            "rte_gwh": [25.0, 640.0],
        },
        index=[2020, 2024],
    )
    days = pd.date_range("2025-01-01", periods=60, freq="D", tz=BERLIN)
    gap = pd.Series(np.where(np.arange(60) < 30, 0.0, 5.0), index=days)
    flagged = pd.DataFrame({"start": [days[30]], "end": [days[40]]}).apply(
        lambda column: column.dt.tz_convert("UTC")
    )
    return [
        *figures.write_week(directory, {"solar": hourly, "wind_onshore": hourly}, negative, "F."),
        *figures.write_modulation(directory, {"solar": years, "wind_onshore": years}, "F."),
        *figures.write_sum_of_parts(directory, gap, flagged, BERLIN, "Title", "F."),
    ]


def test_each_figure_has_a_light_and_a_dark_version_drawn_again_byte_for_byte(
    tmp_path: Path,
) -> None:
    first = write_each(tmp_path / "first")
    second = write_each(tmp_path / "second")

    assert [path.name for path in first] == [
        f"{name}-{theme}.svg"
        for name in ("week", "modulation", "sum-of-parts")
        for theme in ("light", "dark")
    ]
    assert [path.read_bytes() for path in first] == [path.read_bytes() for path in second]
    assert first[0].read_bytes() != first[1].read_bytes()
