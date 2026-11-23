import json
from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from eu_solar_wind_audit.anomalies.findings import Findings
from eu_solar_wind_audit.anomalies.outliers import Technology
from eu_solar_wind_audit.anomalies.residuals import (
    CAPACITY_SHARE,
    NORMAL_SCALE,
    SOLAR_FACTOR_FROM,
    ResidualScores,
    residual_outliers,
    residual_scores,
)

FloatArray = npt.NDArray[np.float64]

SERIES = "test.FR.solar"
PROVENANCE = "Producer, test dataset"
# Thresholds of the detector design.
THRESHOLDS: dict[Technology, float] = {"solar": 15.0, "wind": 10.0}
HOUR = pd.Timedelta(hours=1)
FIRST_DAY = "2022-03-01"
CAPACITY_MW = 1000.0
# Standard deviation of the log noise of the published output.
NOISE = 0.05
# 61 days: the middle one is the only day whose 30 days before and after are in the inputs.
DAYS = 61
MIDDLE = 30


def utc(stamp: str) -> pd.Timestamp:
    return pd.Timestamp(stamp, tz="UTC")


def hours_of(days: int) -> pd.DatetimeIndex:
    return pd.date_range(utc(FIRST_DAY), periods=24 * days, freq="h", unit="ns", name="time")


def on_day(day: int, hours: range = range(24)) -> pd.DatetimeIndex:
    """`hours` of the day numbered `day` from FIRST_DAY."""
    start = utc(FIRST_DAY) + pd.Timedelta(days=day)
    return pd.DatetimeIndex([start + pd.Timedelta(hours=hour) for hour in hours]).as_unit("ns")


def bell(days: int) -> FloatArray:
    """Modelled solar output in MWh: sin² of the time at mid-hour from 06:00 to 18:00 UTC, up to
    80 % of CAPACITY_MW at noon, the same every day."""
    hour = np.arange(24) + 0.5
    shape = np.where((hour > 6) & (hour < 18), np.sin(np.pi * (hour - 6) / 12) ** 2, 0.0)
    return np.tile(0.8 * CAPACITY_MW * shape, days)


def breeze(days: int) -> FloatArray:
    """Modelled wind output in MWh: a capacity factor from 0.1 to 0.5 over a period of 3 days."""
    return CAPACITY_MW * (0.3 + 0.2 * np.sin(2 * np.pi * np.arange(24 * days) / 72))


def modelled_frame(energy_mwh: FloatArray) -> pd.DataFrame:
    """A physics.Modelled frame from one modelled energy per hour from FIRST_DAY."""
    return pd.DataFrame(
        {
            "energy_mwh": energy_mwh,
            "complete": True,
            "capacity_mw": CAPACITY_MW,
            "capacity_provisional": False,
            "era5_source": "ERA5",
        },
        index=hours_of(energy_mwh.size // 24),
    )


def noisy(
    modelled: pd.DataFrame, seed: int = 0, log_level: FloatArray | float = 0.0
) -> pd.DataFrame:
    """A grid.Hourly frame: the modelled output times exp(log_level + noise), the noise normal
    with a standard deviation of NOISE."""
    noise = np.random.default_rng(seed).normal(0.0, NOISE, len(modelled))
    energy = modelled["energy_mwh"].to_numpy() * np.exp(log_level + noise)
    return pd.DataFrame({"energy_mwh": energy, "complete": True}, index=modelled.index)


def reference(hours: pd.Index) -> pd.DataFrame:
    """Price classes with every hour in the reference class."""
    return pd.DataFrame({"regime": "reference"}, index=hours)


def scores(
    published: pd.DataFrame,
    modelled: pd.DataFrame,
    technology: Technology,
    classes: pd.DataFrame | None = None,
    offshore: pd.Series | None = None,
) -> pd.DataFrame:
    """Residual scores with every hour in the reference class and, for wind, no offshore output,
    unless `classes` or `offshore` are given."""
    if technology == "wind" and offshore is None:
        offshore = pd.Series(False, index=published.index)
    return residual_scores(
        published,
        modelled,
        reference(published.index) if classes is None else classes,
        technology=technology,
        offshore=offshore,
    )


def by_day(table: pd.DataFrame, column: str, test: Callable[[pd.Series], pd.Series]) -> list[bool]:
    """Whether some hour of each day of `table` passes `test` on `column`."""
    days = pd.DatetimeIndex(table.index).floor("D")
    return test(table[column]).groupby(days).any().tolist()


def test_the_score_follows_its_definition() -> None:
    modelled = modelled_frame(breeze(DAYS))
    published = noisy(modelled)

    table = scores(published, modelled, "wind")

    y = published["energy_mwh"].to_numpy().reshape(DAYS, 24)
    m = modelled["energy_mwh"].to_numpy().reshape(DAYS, 24)
    shift = CAPACITY_SHARE * CAPACITY_MW
    daily_log_ratio = np.log(y.sum(axis=1) / m.sum(axis=1))
    window = range(MIDDLE - 15, MIDDLE + 16)
    level = {day: np.exp(np.median(daily_log_ratio[day - 15 : day + 16])) for day in window}
    residual = {day: np.log((y[day] + shift) / (level[day] * m[day] + shift)) for day in window}
    scale = NORMAL_SCALE * np.median(np.abs(np.concatenate(list(residual.values()))))
    middle = table.loc[on_day(MIDDLE)]
    np.testing.assert_allclose(middle["expected_mwh"], level[MIDDLE] * m[MIDDLE])
    np.testing.assert_allclose(middle["residual"], residual[MIDDLE])
    np.testing.assert_allclose(middle["scale"], scale)
    np.testing.assert_allclose(middle["score"], residual[MIDDLE] / scale)
    np.testing.assert_allclose(middle["published_mwh"], y[MIDDLE])
    assert middle["usable"].all()
    assert not middle["deferred"].any()


def test_scores_wait_for_inputs_that_cover_30_days_on_each_side() -> None:
    modelled = modelled_frame(breeze(80))

    table = scores(noisy(modelled), modelled, "wind")

    days = range(80)
    # Levels need the 15 days on each side, scales the levels of their whole window.
    assert by_day(table, "expected_mwh", pd.Series.notna) == [15 <= day < 65 for day in days]
    assert by_day(table, "residual", pd.Series.notna) == [15 <= day < 65 for day in days]
    assert by_day(table, "score", pd.Series.notna) == [30 <= day < 50 for day in days]
    assert by_day(table, "deferred", lambda deferred: deferred) == [
        not 30 <= day < 50 for day in days
    ]


Inputs = tuple[pd.DataFrame, pd.DataFrame]


def realtime_from(published: pd.DataFrame, modelled: pd.DataFrame, day: int) -> Inputs:
    status = np.where(published.index < on_day(day)[0], "consolidated", "realtime")
    return published.assign(status=status), modelled


def incomplete_from(published: pd.DataFrame, modelled: pd.DataFrame, day: int) -> Inputs:
    later = published.index >= on_day(day)[0]
    energy_mwh = published["energy_mwh"].mask(later)
    return published.assign(energy_mwh=energy_mwh, complete=~later), modelled


def unmodelled_from(published: pd.DataFrame, modelled: pd.DataFrame, day: int) -> Inputs:
    return published, modelled[modelled.index < on_day(day)[0]]


@pytest.mark.parametrize("cut", [realtime_from, incomplete_from, unmodelled_from])
def test_the_inputs_end_with_the_last_hour_that_has_both_final_values(
    cut: Callable[[pd.DataFrame, pd.DataFrame, int], Inputs],
) -> None:
    modelled = modelled_frame(breeze(80))
    # The inputs then cover 70 days.
    published, modelled = cut(noisy(modelled), modelled, 70)

    table = scores(published, modelled, "wind")

    assert by_day(table, "score", pd.Series.notna) == [30 <= day < 40 for day in range(80)]
    assert not table.loc[table.index >= on_day(70)[0], "usable"].any()


def test_the_inputs_start_with_the_first_hour_that_has_both_final_values() -> None:
    modelled = modelled_frame(breeze(80))
    published = noisy(modelled)
    # The first hour of the inputs is 01:00, so the first day is not covered.
    published.loc[on_day(0, range(1)), ["energy_mwh", "complete"]] = [np.nan, False]

    table = scores(published, modelled, "wind")

    assert by_day(table, "score", pd.Series.notna) == [31 <= day < 50 for day in range(80)]


def test_the_local_level_absorbs_a_slow_drift() -> None:
    # The published output loses 1 % a day against the model, 55 % over the 80 days: a single
    # level would leave residuals of up to 0.4, eight times the noise.
    modelled = modelled_frame(bell(80))
    log_level = -0.01 * np.arange(80 * 24) / 24

    table = scores(noisy(modelled, log_level=log_level), modelled, "solar")

    scored = table.dropna(subset=["score"])
    assert len(scored) == 20 * 10
    assert scored["score"].abs().max() < 4
    # The level of a day is the drift at its noon.
    day = (pd.DatetimeIndex(scored.index) - utc(FIRST_DAY)) // pd.Timedelta(days=1)
    level = scored["expected_mwh"] / modelled.loc[scored.index, "energy_mwh"]
    np.testing.assert_allclose(np.log(level), -0.01 * (day + 0.5), atol=0.02)


@pytest.mark.parametrize("technology", ["solar", "wind"])
def test_solar_hours_count_from_a_modelled_capacity_factor_of_0_05(technology: Technology) -> None:
    modelled = modelled_frame(bell(DAYS))

    table = scores(noisy(modelled), modelled, technology)

    factor = modelled["energy_mwh"] / CAPACITY_MW
    expected = (factor >= SOLAR_FACTOR_FROM) | (technology == "wind")
    assert table["usable"].tolist() == expected.tolist()
    # The 10 hours from 07:00 reach the floor.
    assert table.loc[on_day(MIDDLE), "usable"].sum() == (10 if technology == "solar" else 24)


@pytest.mark.parametrize("regime", ["negative", "near_zero", "buffer", "missing", None])
def test_hours_without_the_reference_price_class_are_left_out(regime: str | None) -> None:
    modelled = modelled_frame(bell(DAYS))
    published = noisy(modelled)
    classes = reference(published.index)
    noon = on_day(MIDDLE, range(12, 13))
    if regime is None:
        classes = classes.drop(index=noon)
    else:
        classes.loc[noon, "regime"] = regime

    table = scores(published, modelled, "solar", classes)

    assert not table.loc[noon, "usable"].any()
    assert table.loc[noon, ["residual", "score"]].isna().all(axis=None)
    assert table.loc[on_day(MIDDLE), "score"].notna().sum() == 9


def test_wind_hours_with_offshore_output_are_left_out() -> None:
    modelled = modelled_frame(breeze(DAYS))
    published = noisy(modelled)
    # Offshore output from noon of the middle day on.
    offshore = pd.Series(published.index >= on_day(MIDDLE)[12], index=published.index)

    table = scores(published, modelled, "wind", offshore=offshore)

    assert table["usable"].tolist() == (~offshore).tolist()
    assert table.loc[on_day(MIDDLE), "score"].notna().tolist() == [hour < 12 for hour in range(24)]


def test_a_wind_series_needs_its_hours_with_offshore_output() -> None:
    modelled = modelled_frame(breeze(DAYS))
    published = noisy(modelled)

    with pytest.raises(ValueError, match="offshore output"):
        residual_scores(published, modelled, reference(published.index), technology="wind")


@pytest.mark.parametrize("naive", [False, True])
def test_an_offshore_mask_needs_every_hour_of_the_series(naive: bool) -> None:
    modelled = modelled_frame(breeze(DAYS))
    published = noisy(modelled)
    offshore = pd.Series(False, index=published.index)
    offshore = offshore.tz_localize(None) if naive else offshore.iloc[1:]

    with pytest.raises(ValueError, match="offshore flag"):
        scores(published, modelled, "wind", offshore=offshore)


def test_hours_without_both_final_values_or_a_capacity_are_left_out() -> None:
    modelled = modelled_frame(breeze(DAYS))
    published = noisy(modelled).assign(status="consolidated")
    realtime, incomplete, unmodelled, without_capacity = on_day(MIDDLE, range(10, 14))
    published.loc[realtime, "status"] = "realtime"
    published.loc[incomplete, ["energy_mwh", "complete"]] = [np.nan, False]
    modelled.loc[unmodelled, ["energy_mwh", "complete"]] = [np.nan, False]
    modelled.loc[without_capacity, ["energy_mwh", "capacity_mw"]] = 0.0

    table = scores(published, modelled, "wind")

    left_out = on_day(MIDDLE, range(10, 14))
    assert not table.loc[left_out, "usable"].any()
    assert table.loc[left_out, "score"].isna().all()
    assert table["usable"].sum() == DAYS * 24 - 4


@pytest.mark.parametrize(
    ("days", "hours", "level", "scored"),
    [
        (range(27, 34), range(24), True, True),
        (range(27, 33), range(24), False, False),
        (range(27, 34), range(23), True, False),
    ],
    ids=["a-week", "six-days", "a-week-of-23-hours"],
)
def test_a_score_needs_a_week_of_usable_hours_in_its_window(
    days: range, hours: range, level: bool, scored: bool
) -> None:
    # Only the hours of `days` have the reference price class.
    modelled = modelled_frame(breeze(DAYS))
    published = noisy(modelled)
    classes = reference(published.index).assign(regime="near_zero")
    for day in days:
        classes.loc[on_day(day, hours), "regime"] = "reference"

    table = scores(published, modelled, "wind", classes)

    middle = table.loc[on_day(MIDDLE)]
    assert middle["expected_mwh"].notna().all() == level
    assert middle["score"].notna().any() == scored


def test_a_published_value_below_minus_one_percent_of_the_capacity_has_no_residual() -> None:
    modelled = modelled_frame(breeze(DAYS))
    published = noisy(modelled)
    hours = on_day(MIDDLE, range(3, 5))
    # Minus 1 % of the capacity, then no output.
    published.loc[hours, "energy_mwh"] = [-CAPACITY_SHARE * CAPACITY_MW, 0.0]

    table = scores(published, modelled, "wind").loc[hours]

    assert table["usable"].all()
    assert table["residual"].isna().tolist() == [True, False]
    assert (table["score"] < -10).tolist() == [False, True]


def test_hours_out_of_order_give_the_same_scores() -> None:
    modelled = modelled_frame(bell(DAYS))
    published = noisy(modelled)

    pd.testing.assert_frame_equal(
        scores(published.iloc[::-1], modelled.iloc[::-1], "solar"),
        scores(published, modelled, "solar"),
    )


def test_price_classes_on_naive_hours_are_refused() -> None:
    modelled = modelled_frame(bell(DAYS))
    published = noisy(modelled)
    naive = reference(pd.DatetimeIndex(published.index).tz_localize(None))

    with pytest.raises(ValueError, match="price classes"):
        scores(published, modelled, "solar", naive)


def test_a_series_without_hours_gives_an_empty_valid_table() -> None:
    modelled = modelled_frame(bell(DAYS))
    published = noisy(modelled).iloc[:0]

    table = scores(published, modelled, "solar")

    assert table.empty
    assert list(table.columns) == list(ResidualScores.to_schema().columns)


def outliers(table: pd.DataFrame, threshold: float) -> pd.DataFrame:
    return residual_outliers(table, series=SERIES, provenance=PROVENANCE, threshold=threshold)


def zero_output(published: pd.DataFrame, hours: pd.DatetimeIndex) -> pd.DataFrame:
    """`published` without output at `hours`."""
    energy_mwh = published["energy_mwh"].mask(published.index.isin(hours), 0.0)
    return published.assign(energy_mwh=energy_mwh)


def spans(found: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    return list(found[["start", "end"]].itertuples(index=False, name=None))


def test_a_three_hour_zero_run_in_a_noisy_bell_is_flagged() -> None:
    modelled = modelled_frame(bell(DAYS))
    run = on_day(MIDDLE, range(11, 14))
    table = scores(zero_output(noisy(modelled), run), modelled, "solar")

    found = outliers(table, THRESHOLDS["solar"])

    assert spans(found) == [(run[0], run[-1] + HOUR)]
    assert (found["score"] < -2 * THRESHOLDS["solar"]).all()
    peak = run[int(table.loc[run, "score"].abs().argmax())]
    values = table.loc[peak, ["published_mwh", "expected_mwh", "residual", "scale", "score"]]
    assert found["score"].tolist() == [values["score"]]
    assert json.loads(found["evidence"].iloc[0]) == {
        "threshold": THRESHOLDS["solar"],
        "flagged_hours": 3,
        "peak_hour": peak.isoformat(),
        **values.to_dict(),
    }
    assert found[["series", "provenance", "kind"]].values.tolist() == [
        [SERIES, PROVENANCE, "residual_outlier"]
    ]
    assert not found["provisional"].any()


@pytest.mark.parametrize(("technology", "shape"), [("solar", bell), ("wind", breeze)])
def test_noise_alone_gives_no_finding_at_the_design_thresholds(
    technology: Technology, shape: Callable[[int], FloatArray]
) -> None:
    modelled = modelled_frame(shape(80))
    table = scores(noisy(modelled, seed=1), modelled, technology)

    found = outliers(table, THRESHOLDS[technology])

    assert found.empty
    assert list(found.columns) == list(Findings.to_schema().columns)
    # The largest score of the noise is flagged by a threshold of its own size only.
    largest = float(table["score"].abs().max())
    assert 2 < largest < 5
    assert len(outliers(table, largest)) == 1


@pytest.mark.parametrize(("hours_between", "joined"), [(23, True), (24, False)])
def test_runs_of_flagged_hours_less_than_24_hours_apart_are_one_finding(
    hours_between: int, joined: bool
) -> None:
    modelled = modelled_frame(breeze(80))
    first = on_day(35)[0]
    second = first + (hours_between + 1) * HOUR
    published = zero_output(noisy(modelled), pd.DatetimeIndex([first, second]))

    found = outliers(scores(published, modelled, "wind"), THRESHOLDS["wind"])

    if joined:
        assert spans(found) == [(first, second + HOUR)]
        assert json.loads(found["evidence"].iloc[0])["flagged_hours"] == 2
    else:
        assert spans(found) == [(first, first + HOUR), (second, second + HOUR)]


def test_zero_output_in_hours_with_offshore_output_gives_no_finding() -> None:
    modelled = modelled_frame(breeze(80))
    run = on_day(35, range(10, 13))
    published = zero_output(noisy(modelled), run)
    offshore = pd.Series(published.index.isin(run), index=published.index)

    assert len(outliers(scores(published, modelled, "wind"), THRESHOLDS["wind"])) == 1
    found = outliers(scores(published, modelled, "wind", offshore=offshore), THRESHOLDS["wind"])
    assert found.empty


def test_zero_output_in_hours_of_negative_prices_gives_no_finding() -> None:
    # Output cut while prices are negative is modulation, which the reference class leaves out.
    modelled = modelled_frame(bell(80))
    run = on_day(35, range(11, 14))
    published = zero_output(noisy(modelled), run)
    classes = reference(published.index)
    classes.loc[run, "regime"] = "negative"

    assert len(outliers(scores(published, modelled, "solar"), THRESHOLDS["solar"])) == 1
    assert outliers(scores(published, modelled, "solar", classes), THRESHOLDS["solar"]).empty


def test_a_zero_run_near_the_end_of_the_inputs_waits_for_later_inputs() -> None:
    modelled = modelled_frame(bell(100))
    published = zero_output(noisy(modelled), on_day(60, range(11, 14)))
    # Day 60 is 20 days from the end of the first 80 days.
    first_80_days = published.index < on_day(80)[0]

    earlier = scores(published[first_80_days], modelled[first_80_days], "solar")
    later = scores(published, modelled, "solar")

    assert earlier.loc[on_day(60), "deferred"].all()
    assert outliers(earlier, THRESHOLDS["solar"]).empty
    assert spans(outliers(later, THRESHOLDS["solar"])) == [(on_day(60)[11], on_day(60)[14])]


def test_scores_out_of_order_give_the_same_findings() -> None:
    modelled = modelled_frame(bell(DAYS))
    run = on_day(MIDDLE, range(11, 14))
    table = scores(zero_output(noisy(modelled), run), modelled, "solar")

    pd.testing.assert_frame_equal(
        outliers(table.iloc[::-1], THRESHOLDS["solar"]), outliers(table, THRESHOLDS["solar"])
    )
