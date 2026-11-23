from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from eu_solar_wind_audit.anomalies.timestamp import BIAS_MINUTES, daily_centroids

FloatArray = npt.NDArray[np.float64]

FIRST_DAY = "2022-06-01"
# Middle of each minute of a day.
MINUTES = np.arange(24 * 60) + 0.5
# Share of the energy of a bell outside its hour from 11:00 UTC.
WITHOUT_NOON = 0.837


def utc(stamp: str) -> pd.Timestamp:
    return pd.Timestamp(stamp, tz="UTC")


def bell(offset_minutes: float) -> FloatArray:
    """Hourly energy in MWh of a clear day on 1000 MW, sin² of the time from 06:00 to 18:00 UTC,
    moved later by `offset_minutes`."""
    hours = (MINUTES - offset_minutes) / 60
    power = np.where((hours > 6) & (hours < 18), np.sin(np.pi * (hours - 6) / 12) ** 2, 0.0)
    return 1000.0 * power.reshape(24, 60).mean(axis=1)


def clear_days(offsets: Sequence[float], first_day: str = FIRST_DAY) -> pd.DataFrame:
    """A grid.Hourly frame of consecutive UTC days from `first_day`, the bell of each day moved by
    its offset in minutes."""
    energy = np.concatenate([bell(offset) for offset in offsets])
    time = pd.date_range(utc(first_day), periods=energy.size, freq="h", unit="ns", name="time")
    return pd.DataFrame({"energy_mwh": energy, "complete": True}, index=time)


def reference(hours: pd.Index) -> pd.DataFrame:
    """Price classes with every hour in the reference class."""
    return pd.DataFrame({"regime": "reference"}, index=hours)


def model_of(published: pd.DataFrame) -> pd.DataFrame:
    """Bells without offset on the days of `published`."""
    days = len(published) // 24
    return clear_days([0.0] * days, first_day=str(published.index.min().date()))


def centroids(
    published: pd.DataFrame,
    modelled: pd.DataFrame | None = None,
    classes: pd.DataFrame | None = None,
    bias_minutes: float = 0.0,
) -> pd.DataFrame:
    """Centroids of `published` against bells without offset, with every hour in the reference
    class and no bias, unless given."""
    return daily_centroids(
        published,
        model_of(published) if modelled is None else modelled,
        reference(published.index) if classes is None else classes,
        bias_minutes=bias_minutes,
    )


def on_day(frame: pd.DataFrame, day: str, hours: Sequence[int] = range(24)) -> pd.Index:
    """Index labels of `hours` of the UTC day `day` in `frame`."""
    time = pd.DatetimeIndex(frame.index)
    return time[(time.floor("D") == utc(day)) & time.hour.isin(hours)]


def test_the_residual_is_the_offset_of_the_published_bell() -> None:
    offsets = [0.0, 30.0, -45.0, 60.0, 120.0]

    table = centroids(clear_days(offsets))

    assert table.index.tolist() == [utc(f"2022-06-0{day}") for day in range(1, 6)]
    np.testing.assert_allclose(table["residual_minutes"], offsets, atol=1e-9)
    np.testing.assert_allclose(table["published_centroid_hours"], 12 + np.array(offsets) / 60)
    np.testing.assert_allclose(table["modelled_centroid_hours"], 12.0)
    assert table["valid"].all()
    assert (table[["usable_share", "peak_share"]] == 1.0).all(axis=None)


def test_the_residual_takes_off_the_bias_of_the_model() -> None:
    published = clear_days([60.0])

    assert centroids(published, bias_minutes=60.0)["residual_minutes"].iloc[0] == pytest.approx(0)
    by_default = daily_centroids(published, model_of(published), reference(published.index))
    assert by_default["residual_minutes"].iloc[0] == pytest.approx(60.0 - BIAS_MINUTES)


@pytest.mark.parametrize(
    ("hours_left_out", "usable_share", "valid"),
    # Hours from 06:00 hold 9.1, 19.5 and 33.7 % of the bell.
    [(3, 0.909, True), (4, 0.805, True), (5, 0.663, False)],
)
def test_a_day_is_valid_from_80_percent_of_its_modelled_energy_in_usable_hours(
    hours_left_out: int, usable_share: float, valid: bool
) -> None:
    published = clear_days([0.0, 0.0])
    morning = on_day(published, "2022-06-02", range(6, 6 + hours_left_out))
    published.loc[morning, ["energy_mwh", "complete"]] = [np.nan, False]

    table = centroids(published)

    assert table["usable_share"].tolist() == pytest.approx([1.0, usable_share], abs=1e-3)
    assert table["valid"].tolist() == [True, valid]
    # The hours left out count in neither centroid.
    assert table["residual_minutes"].iloc[1] == pytest.approx(0)


@pytest.mark.parametrize(("scale", "valid"), [(0.49, False), (0.51, True)])
def test_a_day_is_valid_from_half_the_highest_modelled_energy_of_31_days(
    scale: float, valid: bool
) -> None:
    published = clear_days([0.0] * 3)
    modelled = model_of(published)
    for curve in (published, modelled):
        curve.loc[on_day(curve, "2022-06-02"), "energy_mwh"] *= scale

    table = centroids(published, modelled)

    assert table["peak_share"].tolist() == pytest.approx([1.0, scale, 1.0])
    assert table["valid"].tolist() == [True, valid, True]


def test_the_highest_modelled_energy_is_taken_over_the_15_days_on_each_side() -> None:
    # A bright first day, then 40 days at 40 % of it: it reaches the 15th day after it only.
    published = clear_days([0.0] * 41)
    modelled = model_of(published)
    for curve in (published, modelled):
        curve.loc[curve.index >= utc("2022-06-02"), "energy_mwh"] *= 0.4

    table = centroids(published, modelled)

    assert table.loc[utc("2022-06-16"), "peak_share"] == pytest.approx(0.4)
    assert table.loc[utc("2022-06-17"), "peak_share"] == pytest.approx(1.0)
    assert table["valid"].sum() == 41 - 15


def test_a_day_that_the_model_does_not_cover_whole_is_not_valid() -> None:
    published = clear_days([0.0, 0.0])
    modelled = model_of(published).drop(index=utc("2022-06-02 02:00"))

    table = centroids(published, modelled)

    assert table["valid"].tolist() == [True, False]
    assert table[["usable_share", "peak_share"]].iloc[1].isna().all()


def test_a_day_without_published_output_has_no_published_centroid() -> None:
    published = clear_days([0.0, 0.0])
    published.loc[on_day(published, "2022-06-02"), "energy_mwh"] = 0.0

    table = centroids(published)

    assert table["published_centroid_hours"].isna().tolist() == [False, True]
    assert table["valid"].tolist() == [True, False]


def test_negative_price_hours_at_noon_are_left_out_of_both_centroids() -> None:
    # Output is cut at 11:00 UTC every day, when prices are negative.
    published = clear_days([0.0] * 3)
    noon = published.index[pd.DatetimeIndex(published.index).hour == 11]
    published.loc[noon, "energy_mwh"] = 0.0
    classes = reference(published.index)
    classes.loc[noon, "regime"] = "negative"

    with_noon = centroids(published)
    table = centroids(published, classes=classes)

    assert (with_noon["residual_minutes"] > 5).all()
    np.testing.assert_allclose(table["residual_minutes"], 0.0, atol=1e-9)
    assert table["usable_share"].tolist() == pytest.approx([WITHOUT_NOON] * 3, abs=1e-3)
    assert table["valid"].all()


@pytest.mark.parametrize("regime", ["near_zero", "buffer", "missing"])
def test_hours_of_other_price_classes_than_reference_are_left_out(regime: str) -> None:
    published = clear_days([0.0])
    classes = reference(published.index)
    classes.loc[published.index[11], "regime"] = regime

    table = centroids(published, classes=classes)

    assert table["usable_share"].iloc[0] == pytest.approx(WITHOUT_NOON, abs=1e-3)


def test_hours_without_a_price_class_are_left_out() -> None:
    published = clear_days([0.0])
    classes = reference(published.index).drop(index=published.index[11])

    table = centroids(published, classes=classes)

    assert table["usable_share"].iloc[0] == pytest.approx(WITHOUT_NOON, abs=1e-3)


def test_real_time_hours_are_left_out() -> None:
    published = clear_days([0.0, 0.0]).assign(status="consolidated")
    published.loc[on_day(published, "2022-06-02"), "status"] = "realtime"

    table = centroids(published)

    assert table["usable_share"].tolist() == [1.0, 0.0]
    assert table["valid"].tolist() == [True, False]


def test_hours_out_of_order_give_the_same_centroids() -> None:
    published = clear_days([0.0, 60.0, -30.0])
    modelled = model_of(published)

    pd.testing.assert_frame_equal(
        centroids(published.iloc[::-1], modelled.iloc[::-1]), centroids(published, modelled)
    )


def test_price_classes_on_naive_hours_are_refused() -> None:
    published = clear_days([0.0])

    with pytest.raises(ValueError, match="price classes"):
        centroids(published, classes=reference(pd.DatetimeIndex(published.index).tz_localize(None)))
