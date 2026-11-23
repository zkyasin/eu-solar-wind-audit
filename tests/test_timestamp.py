import json
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from eu_solar_wind_audit.anomalies.findings import Findings
from eu_solar_wind_audit.anomalies.timestamp import (
    BIAS_MINUTES,
    clock_change_jumps,
    daily_centroids,
    shifts,
)

FloatArray = npt.NDArray[np.float64]

SERIES = "test.FR.solar"
PROVENANCE = "Producer, test dataset"
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


def claims(
    published: pd.DataFrame,
    modelled: pd.DataFrame | None = None,
    classes: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Shift findings of `published`, with the defaults of `centroids`."""
    table = centroids(published, modelled, classes)
    return shifts(table, series=SERIES, provenance=PROVENANCE)


def episode(offsets: Sequence[float]) -> list[float]:
    """`offsets` from 11 June 2022, with ten days without offset before and after them."""
    return [0.0] * 10 + list(offsets) + [0.0] * 10


def spans(found: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp, float]]:
    """Start, end and score of each finding, the score rounded to 1e-9."""
    rows = found[["start", "end", "score"]].itertuples(index=False, name=None)
    return [(start, end, round(score, 9)) for start, end, score in rows]


def evidence(found: pd.DataFrame) -> list[dict[str, object]]:
    return [json.loads(text) for text in found["evidence"]]


@pytest.mark.parametrize("hours", [1, -1, 2, -2])
def test_a_bell_shifted_by_whole_hours_is_claimed_with_its_size(hours: int) -> None:
    found = claims(clear_days(episode([60.0 * hours] * 14)))

    assert spans(found) == [(utc("2022-06-11"), utc("2022-06-25"), hours)]
    assert evidence(found) == [
        {
            "shift_hours": hours,
            "median_residual_minutes": pytest.approx(60.0 * hours),
            "deciding_days": ["2022-06-11", "2022-06-12"],
            "deciding_mean_minutes": pytest.approx(60.0 * hours),
            "days_off": 14,
        }
    ]
    assert found[["series", "provenance", "kind"]].values.tolist() == [
        [SERIES, PROVENANCE, "timestamp_shift"]
    ]
    assert not found["provisional"].any()


@pytest.mark.parametrize(("offset", "hours"), [(84.0, 1), (96.0, 2), (-84.0, -1), (-96.0, -2)])
def test_the_size_is_the_median_offset_rounded_to_whole_hours(offset: float, hours: int) -> None:
    found = claims(clear_days(episode([offset] * 3)))

    assert evidence(found)[0]["shift_hours"] == hours


def test_a_series_off_from_its_first_day_is_claimed() -> None:
    found = claims(clear_days([60.0] * 20))

    assert spans(found) == [(utc(FIRST_DAY), utc("2022-06-21"), 1.0)]


@pytest.mark.parametrize("sign", [1, -1])
def test_a_half_hour_shift_is_never_claimed_as_an_hour(sign: int) -> None:
    # Two days average 47 minutes, but the median of the first seven is 33 minutes.
    offsets = [sign * offset for offset in [31, 33, 35, 32, 32, 50, 44, 31, 33, 36, 31, 31, 34]]
    published = clear_days(episode(offsets))

    found = claims(published)

    assert found.empty
    # The offset stays visible in the centroid table.
    residual = centroids(published)["residual_minutes"]
    np.testing.assert_allclose(residual.iloc[10:23], offsets, atol=1e-9)


@pytest.mark.parametrize(
    ("offsets", "claimed"),
    [([40.0, 52.0], True), ([40.0, 48.0], False), ([-40.0, -52.0], True), ([-40.0, -48.0], False)],
)
def test_the_two_deciding_days_must_be_off_by_45_minutes_on_average(
    offsets: list[float], claimed: bool
) -> None:
    assert len(claims(clear_days(episode(offsets)))) == claimed


@pytest.mark.parametrize("offsets", [[120.0], [-120.0], [120.0, 0.0, 120.0]])
def test_a_one_day_excursion_is_not_claimed(offsets: list[float]) -> None:
    assert claims(clear_days(episode(offsets))).empty


@pytest.mark.parametrize(
    ("gap", "expected"),
    [
        (3, [(utc("2022-06-06"), utc("2022-06-19"), 1.0)]),
        (
            4,
            [
                (utc("2022-06-06"), utc("2022-06-11"), 1.0),
                (utc("2022-06-15"), utc("2022-06-20"), 1.0),
            ],
        ),
    ],
)
def test_same_sign_runs_up_to_three_valid_days_apart_are_one_finding(
    gap: int, expected: list[tuple[pd.Timestamp, pd.Timestamp, float]]
) -> None:
    published = clear_days([0.0] * 5 + [60.0] * 5 + [0.0] * gap + [60.0] * 5 + [0.0] * 5)

    assert spans(claims(published)) == expected


def test_days_that_are_not_valid_do_not_separate_runs() -> None:
    # Five days between the runs, two of them too dark to be valid.
    published = clear_days([0.0] * 5 + [60.0] * 5 + [0.0] * 5 + [60.0] * 5 + [0.0] * 5)
    modelled = model_of(published)
    for curve in (published, modelled):
        for day in ("2022-06-12", "2022-06-14"):
            curve.loc[on_day(curve, day), "energy_mwh"] *= 0.3

    found = claims(published, modelled)

    assert spans(found) == [(utc("2022-06-06"), utc("2022-06-21"), 1.0)]
    assert evidence(found)[0]["days_off"] == 10


def test_runs_of_opposite_signs_are_separate_findings() -> None:
    published = clear_days([0.0] * 5 + [60.0] * 5 + [0.0] * 2 + [-60.0] * 5 + [0.0] * 5)

    assert spans(claims(published)) == [
        (utc("2022-06-06"), utc("2022-06-11"), 1.0),
        (utc("2022-06-13"), utc("2022-06-18"), -1.0),
    ]


def test_real_time_hours_are_left_out_of_the_claims() -> None:
    consolidated = clear_days(episode([60.0] * 14)).assign(status="consolidated")
    # Real-time rows from the first day of the shift.
    status = np.where(consolidated.index < utc("2022-06-11"), "consolidated", "realtime")

    assert len(claims(consolidated)) == 1
    assert claims(consolidated.assign(status=status)).empty


def test_output_cut_in_negative_price_hours_is_not_claimed_as_a_shift() -> None:
    # Output is cut from 09:00 to 12:00 UTC on 14 days of negative prices, which moves the
    # published centroid by 56 minutes.
    published = clear_days(episode([0.0] * 14))
    cut = published.index[
        (published.index >= utc("2022-06-11"))
        & (published.index < utc("2022-06-25"))
        & pd.DatetimeIndex(published.index).hour.isin([9, 10, 11])
    ]
    published.loc[cut, "energy_mwh"] = 0.0
    classes = reference(published.index)
    classes.loc[cut, "regime"] = "negative"

    assert len(claims(published)) == 1
    assert claims(published, classes=classes).empty


def test_centroids_out_of_order_give_the_same_claims() -> None:
    table = centroids(clear_days(episode([60.0] * 14)))

    pd.testing.assert_frame_equal(
        shifts(table.iloc[::-1], series=SERIES, provenance=PROVENANCE),
        shifts(table, series=SERIES, provenance=PROVENANCE),
    )


def test_no_claim_gives_an_empty_valid_frame() -> None:
    found = claims(clear_days([0.0] * 10))

    assert found.empty
    assert list(found.columns) == list(Findings.to_schema().columns)


def jumps(
    published: pd.DataFrame,
    modelled: pd.DataFrame | None = None,
    time_zone: str = "Europe/Paris",
) -> pd.DataFrame:
    """Clock-change findings of `published`, with the defaults of `centroids`."""
    table = centroids(published, modelled)
    return clock_change_jumps(table, time_zone=time_zone, series=SERIES, provenance=PROVENANCE)


def paris_time_as_utc(first_day: str, days: int) -> list[float]:
    """Offsets in minutes of the days of a source that labels Paris time as UTC: the UTC offset of
    Paris at noon of each UTC day from `first_day`."""
    noon = pd.date_range(utc(first_day) + pd.Timedelta(hours=12), periods=days, freq="D")
    utc_offset = noon.tz_convert("Europe/Paris").tz_localize(None) - noon.tz_localize(None)
    return [float(minutes) for minutes in utc_offset / pd.Timedelta(minutes=1)]


def test_local_time_labelled_as_utc_jumps_at_both_clock_changes() -> None:
    # Clocks went forward on 27 March 2022 and back on 30 October 2022, at 01:00 UTC.
    found = jumps(clear_days(paris_time_as_utc("2022-01-01", 365), first_day="2022-01-01"))

    assert spans(found) == [
        (utc("2022-03-13"), utc("2022-04-11"), 1.0),
        (utc("2022-10-16"), utc("2022-11-14"), -1.0),
    ]
    assert evidence(found)[0] == {
        "time_zone": "Europe/Paris",
        "clock_change": "2022-03-27T01:00:00+00:00",
        "jump_minutes": pytest.approx(60.0),
        "median_before_minutes": pytest.approx(60.0),
        "median_after_minutes": pytest.approx(120.0),
        "valid_days_before": 14,
        "valid_days_after": 14,
    }
    assert found["kind"].unique().tolist() == ["clock_change_jump"]
    assert not found["provisional"].any()


@pytest.mark.parametrize(
    ("offsets", "time_zone"),
    [
        (paris_time_as_utc("2022-01-01", 365), "UTC"),
        ([0.0] * 365, "Europe/Paris"),
    ],
    ids=["no-clock-change", "utc-labels"],
)
def test_no_jump_without_a_change_of_offset(offsets: list[float], time_zone: str) -> None:
    published = clear_days(offsets, first_day="2022-01-01")

    assert jumps(published, time_zone=time_zone).empty


def around_spring_change(after_minutes: float) -> pd.DataFrame:
    """The 14 days before 27 March 2022 without offset, then that day and the 14 days after it
    moved by `after_minutes`."""
    return clear_days([0.0] * 14 + [after_minutes] * 15, first_day="2022-03-13")


@pytest.mark.parametrize(("after_minutes", "found"), [(25.0, False), (35.0, True), (-35.0, True)])
def test_a_jump_counts_from_30_minutes(after_minutes: float, found: bool) -> None:
    assert len(jumps(around_spring_change(after_minutes))) == found


@pytest.mark.parametrize(("valid_after", "found"), [(2, False), (3, True)])
def test_a_jump_needs_three_valid_days_on_each_side(valid_after: int, found: bool) -> None:
    published = around_spring_change(60.0)
    modelled = model_of(published)
    # Days after the change that are too dark to be valid.
    dark = (published.index >= utc("2022-03-28") + pd.Timedelta(days=valid_after)) & (
        published.index < utc("2022-04-11")
    )
    for curve in (published, modelled):
        curve.loc[dark, "energy_mwh"] *= 0.3

    assert len(jumps(published, modelled)) == found


def test_a_table_without_valid_days_gives_an_empty_valid_frame() -> None:
    published = clear_days([0.0] * 3).assign(status="realtime")

    found = jumps(published)

    assert found.empty
    assert list(found.columns) == list(Findings.to_schema().columns)
