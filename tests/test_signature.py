import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from eu_solar_wind_audit import signature
from eu_solar_wind_audit.evaluation import SEASON_OF_MONTH

# Raw model output of each sky class at the midday hours, for a capacity of 2,000 MW.
POTENTIAL_MWH = {"clear": 1000.0, "mixed": 600.0, "overcast": 300.0}
CAPACITY_MW = 2000.0
# Dawn hour of every day, with a modelled capacity factor of 0.025.
DAWN_MWH = 50.0


def predictions(self_consumed: dict[str, float], seed: int = 0) -> tuple[pd.Series, pd.DataFrame]:
    """Twelve days of June 2024 from Monday 3, clear, mixed and overcast in turn, at 4 UTC and from
    10 to 12 UTC, and a published output that leaves out the share `self_consumed` of the raw model
    of each class, times a noise of 1 %: the published series and evaluation.Predictions."""
    rng = np.random.default_rng(seed)
    rows = []
    for day, sky in enumerate(["clear", "mixed", "overcast"] * 4, start=3):
        for hour in (4, 10, 11, 12):
            raw = DAWN_MWH if hour == 4 else POTENTIAL_MWH[sky]
            noise = 1 + 0.01 * rng.standard_normal()
            rows.append(
                {
                    "time": pd.Timestamp(f"2024-06-{day:02d} {hour}:00", tz="UTC"),
                    "published_mwh": raw * (1 - self_consumed[sky]) * noise,
                    "raw_mwh": raw,
                    "sky_class": sky,
                }
            )
    table = pd.DataFrame(rows).set_index("time")
    time = pd.DatetimeIndex(table.index).as_unit("ns")
    frame = table.set_axis(time).assign(
        held_out_year=time.year.astype("int64"),
        capacity_mw=CAPACITY_MW,
        eligible=True,
        season=time.month.map(SEASON_OF_MONTH),
    )
    return frame["published_mwh"], frame


WHOLE_OUTPUT = {"clear": 0.0, "mixed": 0.0, "overcast": 0.0}
FEED_IN = {"clear": 0.05, "mixed": 0.15, "overcast": 0.3}


def test_a_series_of_the_whole_output_shows_no_contrast() -> None:
    published, frame = predictions(WHOLE_OUTPUT)

    found = signature.contrast(signature.hours_table(published, frame, "DE"))

    assert found.low < 0 < found.high


def test_a_series_of_feed_in_only_rises_more_on_clear_days() -> None:
    published, frame = predictions(FEED_IN)

    found = signature.contrast(signature.hours_table(published, frame, "DE"))

    assert found.estimate == pytest.approx(np.log(0.95 / 0.7), abs=0.01)
    assert found.low > 0
    assert (found.hours, found.days) == (48, 12)


def test_a_level_of_one_day_type_does_not_enter_the_contrast() -> None:
    published, frame = predictions(WHOLE_OUTPUT)
    # Saturday 8 June is overcast: without its own cells, its lower level would favour clear days.
    saturday = pd.DatetimeIndex(published.index).tz_convert("Europe/Paris").day_of_week == 5

    found = signature.contrast(
        signature.hours_table(published.where(~saturday, published / 2), frame, "DE")
    )

    assert found.estimate == pytest.approx(0.0, abs=0.02)


def test_the_interval_is_the_90_percent_normal_interval() -> None:
    published, frame = predictions(FEED_IN)

    found = signature.contrast(signature.hours_table(published, frame, "DE"))

    half_width = norm.ppf(0.95) * found.standard_error
    assert found.high - found.estimate == pytest.approx(half_width)
    assert found.estimate - found.low == pytest.approx(half_width)


def test_the_hours_are_eligible_with_a_sky_class_and_positive_output() -> None:
    published, frame = predictions(WHOLE_OUTPUT)
    first, second, third = frame.index[1:4]
    frame.loc[first, "eligible"] = False
    frame.loc[second, "sky_class"] = np.nan
    published.loc[third] = 0.0

    table = signature.hours_table(published, frame, "DE")

    assert len(table) == 45
    assert not table.index.isin([first, second, third]).any()


def test_a_cell_holds_one_local_hour_of_one_month_year_and_day_type() -> None:
    published, frame = predictions(WHOLE_OUTPUT)

    table = signature.hours_table(published, frame, "DE")

    # 12 UTC is 14:00 in Paris summer time; 8 June 2024 is a Saturday.
    saturday = table.loc[pd.Timestamp("2024-06-08 12:00", tz="UTC")]
    assert saturday["cell"] == "14-6-2024-saturday"
    assert saturday["season"] == "JJA"
    assert saturday["day"] == pd.Timestamp("2024-06-08")
    assert saturday["log_ratio"] == pytest.approx(0.0, abs=0.05)
    assert table["capacity_factor"].min() == DAWN_MWH / CAPACITY_MW


def test_the_published_series_must_be_on_the_hours_of_the_predictions() -> None:
    published, frame = predictions(WHOLE_OUTPUT)

    with pytest.raises(ValueError, match="hours of the predictions"):
        signature.hours_table(published.iloc[1:], frame, "DE")


def test_contrasts_cover_all_hours_and_each_season_with_and_without_dawn_and_dusk() -> None:
    published, frame = predictions(FEED_IN)

    rows = signature.contrasts(signature.hours_table(published, frame, "DE"))

    assert rows["min_capacity_factor"].tolist() == [0.0, 0.0, 0.05, 0.05]
    assert rows["season"].tolist() == ["all", "JJA", "all", "JJA"]
    assert rows["hours"].tolist() == [48, 48, 36, 36]
    assert set(rows["sign"]) == {"positive"}


def control_rows(eco2mix: str, smard: str) -> pd.DataFrame:
    """The sign of each series on all its hours, and a seasonal row that never counts."""
    return pd.DataFrame(
        {
            "series": ["eco2mix", "smard", "eco2mix"],
            "season": ["all", "all", "JJA"],
            "min_capacity_factor": [0.0, 0.0, 0.0],
            "sign": [eco2mix, smard, "positive"],
        }
    )


@pytest.mark.parametrize(
    ("eco2mix", "smard", "expected"),
    [
        ("none", "positive", (True, True, True)),
        ("negative", "positive", (True, True, True)),
        ("positive", "positive", (False, True, False)),
        ("none", "none", (True, False, False)),
        ("none", "negative", (True, False, False)),
    ],
)
def test_the_evidence_is_used_only_when_both_controls_pass(
    eco2mix: str, smard: str, expected: tuple[bool, bool, bool]
) -> None:
    found = signature.controls(control_rows(eco2mix, smard))

    assert (found["negative_control"], found["positive_control"], found["used"]) == expected
