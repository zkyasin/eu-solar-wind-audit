import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from eu_solar_wind_audit import cli, signature
from eu_solar_wind_audit.evaluation import SEASON_OF_MONTH
from eu_solar_wind_audit.sources import energy_charts, odre, smard

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


def test_the_command_writes_the_contrasts_and_the_controls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    french, german = predictions(WHOLE_OUTPUT)[1], predictions(FEED_IN)[1]
    for name, frame in {"calibration-fr": french, "calibration-de": german}.items():
        (tmp_path / name).mkdir()
        frame.to_parquet(tmp_path / name / "solar-predictions.parquet")
    # Energy-Charts total power: the feed-in times a factor of the month.
    total = pd.DataFrame({"energy_mwh": german["published_mwh"] * 1.25, "complete": True})

    def fetch_hourly_power(start: pd.Timestamp, end: pd.Timestamp) -> dict[str, pd.DataFrame]:
        assert (start, end) == (german.index[0], german.index[-1] + pd.Timedelta(hours=1))
        return {"total.solar": total}

    monkeypatch.setattr(energy_charts, "fetch_hourly_power", fetch_hourly_power)
    output = tmp_path / "signature"

    cli.main(
        [
            "feed-in-signature",
            str(output),
            "--calibration-fr",
            str(tmp_path / "calibration-fr"),
            "--calibration-de",
            str(tmp_path / "calibration-de"),
        ]
    )

    summary = json.loads((output / "feed-in-signature.json").read_text(encoding="utf-8"))
    assert summary["controls"] == {"negative_control": True, "positive_control": True, "used": True}
    assert set(summary["predictions"]) == {"FR", "DE"}
    rows = pd.read_parquet(output / "feed-in-signature.parquet")
    overall = rows[(rows["season"] == "all") & (rows["min_capacity_factor"] == 0)]
    estimates = overall.set_index("series")["estimate"]
    assert estimates["energy_charts_total"] == pytest.approx(estimates["smard"])
    assert len(summary["contrasts"]) == 6
    # The hours each contrast is fitted on, for another implementation to refit them.
    hours = pd.read_parquet(output / "feed-in-signature-hours.parquet")
    assert (
        hours.groupby("series").size().to_dict() == overall.set_index("series")["hours"].to_dict()
    )


# Modelled output from 6 to 16 UTC as a share of the peak of the day, and the peak of each class.
DAYLIGHT = (0.1, 0.3, 0.5, 0.7, 0.85, 0.9, 0.85, 0.7, 0.5, 0.3, 0.1)
PEAK_MWH = {"clear": 1500.0, "mixed": 900.0, "overcast": 400.0}


def load_predictions(
    self_consumed: float, overcast_level_mwh: float = 0.0, seed: int = 0
) -> tuple[pd.Series, pd.DataFrame]:
    """Three weeks of June 2024 from Monday 3, clear, mixed and overcast in turn, at 2 UTC, without
    modelled output, and from 6 to 16 UTC. The load is a usual profile that peaks at midday like
    the modelled output, plus a random level of the day and `overcast_level_mwh` on overcast days,
    less the share `self_consumed` of the raw model, with a noise of 5 MWh: the load and
    evaluation.Predictions."""
    rng = np.random.default_rng(seed)
    rows = []
    for day, sky in enumerate(["clear", "mixed", "overcast"] * 7, start=3):
        level = rng.normal(0, 1000) + (overcast_level_mwh if sky == "overcast" else 0.0)
        for hour, share in [(2, 0.0), *zip(range(6, 17), DAYLIGHT, strict=True)]:
            raw = PEAK_MWH[sky] * share
            usual = 40e3 + 3e3 * share
            rows.append(
                {
                    "time": pd.Timestamp(f"2024-06-{day:02d} {hour}:00", tz="UTC"),
                    "load_mwh": usual + level - self_consumed * raw + rng.normal(0, 5),
                    "raw_mwh": raw,
                    "sky_class": sky,
                }
            )
    table = pd.DataFrame(rows).set_index("time")
    time = pd.DatetimeIndex(table.index).as_unit("ns")
    frame = table.set_axis(time).assign(
        held_out_year=time.year.astype("int64"),
        # A published output that the dip does not read.
        published_mwh=table["raw_mwh"].to_numpy(),
        capacity_mw=CAPACITY_MW,
        eligible=True,
        season=time.month.map(SEASON_OF_MONTH),
    )
    return frame["load_mwh"], frame.drop(columns="load_mwh")


def test_a_load_that_includes_self_consumption_shows_no_dip() -> None:
    load, frame = load_predictions(self_consumed=0.0)

    found = signature.dip(signature.consumption_table(load, frame, "DE"))

    # Without the cell effect, the usual midday peak would read as a rise with the output.
    assert found.low < 0 < found.high


def test_a_load_net_of_self_consumption_falls_by_the_self_consumed_share() -> None:
    load, frame = load_predictions(self_consumed=0.2)

    found = signature.dip(signature.consumption_table(load, frame, "DE"))

    assert found.estimate == pytest.approx(-0.2, abs=0.01)
    assert found.high < 0
    assert (found.hours, found.days) == (231, 21)


def test_the_level_of_the_day_does_not_enter_the_dip() -> None:
    # Overcast days have less modelled output and, here, a higher load at every hour.
    load, frame = load_predictions(self_consumed=0.0, overcast_level_mwh=2000.0)

    found = signature.dip(signature.consumption_table(load, frame, "DE"))

    assert found.estimate == pytest.approx(0.0, abs=0.01)


def test_the_hours_of_a_dip_have_modelled_output_and_a_load() -> None:
    load, frame = load_predictions(self_consumed=0.0)
    missing = frame.index[1]
    load.loc[missing] = np.nan

    table = signature.consumption_table(load, frame, "DE")

    # 21 days of 11 hours with modelled output, less the hour without a load.
    assert len(table) == 230
    assert missing not in table.index
    assert table.columns.tolist() == ["cell", "season", "day", "load_mwh", "raw_mwh"]


def test_dips_cover_all_hours_and_each_season() -> None:
    load, frame = load_predictions(self_consumed=0.2)

    rows = signature.dips(signature.consumption_table(load, frame, "DE"))

    assert rows["season"].tolist() == ["all", "JJA"]
    assert rows["sign"].tolist() == ["negative", "negative"]


def consumption_control_rows(eco2mix: str, smard: str) -> pd.DataFrame:
    """The sign of each load on all its hours, and a seasonal row that never counts."""
    return pd.DataFrame(
        {
            "series": ["eco2mix_consumption", "smard_consumption", "eco2mix_consumption"],
            "season": ["all", "all", "JJA"],
            "sign": [eco2mix, smard, "negative"],
        }
    )


@pytest.mark.parametrize(
    ("eco2mix", "smard", "expected"),
    [
        ("none", "negative", (True, True, True)),
        ("positive", "negative", (True, True, True)),
        ("negative", "negative", (False, True, False)),
        ("none", "none", (True, False, False)),
        ("none", "positive", (True, False, False)),
    ],
)
def test_the_consumption_evidence_is_used_only_when_both_controls_pass(
    eco2mix: str, smard: str, expected: tuple[bool, bool, bool]
) -> None:
    found = signature.consumption_controls(consumption_control_rows(eco2mix, smard))

    assert (found["negative_control"], found["positive_control"], found["used"]) == expected


def test_the_consumption_command_writes_the_dips_and_the_controls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    french_load, french = load_predictions(self_consumed=0.0)
    german_load, german = load_predictions(self_consumed=0.2, seed=1)
    for name, frame in {"calibration-fr": french, "calibration-de": german}.items():
        (tmp_path / name).mkdir()
        frame.to_parquet(tmp_path / name / "solar-predictions.parquet")
    period = (french.index[0], french.index[-1] + pd.Timedelta(hours=1))

    def national_hourly(
        client: object, series: str, start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.DataFrame:
        assert (series, start, end) == ("load", *period)
        return pd.DataFrame({"energy_mwh": french_load, "complete": True})

    def fetch_hourly_load(
        region: str, start: pd.Timestamp, end: pd.Timestamp, **_: object
    ) -> pd.DataFrame:
        assert (region, start, end) == ("DE", *period)
        return pd.DataFrame({"energy_mwh": german_load, "complete": True})

    monkeypatch.setattr(odre, "national_hourly", national_hourly)
    monkeypatch.setattr(smard, "fetch_hourly_load", fetch_hourly_load)
    output = tmp_path / "signature"

    cli.main(
        [
            "consumption-signature",
            str(output),
            "--calibration-fr",
            str(tmp_path / "calibration-fr"),
            "--calibration-de",
            str(tmp_path / "calibration-de"),
        ]
    )

    summary = json.loads((output / "consumption-signature.json").read_text(encoding="utf-8"))
    assert summary["controls"] == {"negative_control": True, "positive_control": True, "used": True}
    assert set(summary["predictions"]) == {"FR", "DE"}
    dips = {row["series"]: row["sign"] for row in summary["dips"]}
    assert dips == {"eco2mix_consumption": "none", "smard_consumption": "negative"}
    rows = pd.read_parquet(output / "consumption-signature.parquet")
    assert len(rows) == 4
    hours = pd.read_parquet(output / "consumption-signature-hours.parquet")
    overall = rows[rows["season"] == "all"].set_index("series")["hours"]
    assert hours.groupby("series").size().to_dict() == overall.to_dict()
    # Over the hours of the dip, those with modelled output.
    daylight = french["raw_mwh"] > 0
    french_share = french["raw_mwh"][daylight].sum() / french_load[daylight].sum()
    assert summary["modelled_over_load"]["eco2mix_consumption"] == pytest.approx(french_share)
