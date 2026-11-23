import numpy as np
import pandas as pd
import pytest

from eu_solar_wind_audit.anomalies import breaks
from eu_solar_wind_audit.anomalies.breaks import Penalties, Ratios, SeasonalProfile

SIGMA = 0.05
FLAT = SeasonalProfile((0.0, 0.0, 0.0, 0.0, 0.0), SIGMA)


def utc(stamp: str) -> pd.Timestamp:
    return pd.Timestamp(stamp, tz="UTC")


def hours_of(start: str, end: str) -> pd.DatetimeIndex:
    return pd.date_range(utc(start), utc(end), freq="h", inclusive="left", unit="ns", name="time")


def daily(
    start: str, end: str, steps: dict[str, float] | None = None, seed: int = 0
) -> pd.DataFrame:
    """DailyRatios of the days [start, end): noise of SIGMA, plus each log step of `steps` from
    its day on."""
    days = pd.date_range(utc(start), utc(end), freq="D", inclusive="left", unit="ns", name="day")
    log_ratio = np.random.default_rng(seed).normal(0.0, SIGMA, days.size)
    for day, step in (steps or {}).items():
        log_ratio[days >= utc(day)] += step
    return pd.DataFrame(
        {
            "log_ratio": log_ratio,
            "usable_share": 1.0,
            "modelled_capacity_factor": 0.2,
            "used": True,
        },
        index=days,
    )


def flat(frame: pd.DataFrame) -> Ratios:
    return Ratios(breaks.DailyRatios.validate(frame), FLAT)


def modelled(hours: pd.DatetimeIndex, energy_mwh: np.ndarray, **changes: object) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "energy_mwh": energy_mwh,
            "complete": True,
            "capacity_mw": 100.0,
            "capacity_provisional": False,
            "era5_source": "ERA5",
            **changes,
        },
        index=hours,
    )


def reference(hours: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame({"regime": "reference"}, index=hours)


def bell(hours: pd.DatetimeIndex) -> np.ndarray:
    """Modelled solar energy from 06:00 to 18:00 UTC, up to 50 MWh at noon."""
    return 50 * np.sin(np.pi * (hours.hour.to_numpy() + 0.5 - 6) / 12).clip(0)


def test_the_daily_ratio_reads_the_usable_hours_of_each_day() -> None:
    hours = hours_of("2020-06-01", "2020-06-03")
    model = modelled(hours, bell(hours))
    published = pd.DataFrame(
        {"energy_mwh": 0.8 * bell(hours), "complete": True, "status": "definitive"}, index=hours
    )
    classes = reference(hours)
    # The second day loses its noon hours to a negative price: under 80 % of its energy is usable.
    classes.loc[utc("2020-06-02 10:00") : utc("2020-06-02 13:00"), "regime"] = "negative"

    found = breaks.daily_log_ratios(published, model, classes, technology="solar")

    assert found.index.equals(pd.DatetimeIndex([utc("2020-06-01"), utc("2020-06-02")], name="day"))
    assert found["log_ratio"].to_numpy() == pytest.approx(np.log([0.8, 0.8]))
    assert found["usable_share"].iloc[0] == pytest.approx(1.0)
    assert found["usable_share"].iloc[1] < breaks.USABLE_SHARE_FROM
    assert found["used"].tolist() == [True, False]


def test_real_time_offshore_and_calm_wind_hours_are_not_used() -> None:
    hours = hours_of("2022-06-01", "2022-06-04")
    model = modelled(hours, np.full(hours.size, 20.0))
    model.loc[hours >= utc("2022-06-03"), "energy_mwh"] = 2.0
    status = np.where(hours < utc("2022-06-02"), "definitive", "realtime")
    published = pd.DataFrame({"energy_mwh": 20.0, "complete": True, "status": status}, index=hours)
    offshore = pd.Series(hours >= utc("2022-06-01 12:00"), index=hours)

    found = breaks.daily_log_ratios(
        published, model, reference(hours), technology="wind", offshore=offshore
    )

    # Half of the first day holds offshore output; the other days are real-time, the last one
    # calm as well.
    assert found["usable_share"].tolist() == pytest.approx([0.5, 0.0, 0.0])
    assert not found["used"].any()
    assert found["modelled_capacity_factor"].tolist() == pytest.approx([0.2, 0.2, 0.02])


def test_the_profile_recovers_a_seasonal_wave_and_the_noise_scale() -> None:
    frame = daily("2019-01-01", "2023-01-01")
    angle = 2 * np.pi * pd.DatetimeIndex(frame.index).day_of_year.to_numpy() / breaks.YEAR_DAYS
    frame["log_ratio"] += 0.1 + 0.3 * np.cos(angle) - 0.2 * np.sin(2 * angle)

    profile = breaks.fit_profile(frame)

    assert profile.coefficients == pytest.approx((0.1, 0.3, 0.0, 0.0, -0.2), abs=0.01)
    assert profile.sigma == pytest.approx(SIGMA, rel=0.1)


def test_the_profile_rests_on_2019_2022_only() -> None:
    with pytest.raises(ValueError, match="2023-01-01 lies outside the profile years 2019-2022"):
        breaks.fit_profile(daily("2022-06-01", "2023-02-01"))


def test_a_step_of_minus_30_percent_is_dated_within_a_day() -> None:
    ratios = flat(daily("2020-01-01", "2021-01-01", {"2020-06-15": np.log(0.7)}))

    found = breaks.change_points({"solar": ratios}, penalty=300.0)

    (day,) = found["day"]
    assert abs((day - utc("2020-06-15")) / pd.Timedelta(days=1)) <= 1
    assert found["step"].item() == pytest.approx(np.log(0.7), abs=0.02)
    assert found["step_sigmas"].item() == pytest.approx(np.log(0.7) / SIGMA, abs=0.5)


def test_a_seasonal_wave_that_the_profile_removes_gives_no_break() -> None:
    frame = daily("2019-01-01", "2023-01-01")
    angle = 2 * np.pi * pd.DatetimeIndex(frame.index).day_of_year.to_numpy() / breaks.YEAR_DAYS
    frame["log_ratio"] += 0.4 * np.cos(angle)
    profile = breaks.fit_profile(frame)

    found = breaks.change_points(
        {"solar": Ratios(breaks.DailyRatios.validate(frame), profile)}, penalty=300.0
    )

    assert found.empty


def test_wind_is_never_searched_across_the_switch_to_onshore_wind() -> None:
    frame = daily("2023-07-01", "2024-07-01", {"2024-01-01": np.log(0.7)})

    assert breaks.change_points({"wind": flat(frame)}, penalty=300.0).empty
    # The same days as solar give the break.
    assert breaks.change_points({"solar": flat(frame)}, penalty=300.0)["day"].tolist() == [
        utc("2024-01-01")
    ]


PENALTIES = Penalties(solar=300.0, wind=200.0, joint=300.0)
SERIES: dict[breaks.Technology, str] = {"solar": "odre.FR-84.solar", "wind": "odre.FR-84.wind"}
PROVENANCE: dict[breaks.Technology, str] = {
    "solar": "RTE, éCO2mix, ODRÉ dataset eco2mix-regional-cons-def",
    "wind": "RTE, éCO2mix, ODRÉ dataset eco2mix-regional-cons-def",
}


def area(
    solar_steps: dict[str, float], wind_steps: dict[str, float]
) -> dict[breaks.Technology, Ratios]:
    return {
        "solar": flat(daily("2019-01-01", "2021-03-01", solar_steps, seed=1)),
        "wind": flat(daily("2019-01-01", "2021-03-01", wind_steps, seed=2)),
    }


def no_junctions() -> pd.DataFrame:
    return breaks.Junctions.empty()


def detect(
    ratios: dict[breaks.Technology, Ratios], junctions: pd.DataFrame, start: str = "2020-01-01"
) -> breaks.LevelBreaks:
    return breaks.level_breaks(
        ratios,
        junctions,
        series=SERIES,
        provenance=PROVENANCE,
        penalties=PENALTIES,
        start=utc(start),
        end=utc("2021-01-01"),
    )


def test_steps_of_solar_and_wind_on_the_same_days_are_a_break_of_both() -> None:
    step = np.log(0.7)

    found = detect(area({"2020-06-15": step}, {"2020-06-17": step}), no_junctions())

    assert sorted(found.findings["series"]) == sorted(SERIES.values())
    assert set(found.findings["kind"]) == {"level_break"}
    assert found.unattributed.empty


def test_steps_ten_days_apart_are_left_unattributed() -> None:
    step = np.log(0.7)

    found = detect(area({"2020-06-15": step}, {"2020-06-25": step}), no_junctions())

    assert found.findings.empty
    assert set(found.unattributed["reason"]) == {"single_technology"}
    assert sorted(found.unattributed["series"]) == sorted(SERIES.values())


def test_a_large_step_of_one_technology_is_left_unattributed() -> None:
    found = detect(area({"2020-06-15": np.log(0.5)}, {}), no_junctions())

    assert found.findings.empty
    assert found.unattributed["series"].tolist() == [SERIES["solar"]]
    assert found.unattributed["reason"].tolist() == ["single_technology"]


def test_a_common_step_near_a_junction_is_left_unattributed() -> None:
    step = np.log(0.7)
    junctions = pd.DataFrame(
        {"time": [utc("2020-06-18 00:00")], "kind": ["first_era5t_hour"]}
    ).astype({"time": "datetime64[ns, UTC]"})

    found = detect(area({"2020-06-15": step}, {"2020-06-15": step}), junctions)

    assert found.findings.empty
    assert set(found.unattributed["reason"]) == {"near_junction"}


def test_only_breaks_of_the_period_are_reported() -> None:
    step = np.log(0.7)

    found = detect(area({"2020-06-15": step}, {"2020-06-15": step}), no_junctions(), "2020-07-01")

    assert found.findings.empty
    assert found.unattributed.empty


def test_daily_ratios_must_reach_15_days_past_the_period() -> None:
    ratios: dict[breaks.Technology, Ratios] = {
        technology: flat(daily("2019-01-01", "2021-01-10", seed=seed))
        for seed, technology in enumerate(breaks.TECHNOLOGIES)
    }

    with pytest.raises(ValueError, match=r"the breaks up to 2021-01-01 .* are deferred"):
        detect(ratios, no_junctions())


def test_junctions_are_the_first_era5t_and_provisional_hours_and_status_changes() -> None:
    hours = hours_of("2020-11-30", "2020-12-03")
    model = modelled(
        hours,
        np.full(hours.size, 1.0),
        era5_source=np.where(hours >= utc("2020-12-02"), "ERA5T", "ERA5"),
        capacity_provisional=hours >= utc("2020-12-01 06:00"),
    )
    status = np.where(hours >= utc("2020-12-01"), "consolidated", "definitive")
    published = pd.DataFrame({"energy_mwh": 1.0, "complete": True, "status": status}, index=hours)

    found = breaks.junctions(published, model)

    assert list(zip(found["kind"], found["time"], strict=True)) == [
        ("first_era5t_hour", utc("2020-12-02")),
        ("first_provisional_capacity_hour", utc("2020-12-01 06:00")),
        ("status_change", utc("2020-12-01")),
    ]
