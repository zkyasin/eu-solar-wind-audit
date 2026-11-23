import dataclasses
import json

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
from pandera.typing import DataFrame

from eu_solar_wind_audit import physics
from eu_solar_wind_audit.anomalies import breaks, detection, inputs, thresholds
from eu_solar_wind_audit.anomalies.inputs import DetectionInputs, Published, Rows, SeriesInputs
from eu_solar_wind_audit.config import CONFIG_DIR, load_country
from eu_solar_wind_audit.sources import odre, smard

START, END = pd.Timestamp("2019-01-01", tz="UTC"), pd.Timestamp("2019-03-01", tz="UTC")
HOURS = pd.date_range(
    START - inputs.MARGIN, END + inputs.MARGIN, freq="h", inclusive="left", unit="ns", name="time"
)
FRENCH_AREAS = ["FR", *(f"FR-{code}" for code in odre.REGIONS)]
PROVENANCE = "Producer, test dataset"
# Published over modelled output, and the noise of the published hours.
LEVELS = {"solar": 0.9, "wind": 0.95}
NOISE = {"solar": 0.02, "wind": 0.03}
CAPACITY_MW = {"solar": 1000.0, "wind": 2000.0}


def utc(stamp: str) -> pd.Timestamp:
    return pd.Timestamp(stamp, tz="UTC")


def at(stamp: str) -> npt.NDArray[np.bool_]:
    """Whether each of HOURS is the hour starting at `stamp`, UTC."""
    return np.asarray(HOURS.isin([utc(stamp)]))


def capacity_factors(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Modelled capacity factors on HOURS: solar from 6 to 18 h UTC with a clearness per day, and
    wind between 0.15 and 0.55."""
    hour = HOURS.hour.to_numpy()
    day = ((HOURS - HOURS[0]) // pd.Timedelta(days=1)).to_numpy()
    clearness = rng.uniform(0.6, 1.0, size=day.max() + 1)[day]
    solar = 0.7 * np.clip(np.sin(np.pi * (hour - 6) / 12), 0, None) * clearness
    wave = 0.15 * np.sin(2 * np.pi * np.arange(len(HOURS)) / 97)
    wind = 0.35 + wave + rng.uniform(-0.05, 0.05, len(HOURS))
    return {"solar": solar, "wind": wind}


def modelled(energy_mwh: np.ndarray, capacity_mw: float) -> DataFrame[physics.Modelled]:
    return physics.Modelled.validate(
        pd.DataFrame(
            {
                "energy_mwh": energy_mwh,
                "complete": True,
                "capacity_mw": capacity_mw,
                "capacity_provisional": False,
                "era5_source": "ERA5",
            },
            index=HOURS,
        )
    )


def published(energy_mwh: pd.Series) -> pd.DataFrame:
    return odre.Hourly.validate(
        pd.DataFrame({"energy_mwh": energy_mwh, "complete": True, "status": "definitive"})
    )


def reference_classes(hours: pd.DatetimeIndex = HOURS) -> pd.DataFrame:
    return pd.DataFrame(
        {"has_negative_quarter": False, "near_zero_negative": False, "regime": "reference"},
        index=hours,
    )


def france() -> DetectionInputs:
    """Mainland France and 12 identical regions over HOURS, the national series the sum of the
    regions, with one defect of each kind; and the rows of one hourly dataset of FR-11 solar,
    without its stamps of 2018-12-20 10:00 and 2019-01-05 10:00."""
    rng = np.random.default_rng(2019)
    factors = capacity_factors(rng)
    regions = len(odre.REGIONS)
    energy = {}
    for technology, factor in factors.items():
        model = factor * CAPACITY_MW[technology]
        noise = rng.normal(0.0, NOISE[technology], len(HOURS))
        energy[technology] = (model, pd.Series(LEVELS[technology] * model * (1 + noise), HOURS))

    values: dict[str, pd.Series] = {}
    for area in FRENCH_AREAS:
        scale = regions if area == "FR" else 1
        for technology, (_, output) in energy.items():
            values[f"odre.{area}.{technology}"] = scale * output
    # Defects in the regions only, so that mainland France keeps the sum of the regions without
    # them: a negative value in the period and one in the margin before it, frozen wind, a
    # joint drop of solar and wind, a peak and a shift of the timestamps by one hour.
    values["odre.FR-11.solar"][at("2019-01-10 02:00")] = -20.0
    values["odre.FR-32.solar"][at("2018-12-10 02:00")] = -20.0
    values["odre.FR-24.wind"].loc[utc("2019-01-20 00:00") : utc("2019-01-21 05:00")] = 500.0
    for technology in ("solar", "wind"):
        values[f"odre.FR-27.{technology}"].loc[utc("2019-02-01") :] *= 0.7
    values["odre.FR-28.solar"][at("2019-02-15 12:00")] *= 3
    shifted = values["odre.FR-44.solar"].shift(1)
    values["odre.FR-44.solar"].loc[utc("2019-02-10") :] = shifted.loc[utc("2019-02-10") :]

    series = {}
    for area in FRENCH_AREAS:
        scale = regions if area == "FR" else 1
        for technology, (model, _) in energy.items():
            name = f"odre.{area}.{technology}"
            capacity_mw = scale * CAPACITY_MW[technology]
            series[name] = SeriesInputs(
                technology=technology,
                published=(Published(name, PROVENANCE, False, published(values[name])),),
                modelled=modelled(scale * model, capacity_mw),
                capacity_mw=pd.Series(capacity_mw, index=HOURS),
                offshore=pd.Series(False, index=HOURS, name="offshore"),
            )
    stamps = HOURS.drop([utc("2018-12-20 10:00"), utc("2019-01-05 10:00")])
    rows = Rows(
        dataset="test.dataset",
        provenance=PROVENANCE,
        provisional=False,
        step=pd.Timedelta(hours=1),
        label="start",
        local_days=None,
        start=HOURS[0],
        end=HOURS[-1] + pd.Timedelta(hours=1),
        values={"odre.FR-11.solar": values["odre.FR-11.solar"][stamps]},
    )
    return DetectionInputs("FR", START, END, HOURS, (rows,), series, reference_classes())


def french_thresholds() -> thresholds.Thresholds:
    profiles = {
        f"odre.{area}.{technology}": breaks.SeasonalProfile(
            (float(np.log(level)), 0.0, 0.0, 0.0, 0.0), 0.01
        )
        for area in FRENCH_AREAS
        for technology, level in LEVELS.items()
    }
    return thresholds.Thresholds(
        country="FR",
        reference_model=thresholds.reference_model(load_country(CONFIG_DIR / "fr.toml"), None),
        centroid_bias_minutes=0.0,
        shift_series=tuple(f"odre.{area}.solar" for area in FRENCH_AREAS),
        residual_thresholds={"solar": 15.0, "wind": 10.0},
        penalties=breaks.Penalties(300.0, 150.0, 300.0),
        profiles=profiles,
        false_alarms=pd.DataFrame(),
    )


@pytest.fixture(scope="module")
def french_run() -> detection.Detection:
    return detection.detect(france(), french_thresholds(), START, END)


def kinds_of(found: pd.DataFrame, series: str) -> set[tuple[str, pd.Timestamp]]:
    return {
        (kind, start)
        for kind, start in found.loc[found["series"] == series, ["kind", "start"]].values
    }


@pytest.mark.parametrize(
    ("series", "kind", "start"),
    [
        ("odre.FR-11.solar", "gap", "2019-01-05 10:00"),
        ("odre.FR-11.solar", "negative", "2019-01-10 02:00"),
        ("odre.FR-24.wind", "frozen", "2019-01-20 00:00"),
        ("odre.FR-27.solar", "level_break", "2019-02-01 00:00"),
        ("odre.FR-27.wind", "level_break", "2019-02-01 00:00"),
        ("odre.FR-28.solar", "residual_outlier", "2019-02-15 12:00"),
        ("odre.FR-28.solar", "above_capacity", "2019-02-15 12:00"),
        ("odre.FR.solar", "parts_gap", "2019-01-10 02:00"),
    ],
)
def test_a_run_finds_each_kind_of_defect(
    french_run: detection.Detection, series: str, kind: str, start: str
) -> None:
    assert (kind, utc(start)) in kinds_of(french_run.findings, series)


def test_a_shift_of_one_hour_is_claimed(french_run: detection.Detection) -> None:
    found = french_run.findings
    shifts = found[(found["series"] == "odre.FR-44.solar") & (found["kind"] == "timestamp_shift")]

    assert len(shifts) == 1
    assert shifts["start"].iloc[0] == utc("2019-02-10")
    assert json.loads(shifts["evidence"].iloc[0])["shift_hours"] == 1


def test_only_findings_that_reach_the_period_are_kept(french_run: detection.Detection) -> None:
    found = french_run.findings

    assert ((found["end"] > START) & (found["start"] < END)).all()
    # The negative value and the missing stamp of the margin are left out.
    assert "odre.FR-32.solar" not in set(found.loc[found["kind"] == "negative", "series"])
    assert len(found[found["kind"] == "gap"]) == 1


def test_a_drop_of_one_technology_is_left_unattributed(french_run: detection.Detection) -> None:
    # The frozen wind of FR-24 lowers its daily ratio and leaves its solar unchanged.
    unattributed = french_run.unattributed

    assert set(unattributed["series"]) == {"odre.FR-24.wind"}
    assert set(unattributed["reason"]) == {"single_technology"}


def test_statistical_detectors_need_the_thresholds() -> None:
    with pytest.raises(ValueError, match="thresholds"):
        detection.detect(france(), None, START, END)


def test_a_series_without_a_profile_is_refused() -> None:
    fixed = french_thresholds()
    profiles = {key: value for key, value in fixed.profiles.items() if key != "odre.FR.wind"}
    without = dataclasses.replace(fixed, profiles=profiles)

    with pytest.raises(ValueError, match=r"no profile for odre\.FR\.wind"):
        detection.detect(france(), without, START, END)


def test_the_wind_of_an_area_is_onshore_wind_from_2024() -> None:
    hours = pd.date_range("2023-12-20", "2024-01-10", freq="h", inclusive="left", tz="UTC")
    hours = hours.as_unit("ns").rename("time")
    model = physics.Modelled.validate(
        pd.DataFrame(
            {
                "energy_mwh": 10.0,
                "complete": True,
                "capacity_mw": 100.0,
                "capacity_provisional": False,
                "era5_source": "ERA5",
            },
            index=hours,
        )
    )

    def wind(name: str, energy_mwh: float, offshore: bool) -> SeriesInputs:
        frame = odre.Hourly.validate(
            pd.DataFrame(
                {"energy_mwh": energy_mwh, "complete": True, "status": "definitive"}, index=hours
            )
        )
        return SeriesInputs(
            technology=name.rsplit(".", 1)[1],
            published=(Published(name, f"rows of {name}", False, frame),),
            modelled=model,
            capacity_mw=None,
            offshore=pd.Series(offshore, index=hours, name="offshore"),
        )

    found = DetectionInputs(
        "FR",
        hours[0],
        hours[-1],
        hours,
        (),
        {
            "odre.FR.wind": wind("odre.FR.wind", 12.0, True),
            "odre.FR.wind_onshore": wind("odre.FR.wind_onshore", 9.0, False),
        },
        reference_classes(hours),
    )

    read = detection.onshore_wind(found, "odre.FR")

    switch = breaks.WIND_SWITCH
    assert read.published.index.equals(hours)
    assert (read.published["energy_mwh"] == np.where(hours < switch, 12.0, 9.0)).all()
    assert (read.offshore == (hours < switch)).all()
    assert [
        (period.series, period.provenance, period.start, period.end) for period in read.periods
    ] == [
        ("odre.FR.wind", "rows of odre.FR.wind", hours[0], switch),
        (
            "odre.FR.wind_onshore",
            "rows of odre.FR.wind_onshore",
            switch,
            hours[-1] + pd.Timedelta(hours=1),
        ),
    ]


def germany() -> DetectionInputs:
    """SMARD Germany as the sum of its control areas and Energy-Charts public power at 0.97 times
    SMARD, except a gap between Germany and its areas at 2019-01-15 12:00 in solar, and public
    solar power 10 % above at 2019-02-05 12:00."""
    rng = np.random.default_rng(2020)
    factors = capacity_factors(rng)
    factors["wind_onshore"] = factors.pop("wind")
    factors["wind_offshore"] = factors["wind_onshore"] * 0.8
    series: dict[str, SeriesInputs] = {}

    def add(name: str, technology: str, energy_mwh: pd.Series) -> None:
        frame = pd.DataFrame({"energy_mwh": energy_mwh, "complete": True})
        series[name] = SeriesInputs(
            technology=technology,
            published=(Published(name, f"rows of {name}", False, frame),),
            modelled=None,
            capacity_mw=None,
            offshore=None,
        )

    for technology, factor in factors.items():
        areas = [
            area
            for area in inputs.SMARD_AREAS
            if area != "DE" and (technology != "wind_offshore" or area in smard.OFFSHORE_REGIONS)
        ]
        area_mwh = pd.Series(1000.0 * factor, index=HOURS)
        total = len(areas) * area_mwh
        for area in areas:
            add(f"smard.{area}.{technology}", technology, area_mwh)
        if technology == "solar":
            total[at("2019-01-15 12:00")] += 500.0
        public = 0.97 * total
        if technology == "solar":
            public[at("2019-02-05 12:00")] *= 1.1
        add(f"smard.DE.{technology}", technology, total)
        add(f"energy_charts.DE.public_power.{technology}", technology, public)
        add(f"energy_charts.DE.total_power.{technology}", technology, total)
    return DetectionInputs("DE", START, END, HOURS, (), series, reference_classes())


def test_germany_gets_the_rules_and_the_cross_source_checks() -> None:
    found = detection.detect(germany(), None, START, END).findings

    assert set(zip(found["series"], found["kind"], found["start"], strict=True)) == {
        ("smard.DE.solar", "parts_gap", utc("2019-01-15 12:00")),
        ("energy_charts.DE.public_power.solar", "cross_source_gap", utc("2019-02-05 12:00")),
    }
