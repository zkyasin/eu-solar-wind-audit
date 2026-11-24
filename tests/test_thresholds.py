import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from eu_solar_wind_audit import cli, physics
from eu_solar_wind_audit.anomalies import breaks, inputs, thresholds
from eu_solar_wind_audit.anomalies.inputs import DetectionInputs, Published, SeriesInputs
from eu_solar_wind_audit.config import CONFIG_DIR, load_country
from eu_solar_wind_audit.sources import mastr

PROVENANCE = "RTE, éCO2mix, ODRÉ dataset eco2mix-national-cons-def"


def hours_of(start: str, end: str) -> pd.DatetimeIndex:
    return pd.date_range(
        pd.Timestamp(start, tz="UTC"),
        pd.Timestamp(end, tz="UTC"),
        freq="h",
        inclusive="left",
        unit="ns",
        name="time",
    )


def series_inputs(
    hours: pd.DatetimeIndex, technology: str, modelled_mwh: np.ndarray, seed: int
) -> SeriesInputs:
    """A definitive series that follows its model with a noise of 5 %, at 80 % of its level."""
    noise = np.random.default_rng(seed).normal(1.0, 0.05, hours.size)
    published = pd.DataFrame(
        {"energy_mwh": 0.8 * modelled_mwh * noise, "complete": True, "status": "definitive"},
        index=hours,
    )
    modelled = pd.DataFrame(
        {
            "energy_mwh": modelled_mwh,
            "complete": True,
            "capacity_mw": 100.0,
            "capacity_provisional": False,
            "era5_source": "ERA5",
        },
        index=hours,
    )
    series = f"odre.FR.{technology}"
    return SeriesInputs(
        technology=technology,
        published=(Published(series, PROVENANCE, False, published),),
        modelled=physics.Modelled.validate(modelled),
        capacity_mw=pd.Series(100.0, index=hours),
        offshore=pd.Series(False, index=hours) if technology == "wind" else None,
    )


def detection_inputs(start: str = "2019-01-01", end: str = "2023-01-01") -> DetectionInputs:
    """Mainland France with its solar and wind series, every hour of reference price class."""
    hours = hours_of(start, end)
    hour = hours.hour.to_numpy()
    season = 1 + 0.3 * np.cos(2 * np.pi * (hours.day_of_year.to_numpy() - 172) / 365.25)
    solar_mwh = 60 * season * np.sin(np.pi * (hour + 0.5 - 6) / 12).clip(0)
    wind_mwh = 30 + 20 * np.sin(2 * np.pi * np.arange(hours.size) / 97)
    classes = pd.DataFrame(
        {"has_negative_quarter": False, "near_zero_negative": False, "regime": "reference"},
        index=hours,
    )
    return DetectionInputs(
        country="FR",
        start=hours[0],
        end=hours[-1] + pd.Timedelta(hours=1),
        hours=hours,
        rows=(),
        series={
            "odre.FR.solar": series_inputs(hours, "solar", solar_mwh, 1),
            "odre.FR.wind": series_inputs(hours, "wind", wind_mwh, 2),
        },
        price_classes=classes,
    )


REFERENCE = thresholds.reference_model(load_country(CONFIG_DIR / "fr.toml"), "v0-test")


@pytest.fixture(scope="module")
def fixed() -> thresholds.Thresholds:
    return thresholds.fix(detection_inputs(), REFERENCE)


def test_curves_of_the_same_shape_have_no_centroid_bias_and_no_shift(
    fixed: thresholds.Thresholds,
) -> None:
    assert fixed.centroid_bias_minutes == pytest.approx(0.0, abs=1.0)
    assert fixed.shift_series == ("odre.FR.solar",)


def test_thresholds_come_from_their_grids(fixed: thresholds.Thresholds) -> None:
    assert set(fixed.residual_thresholds) == {"solar", "wind"}
    assert set(fixed.residual_thresholds.values()) <= set(thresholds.RESIDUAL_THRESHOLDS)
    assert set(fixed.penalties) <= set(breaks.PENALTIES)
    assert set(fixed.profiles) == {"odre.FR.solar", "odre.FR.wind"}


def test_false_alarms_are_counted_per_series_and_year(fixed: thresholds.Thresholds) -> None:
    table = fixed.false_alarms

    assert sorted(zip(table["series"], table["year"], strict=True)) == [
        (series, year)
        for series in ("odre.FR.solar", "odre.FR.wind")
        for year in thresholds.FIX_YEARS
    ]
    detectors = table[["timestamp_shift", "residual_outlier", "level_break"]].sum(axis="columns")
    assert table["total"].equals(detectors)


def test_written_thresholds_read_back_unchanged(
    fixed: thresholds.Thresholds, tmp_path: Path
) -> None:
    thresholds.write(tmp_path / "FR.json", fixed)
    read = thresholds.read(tmp_path / "FR.json")

    assert dataclasses.replace(read, false_alarms=fixed.false_alarms) == fixed
    pd.testing.assert_frame_equal(read.false_alarms, fixed.false_alarms)


def test_another_reference_model_is_refused(fixed: thresholds.Thresholds) -> None:
    # A new commit alone keeps the thresholds.
    thresholds.check_reference(fixed, REFERENCE | {"git_describe": "v1"})
    with pytest.raises(ValueError, match="another reference model: sdes_media"):
        thresholds.check_reference(fixed, REFERENCE | {"sdes_media": {"solar": 1}})


def test_thresholds_never_read_2023() -> None:
    with pytest.raises(ValueError, match="thresholds rest on 2019-2022 only"):
        thresholds.fix(detection_inputs("2019-01-01", "2023-01-02"), REFERENCE)


def test_inputs_without_margin_cover_the_years_of_the_thresholds() -> None:
    assert pd.Timedelta(0) < inputs.MARGIN
    assert pd.Timestamp("2019-01-01", tz="UTC") == thresholds.FIX_START
    assert pd.Timestamp("2023-01-01", tz="UTC") == thresholds.FIX_END


def test_german_thresholds_rest_on_the_modelled_smard_series() -> None:
    french = detection_inputs("2019-01-01", "2020-01-01")
    series = {
        "smard.DE.solar": rename(french.series["odre.FR.solar"], "smard.DE.solar"),
        "smard.DE.wind_onshore": dataclasses.replace(
            rename(french.series["odre.FR.wind"], "smard.DE.wind_onshore"),
            technology="wind_onshore",
        ),
        # Without a model, a series gets no statistical detector.
        "energy_charts.DE.public_power.solar": dataclasses.replace(
            french.series["odre.FR.solar"], modelled=None
        ),
    }
    reference = thresholds.reference_model(load_country(CONFIG_DIR / "de.toml"), "v0-test")

    fixed = thresholds.fix(dataclasses.replace(french, country="DE", series=series), reference)

    assert fixed.country == "DE"
    assert fixed.shift_series == ("smard.DE.solar",)
    assert set(fixed.profiles) == {"smard.DE.solar", "smard.DE.wind_onshore"}
    assert set(fixed.residual_thresholds) == {"solar", "wind"}
    assert reference["mastr_export"] == mastr.EXPORT_URL
    assert "sdes_media" not in reference


def rename(item: SeriesInputs, series: str) -> SeriesInputs:
    (part,) = item.published
    return dataclasses.replace(item, published=(dataclasses.replace(part, series=series),))


def test_the_command_fixes_the_thresholds_of_2019_2022_without_margins(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timedelta]] = []

    def france_inputs(
        client: object, start: pd.Timestamp, end: pd.Timestamp, *_: object, margin: pd.Timedelta
    ) -> DetectionInputs:
        calls.append((start, end, margin))
        return detection_inputs()

    monkeypatch.setattr(inputs, "france_inputs", france_inputs)

    cli.main(["fix-anomaly-thresholds", "FR", "--output", str(tmp_path / "FR.json")])

    assert calls == [(thresholds.FIX_START, thresholds.FIX_END, pd.Timedelta(0))]
    assert thresholds.read(tmp_path / "FR.json").country == "FR"
