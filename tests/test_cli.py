import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import numpy as np
import pandas as pd
import pytest

from eu_solar_wind_audit import cli, eligibility, figures, fits, modulation, physics
from eu_solar_wind_audit.config import Country, load_country
from eu_solar_wind_audit.sources import odre, sdes, smard

FIXTURES = Path(__file__).parent / "fixtures" / "odre"
FR = Path(__file__).parents[1] / "config" / "countries" / "fr.toml"
REALTIME = FIXTURES / "eco2mix-national-tr_2026-10-24_2026-10-25.parquet"


def test_snapshot_realtime_saves_the_export_and_prints_both_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    content = REALTIME.read_bytes()
    transport = httpx.MockTransport(lambda request: httpx.Response(200, content=content))
    monkeypatch.setattr(odre, "make_client", lambda: httpx.Client(transport=transport))

    cli.main(["snapshot-realtime", str(tmp_path)])

    parquet, metadata = map(Path, capsys.readouterr().out.splitlines())
    assert parquet.read_bytes() == content
    assert metadata == parquet.with_suffix(".json")
    assert sorted(tmp_path.iterdir()) == [metadata, parquet]


def test_model_france_writes_the_series_and_its_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    hours = pd.date_range("2025-06-01", periods=3, freq="h", tz="UTC", name="time")
    hourly = pd.DataFrame(
        {
            "energy_mwh": [0.0, 1.0, 2.0],
            "complete": True,
            "capacity_mw": 10.0,
            "capacity_provisional": [False, True, True],
            "era5_source": ["ERA5", "ERA5", "ERA5T"],
        },
        index=hours,
    )
    calls = []

    def fake_model(
        technology: str, start: pd.Timestamp, end: pd.Timestamp, parameters: Country
    ) -> pd.DataFrame:
        calls.append((technology, start, end, parameters))
        return hourly

    monkeypatch.setattr(physics, "model_france", fake_model)
    output = tmp_path / "out" / "solar.parquet"

    cli.main(
        ["model-france", "solar", "2025-06-01", "2025-06-02", str(output), "--config", str(FR)]
    )

    parquet, sidecar = map(Path, capsys.readouterr().out.splitlines())
    assert (parquet, sidecar) == (output, output.with_suffix(".json"))
    pd.testing.assert_frame_equal(pd.read_parquet(parquet), hourly, check_freq=False)
    start, end = pd.Timestamp("2025-06-01", tz="UTC"), pd.Timestamp("2025-06-02", tz="UTC")
    assert calls == [("solar", start, end, load_country(FR))]
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    assert metadata["era5_sources"] == {"ERA5": 2, "ERA5T": 1}
    assert metadata["capacity"]["provisional_hours"] == 2
    assert metadata["capacity"]["url"].endswith(f"/media/{sdes.SOLAR_2026_Q2}/download")
    assert metadata["parameters"]["era5_box"] == {
        "west": -5.5,
        "south": 42.0,
        "east": 8.5,
        "north": 51.5,
    }
    assert metadata["parameters"]["solar"]["dc_ac_ratio"] == 1.2
    assert metadata["first_hour"] == "2025-06-01T00:00:00+00:00"
    assert "git_commit" in metadata
    assert "wind_models" not in metadata["parameters"]


def test_wind_metadata_records_the_model_of_each_generation() -> None:
    hourly = pd.DataFrame(
        {"capacity_provisional": [False], "era5_source": ["ERA5"]},
        index=pd.date_range("2025-06-01", periods=1, freq="h", tz="UTC"),
    )

    metadata = cli.run_metadata("wind_onshore", load_country(FR), hourly)

    models = metadata["parameters"]["wind_models"]
    assert [model["fleet"][0][1] for model in models] == [79.0, 90.0, 99.0]
    assert {model["smoothing_base_m_s"] for model in models} == {0.6}


def test_draw_week_takes_the_paris_week_from_its_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hours = pd.date_range("2025-06-01", "2025-06-10", freq="h", tz="UTC", inclusive="left")
    year = SimpleNamespace(
        published=pd.DataFrame({"energy_mwh": 1.0, "status": "definitive"}, index=hours),
        price_classes=pd.DataFrame(
            {"regime": np.where(hours.hour == 12, "negative", "reference")}, index=hours
        ),
        arrays=SimpleNamespace(covariates=pd.DataFrame({"era5_source": "ERA5"}, index=hours)),
    )
    monkeypatch.setattr(eligibility, "year_inputs", lambda *args: year)
    monkeypatch.setattr(modulation, "calibrated_potential", lambda *args: pd.Series(2.0, hours))
    monkeypatch.setattr(fits, "read", lambda path: SimpleNamespace(git_describe="fit-v1"))
    drawn: dict[str, Any] = {}

    def write_week(directory: Path, hourly: Any, negative: Any, footer: str) -> list[Path]:
        drawn.update(hourly=hourly, negative=negative, footer=footer)
        return []

    monkeypatch.setattr(figures, "write_week", write_week)

    cli.draw_week(tmp_path, load_country(FR), tmp_path, "Drawn.")

    week = drawn["hourly"]["wind_onshore"]
    assert week.index[0] == pd.Timestamp("2025-06-01 22:00", tz="UTC")
    assert len(week) == 168
    assert list(week.columns) == ["published_mwh", "potential_mwh"]
    assert drawn["negative"].sum() == 7
    assert "frozen fit fit-v1, ERA5 weather" in drawn["footer"]
    assert "éCO2mix definitive data" in drawn["footer"]


def test_draw_modulation_puts_each_yearly_estimate_beside_the_rte_volume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results = tmp_path / "modulation"
    results.mkdir()
    for technology in ("solar", "wind_onshore"):
        summary = {
            "git_describe": "run-v1",
            "fit_git_describe": "fit-v1",
            "bootstrap": {"replicates": 9},
        }
        (results / f"{technology}-modulation.json").write_text(json.dumps(summary))
        pd.DataFrame(
            {"modulation_gwh": [20.0, 700.0], "rte_gwh": [25.0, 640.0]},
            index=pd.Index([2020, 2024], name="month"),
        ).to_parquet(results / f"{technology}-rte-years.parquet")
        pd.DataFrame(
            {
                "low_gwh": [1.0, 10.0, 600.0],
                "high_gwh": [3.0, 30.0, 800.0],
                "estimate_gwh": [2.0, 20.0, 700.0],
            },
            index=pd.Index(["2020-05", "2020", "2024"], name="period"),
        ).to_parquet(results / f"{technology}-modulation-intervals.parquet")
    references = tmp_path / "rte"
    references.mkdir()
    manifest = {"files": {"a.csv": {"last_modified": "2026-11-20T16:49:18Z"}}}
    (references / "manifest.json").write_text(json.dumps(manifest))
    drawn: dict[str, Any] = {}

    def write_modulation(directory: Path, years: Any, footer: str) -> list[Path]:
        drawn.update(years=years, footer=footer)
        return []

    monkeypatch.setattr(figures, "write_modulation", write_modulation)

    cli.draw_modulation(tmp_path / "figures", results, references, "Drawn.")

    assert drawn["years"]["solar"].loc[2024].to_dict() == {
        "estimate_gwh": 700.0,
        "low_gwh": 600.0,
        "high_gwh": 800.0,
        "rte_gwh": 640.0,
    }
    assert "run-v1 with the frozen fit fit-v1, 9 bootstrap replicates" in drawn["footer"]
    assert "last updated 2026-11-20" in drawn["footer"]


def test_draw_sum_of_parts_shades_the_parts_gaps_of_german_solar_that_year(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fetch(region: str, technology: str, start: pd.Timestamp, end: pd.Timestamp) -> Any:
        hours = pd.date_range(start, end, freq="h", inclusive="left", name="time")
        energy = 4.0 if region == "DE" else 1.0
        return pd.DataFrame({"energy_mwh": energy, "complete": True}, index=hours)

    monkeypatch.setattr(smard, "fetch_hourly_feed_in", fetch)
    findings = pd.DataFrame(
        {
            "series": ["smard.DE.solar", "smard.DE.solar", "smard.DE.wind_onshore"],
            "kind": "parts_gap",
            "start": pd.to_datetime(["2025-08-01", "2022-07-14", "2025-08-01"], utc=True),
            "end": pd.to_datetime(["2025-08-22", "2022-07-15", "2025-08-22"], utc=True),
        }
    )
    results = tmp_path / "anomalies"
    results.mkdir()
    findings.to_parquet(results / "DE-findings.parquet")
    (results / "DE-detection.json").write_text(json.dumps({"git_describe": "detect-v1"}))
    drawn: dict[str, Any] = {}

    def write_sum_of_parts(
        directory: Path, gap: Any, flagged: Any, time_zone: str, title: str, footer: str
    ) -> list[Path]:
        drawn.update(gap=gap, flagged=flagged, footer=footer)
        return []

    monkeypatch.setattr(figures, "write_sum_of_parts", write_sum_of_parts)

    cli.draw_sum_of_parts(tmp_path / "figures", results, "Drawn.")

    assert len(drawn["gap"]) == 365
    assert (drawn["gap"] == 0).all()
    assert drawn["flagged"]["start"].tolist() == [pd.Timestamp("2025-08-01", tz="UTC")]
    assert "detect-anomalies at detect-v1" in drawn["footer"]
