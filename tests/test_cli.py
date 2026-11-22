import json
from pathlib import Path

import httpx
import pandas as pd
import pytest

from eu_solar_wind_audit import cli, physics
from eu_solar_wind_audit.config import Country, load_country
from eu_solar_wind_audit.sources import odre, sdes

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
    assert calls == [
        ("solar", pd.Timestamp("2025-06-01"), pd.Timestamp("2025-06-02"), load_country(FR))
    ]
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
