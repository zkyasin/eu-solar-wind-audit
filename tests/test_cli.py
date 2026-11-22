from pathlib import Path

import httpx
import pytest

from eu_solar_wind_audit import cli
from eu_solar_wind_audit.sources import odre

FIXTURES = Path(__file__).parent / "fixtures" / "odre"
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
