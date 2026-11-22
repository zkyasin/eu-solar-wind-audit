import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import pytest
from pandera.errors import SchemaError

from eu_solar_wind_audit.sources import energy_charts

# Fixture: https://api.energy-charts.info/v2/installed_power?country=de&time_step=monthly&installation_decommission=false,
# retrieved 2026-11-22.
FIXTURE = Path(__file__).parent / "fixtures" / "energy_charts" / "installed_power_de_monthly.json"

CC_BY = "CC BY 4.0 (creativecommons.org/licenses/by/4.0), attribution: energy-charts.info"

Handler = Callable[[httpx.Request], httpx.Response]


@pytest.fixture
def german_payload() -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(FIXTURE.read_bytes())
    return payload


def payload(
    data: list[dict[str, Any]],
    *,
    units: dict[str, str | None],
    generated_at: str = "2026-11-22T14:57:13+01:00",
    licence: str = CC_BY,
) -> dict[str, Any]:
    """A /v2/installed_power response with the envelope unit set to GW."""
    return {
        "timezone": "Europe/Berlin",
        "unit": "GW",
        "license": licence,
        "generated_at": generated_at,
        "series": [{"id": series, "unit": unit} for series, unit in units.items()],
        "data": data,
    }


def capacity(panel: pd.DataFrame, month: str, technology: str) -> float:
    row = panel[(panel["month"] == pd.Timestamp(month)) & (panel["technology"] == technology)]
    return float(row["capacity_mw"].item())


def test_real_response_gives_end_of_month_capacity_in_mw(german_payload: dict[str, Any]) -> None:
    panel = energy_charts.parse_installed_power(german_payload)

    assert set(panel["technology"]) == {"solar_ac", "solar_dc", "wind_onshore", "wind_offshore"}
    # Published as 107.856, 118.147, 68.178 and 9.733 GW for December 2025.
    assert capacity(panel, "2025-12-01", "solar_ac") == pytest.approx(107_856)
    assert capacity(panel, "2025-12-01", "solar_dc") == pytest.approx(118_147)
    assert capacity(panel, "2025-12-01", "wind_onshore") == pytest.approx(68_178)
    assert capacity(panel, "2025-12-01", "wind_offshore") == pytest.approx(9_733)


def test_real_response_leaves_out_storage_and_biomass(german_payload: dict[str, Any]) -> None:
    panel = energy_charts.parse_installed_power(german_payload)

    assert list(panel.columns) == ["month", "technology", "capacity_mw"]
    assert not panel["technology"].isin(["biomass", "battery_storage_power"]).any()


def test_summer_months_are_not_shifted_to_the_previous_month(
    german_payload: dict[str, Any],
) -> None:
    # Summer stamps carry +02:00: in UTC, 1 April 00:00 is still 31 March.
    panel = energy_charts.parse_installed_power(german_payload)

    for technology, series in panel.groupby("technology"):
        months = pd.DatetimeIndex(series["month"])
        expected = pd.date_range(months.min(), months.max(), freq="MS")
        assert months.equals(expected), technology
    assert capacity(panel, "2025-04-01", "solar_ac") == pytest.approx(97_299)
    assert capacity(panel, "2025-10-01", "solar_ac") == pytest.approx(104_770)


def test_offshore_wind_starts_when_it_is_first_reported(german_payload: dict[str, Any]) -> None:
    panel = energy_charts.parse_installed_power(german_payload)

    offshore = panel[panel["technology"] == "wind_offshore"]
    assert offshore["month"].min() == pd.Timestamp("2009-08-01")
    assert panel["capacity_mw"].notna().all()
    assert panel[panel["technology"] == "solar_ac"]["month"].min() == pd.Timestamp("2002-01-01")


def test_month_in_progress_is_dropped(german_payload: dict[str, Any]) -> None:
    panel = energy_charts.parse_installed_power(german_payload)

    assert panel["month"].max() == pd.Timestamp("2026-10-01")


def test_month_ending_at_generation_time_is_kept() -> None:
    data = [
        {"timestamp": "2026-10-01T00:00:00+02:00", "values": {"solar_ac": 1.0}},
        {"timestamp": "2026-11-01T00:00:00+01:00", "values": {"solar_ac": 2.0}},
    ]

    panel = energy_charts.parse_installed_power(
        payload(data, units={"solar_ac": None}, generated_at="2026-11-01T00:00:00+01:00")
    )

    assert panel["month"].tolist() == [pd.Timestamp("2026-10-01")]


def test_units_are_read_per_series_with_the_envelope_as_default() -> None:
    data = [{"timestamp": "2025-01-01T00:00:00+01:00", "values": {"solar_ac": 2.5, "solar_dc": 3}}]

    panel = energy_charts.parse_installed_power(
        payload(data, units={"solar_ac": None, "solar_dc": "MW"})
    )

    assert capacity(panel, "2025-01-01", "solar_ac") == pytest.approx(2_500)
    assert capacity(panel, "2025-01-01", "solar_dc") == pytest.approx(3)


def test_unknown_unit_is_rejected() -> None:
    data = [{"timestamp": "2025-01-01T00:00:00+01:00", "values": {"solar_ac": 2.5}}]

    with pytest.raises(ValueError, match="kW"):
        energy_charts.parse_installed_power(payload(data, units={"solar_ac": "kW"}))


def test_null_and_absent_values_are_left_out_without_touching_the_others() -> None:
    data = [
        {"timestamp": "2025-01-01T00:00:00+01:00", "values": {"solar_ac": 1.0, "solar_dc": None}},
        {"timestamp": "2025-02-01T00:00:00+01:00", "values": {"solar_dc": 2.0}},
    ]

    panel = energy_charts.parse_installed_power(
        payload(data, units={"solar_ac": None, "solar_dc": None})
    )

    assert sorted(zip(panel["month"].dt.month, panel["technology"], strict=True)) == [
        (1, "solar_ac"),
        (2, "solar_dc"),
    ]


def test_response_under_another_licence_is_rejected() -> None:
    data = [{"timestamp": "2025-01-01T00:00:00+01:00", "values": {"solar_ac": 2.5}}]
    restricted = payload(data, units={"solar_ac": None}, licence="private and internal use only")

    with pytest.raises(ValueError, match="licence"):
        energy_charts.parse_installed_power(restricted)


def test_negative_capacity_is_rejected() -> None:
    data = [{"timestamp": "2025-01-01T00:00:00+01:00", "values": {"wind_onshore": -0.1}}]

    with pytest.raises(SchemaError, match="capacity_mw"):
        energy_charts.parse_installed_power(payload(data, units={"wind_onshore": None}))


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    delays: list[float] = []
    monkeypatch.setattr("time.sleep", delays.append)
    return delays


def replay(responses: list[httpx.Response | Exception], requests: list[httpx.Request]) -> Handler:
    """Answers each request with the next response, or raises the next exception."""

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        response = responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    return handler


def fetch(
    responses: list[httpx.Response | Exception],
    requests: list[httpx.Request],
    cache_dir: Path,
    *,
    refresh: bool = False,
) -> pd.DataFrame:
    transport = httpx.MockTransport(replay(responses, requests))
    with httpx.Client(transport=transport) as client:
        return energy_charts.fetch_installed_capacity(
            "de", client=client, cache_dir=cache_dir, refresh=refresh
        )


def test_fetch_requests_monthly_values_and_caches_the_raw_response(
    tmp_path: Path, sleeps: list[float]
) -> None:
    requests: list[httpx.Request] = []

    panel = fetch([httpx.Response(200, content=FIXTURE.read_bytes())], requests, tmp_path)

    assert len(requests) == 1
    assert str(requests[0].url).startswith(energy_charts.URL)
    assert dict(requests[0].url.params) == {"country": "de", "time_step": "monthly"}
    assert (tmp_path / "installed_power_de_monthly.json").read_bytes() == FIXTURE.read_bytes()
    assert capacity(panel, "2025-12-01", "solar_ac") == pytest.approx(107_856)
    assert sleeps == []


def test_cached_response_is_read_without_a_request_unless_refreshed(tmp_path: Path) -> None:
    (tmp_path / "installed_power_de_monthly.json").write_bytes(FIXTURE.read_bytes())
    requests: list[httpx.Request] = []

    cached = fetch([], requests, tmp_path)
    refreshed = fetch(
        [httpx.Response(200, content=FIXTURE.read_bytes())], requests, tmp_path, refresh=True
    )

    assert len(requests) == 1
    assert cached.equals(refreshed)


def test_rate_limit_waits_as_long_as_the_server_asks(tmp_path: Path, sleeps: list[float]) -> None:
    requests: list[httpx.Request] = []
    responses: list[httpx.Response | Exception] = [
        httpx.Response(429, headers={"Retry-After": "7"}),
        httpx.Response(503),
        httpx.ReadTimeout("timed out"),
        httpx.Response(200, content=FIXTURE.read_bytes()),
    ]

    fetch(responses, requests, tmp_path)

    assert len(requests) == 4
    assert sleeps == [7.0, *energy_charts.RETRY_DELAYS_SECONDS[1:]]


def test_persistent_server_errors_raise_and_cache_nothing(
    tmp_path: Path, sleeps: list[float]
) -> None:
    attempts = len(energy_charts.RETRY_DELAYS_SECONDS) + 1
    requests: list[httpx.Request] = []

    with pytest.raises(httpx.HTTPStatusError):
        fetch([httpx.Response(502)] * attempts, requests, tmp_path)

    assert len(requests) == attempts
    assert list(tmp_path.iterdir()) == []


def test_country_without_monthly_values_fails_at_once(tmp_path: Path, sleeps: list[float]) -> None:
    requests: list[httpx.Request] = []
    # Answer of the live API for country=fr, 2026-11-22.
    not_found = httpx.Response(404, text="monthly data is only available for: de")

    with pytest.raises(httpx.HTTPStatusError):
        fetch([not_found], requests, tmp_path)

    assert len(requests) == 1
    assert sleeps == []
    assert list(tmp_path.iterdir()) == []


def test_response_that_does_not_parse_is_not_cached(tmp_path: Path) -> None:
    data = [{"timestamp": "2025-01-01T00:00:00+01:00", "values": {"solar_ac": 2.5}}]
    body = json.dumps(payload(data, units={"solar_ac": "kW"})).encode()

    with pytest.raises(ValueError, match="kW"):
        fetch([httpx.Response(200, content=body)], [], tmp_path)

    assert list(tmp_path.iterdir()) == []
