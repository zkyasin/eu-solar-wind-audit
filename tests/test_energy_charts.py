import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import pytest
from pandera.errors import SchemaError
from pandera.typing import DataFrame

from eu_solar_wind_audit.grid import Hourly
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


def capacity(panel: pd.DataFrame, month_end: str, technology: str) -> float:
    selected = (panel["month_end"] == pd.Timestamp(month_end)) & (panel["technology"] == technology)
    row = panel[selected]
    return float(row["capacity_mw"].item())


def test_real_response_gives_end_of_month_capacity_in_mw(german_payload: dict[str, Any]) -> None:
    panel = energy_charts.parse_installed_power(german_payload)

    assert set(panel["technology"]) == {"solar_ac", "solar_dc", "wind_onshore", "wind_offshore"}
    # Published as 107.856, 118.147, 68.178 and 9.733 GW for December 2025.
    assert capacity(panel, "2025-12-31", "solar_ac") == pytest.approx(107_856)
    assert capacity(panel, "2025-12-31", "solar_dc") == pytest.approx(118_147)
    assert capacity(panel, "2025-12-31", "wind_onshore") == pytest.approx(68_178)
    assert capacity(panel, "2025-12-31", "wind_offshore") == pytest.approx(9_733)


def test_real_response_leaves_out_storage_and_biomass(german_payload: dict[str, Any]) -> None:
    panel = energy_charts.parse_installed_power(german_payload)

    assert list(panel.columns) == ["month_end", "technology", "capacity_mw"]
    assert not panel["technology"].isin(["biomass", "battery_storage_power"]).any()


def test_summer_months_are_not_shifted_to_the_previous_month(
    german_payload: dict[str, Any],
) -> None:
    # Summer stamps carry +02:00: in UTC, 1 April 00:00 is still 31 March.
    panel = energy_charts.parse_installed_power(german_payload)

    for technology, series in panel.groupby("technology"):
        months = pd.DatetimeIndex(series["month_end"])
        expected = pd.date_range(months.min(), months.max(), freq="ME")
        assert months.equals(expected), technology
    assert capacity(panel, "2025-04-30", "solar_ac") == pytest.approx(97_299)
    assert capacity(panel, "2025-10-31", "solar_ac") == pytest.approx(104_770)


def test_offshore_wind_starts_when_it_is_first_reported(german_payload: dict[str, Any]) -> None:
    panel = energy_charts.parse_installed_power(german_payload)

    offshore = panel[panel["technology"] == "wind_offshore"]
    assert offshore["month_end"].min() == pd.Timestamp("2009-08-31")
    assert panel["capacity_mw"].notna().all()
    assert panel[panel["technology"] == "solar_ac"]["month_end"].min() == pd.Timestamp("2002-01-31")


def test_month_in_progress_is_dropped(german_payload: dict[str, Any]) -> None:
    panel = energy_charts.parse_installed_power(german_payload)

    assert panel["month_end"].max() == pd.Timestamp("2026-10-31")


def test_month_ending_at_generation_time_is_kept() -> None:
    data = [
        {"timestamp": "2026-10-01T00:00:00+02:00", "values": {"solar_ac": 1.0}},
        {"timestamp": "2026-11-01T00:00:00+01:00", "values": {"solar_ac": 2.0}},
    ]

    panel = energy_charts.parse_installed_power(
        payload(data, units={"solar_ac": None}, generated_at="2026-11-01T00:00:00+01:00")
    )

    assert panel["month_end"].tolist() == [pd.Timestamp("2026-10-31")]


def test_units_are_read_per_series_with_the_envelope_as_default() -> None:
    data = [{"timestamp": "2025-01-01T00:00:00+01:00", "values": {"solar_ac": 2.5, "solar_dc": 3}}]

    panel = energy_charts.parse_installed_power(
        payload(data, units={"solar_ac": None, "solar_dc": "MW"})
    )

    assert capacity(panel, "2025-01-31", "solar_ac") == pytest.approx(2_500)
    assert capacity(panel, "2025-01-31", "solar_dc") == pytest.approx(3)


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

    assert sorted(zip(panel["month_end"].dt.month, panel["technology"], strict=True)) == [
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
    assert capacity(panel, "2025-12-31", "solar_ac") == pytest.approx(107_856)
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


def test_rate_limits_server_errors_and_timeouts_are_retried(
    tmp_path: Path, sleeps: list[float]
) -> None:
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


# Fixtures: https://api.energy-charts.info/v2/<public_power|total_power>?country=de&start=<day>&end=<day>
# for 2025-03-30, 2025-06-21 and 2025-10-26, retrieved 2026-11-22, cut to the solar and wind series.
FIXTURES = FIXTURE.parent


def power_content(endpoint: str, day: str) -> bytes:
    return (FIXTURES / f"{endpoint}_de_{day}.json").read_bytes()


def month_content(endpoint: str, month: pd.Period) -> bytes:
    """A response with the same values in every quarter-hour of `month`."""
    payload = json.loads(power_content(endpoint, "2025-06-21"))
    stamps = pd.date_range(month.start_time, month.end_time, freq="15min", tz="Europe/Berlin")
    values = dict.fromkeys(energy_charts.POWER_SERIES, 1.0)
    payload["data"] = [{"timestamp": stamp.isoformat(), "values": values} for stamp in stamps]
    return json.dumps(payload).encode()


def power_server(
    contents: dict[tuple[str, str], bytes], requests: list[httpx.Request]
) -> httpx.Client:
    """Serves each (endpoint, start parameter) pair; anything else is a 404."""

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        key = (request.url.path.rsplit("/", 1)[-1], request.url.params["start"])
        if key in contents:
            return httpx.Response(200, content=contents[key])
        return httpx.Response(404)

    return httpx.Client(transport=httpx.MockTransport(handler))


def day_contents(day: str) -> dict[tuple[str, str], bytes]:
    """Each endpoint answers the month of `day` with the recorded response for that day."""
    month_start = str(pd.Timestamp(day).to_period("M").start_time.date())
    return {
        (endpoint, month_start): power_content(endpoint, day)
        for endpoint in ("public_power", "total_power")
    }


def local_day(day: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(day, tz="Europe/Berlin")
    return start.tz_convert("UTC"), (start + pd.DateOffset(days=1)).tz_convert("UTC")


def published_power(endpoint: str, day: str, technology: str) -> pd.Series:
    rows = json.loads(power_content(endpoint, day))["data"]
    stamps = pd.to_datetime([row["timestamp"] for row in rows], utc=True)
    return pd.Series([row["values"][technology] for row in rows], index=stamps, dtype=float)


def fetch_day(
    day: str, tmp_path: Path, contents: dict[tuple[str, str], bytes] | None = None
) -> dict[str, DataFrame[Hourly]]:
    start, end = local_day(day)
    with power_server(contents or day_contents(day), []) as client:
        return energy_charts.fetch_hourly_power(start, end, client=client, cache_dir=tmp_path)


def test_summer_day_matches_the_published_daily_solar_energy(tmp_path: Path) -> None:
    power = fetch_day("2025-06-21", tmp_path)

    assert set(power) == {
        "public.solar",
        "public.wind_onshore",
        "public.wind_offshore",
        "total.solar",
        "total.wind_onshore",
        "total.wind_offshore",
        "self_consumption.solar",
    }
    for hourly in power.values():
        assert len(hourly) == 24
        assert hourly["complete"].all()
    # Energy-Charts shows 421.5 GWh public and 515.4 GWh total solar for this day.
    assert power["public.solar"]["energy_mwh"].sum() == pytest.approx(421_478, abs=1)
    assert power["total.solar"]["energy_mwh"].sum() == pytest.approx(515_402, abs=1)
    assert power["self_consumption.solar"]["energy_mwh"].sum() == pytest.approx(93_924, abs=1)


def test_an_hour_is_the_mean_of_the_four_quarter_hours_that_start_in_it(tmp_path: Path) -> None:
    power = fetch_day("2025-06-21", tmp_path)
    hour = pd.Timestamp("2025-06-21 10:00", tz="UTC")

    published = published_power("total_power", "2025-06-21", "solar")
    quarters = published[hour : hour + pd.Timedelta(minutes=45)]
    assert len(quarters) == 4
    assert power["total.solar"].at[hour, "energy_mwh"] == pytest.approx(quarters.mean())


@pytest.mark.parametrize(("day", "hours"), [("2025-03-30", 23), ("2025-10-26", 25)])
def test_clock_change_days_have_all_their_utc_hours(day: str, hours: int, tmp_path: Path) -> None:
    power = fetch_day(day, tmp_path)

    for hourly in power.values():
        assert len(hourly) == hours
        assert hourly["complete"].all()
    for endpoint, perimeter in energy_charts.PERIMETERS.items():
        for technology in energy_charts.POWER_SERIES:
            published = published_power(endpoint, day, technology)
            assert len(published) == 4 * hours
            energy = power[f"{perimeter}.{technology}"]["energy_mwh"]
            assert energy.sum() == pytest.approx(published.sum() / 4)


@pytest.mark.parametrize(("endpoint", "perimeter"), energy_charts.PERIMETERS.items())
def test_self_consumption_is_incomplete_when_either_endpoint_is_missing(
    endpoint: str, perimeter: str, tmp_path: Path
) -> None:
    contents = day_contents("2025-06-21")
    payload = json.loads(contents[(endpoint, "2025-06-01")])
    payload["data"][48]["values"]["solar"] = None  # 12:00 local, 10:00 UTC
    contents[(endpoint, "2025-06-01")] = json.dumps(payload).encode()

    complete = pd.concat(fetch_day("2025-06-21", tmp_path, contents))["complete"]

    hour = pd.Timestamp("2025-06-21 10:00", tz="UTC")
    assert complete[~complete].index.tolist() == [
        (f"{perimeter}.solar", hour),
        ("self_consumption.solar", hour),
    ]


def test_hours_without_published_values_are_incomplete(tmp_path: Path) -> None:
    # The recorded June response holds 21 June only, which ends at 22:00 UTC.
    start = pd.Timestamp("2025-06-21 21:00", tz="UTC")
    with power_server(day_contents("2025-06-21"), []) as client:
        power = energy_charts.fetch_hourly_power(
            start, start + pd.Timedelta(hours=2), client=client, cache_dir=tmp_path
        )

    for hourly in power.values():
        assert hourly["complete"].tolist() == [True, False]


def test_whole_local_months_are_requested_from_both_endpoints(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []
    # 21:00 UTC is still 30 June in Berlin, 22:00 UTC is already 1 July.
    start = pd.Timestamp("2025-06-30 21:00", tz="UTC")
    contents = day_contents("2025-06-21")
    for endpoint in ("public_power", "total_power"):
        no_data = {**json.loads(contents[(endpoint, "2025-06-01")]), "data": []}
        contents[(endpoint, "2025-07-01")] = json.dumps(no_data).encode()
    with power_server(contents, requests) as client:
        energy_charts.fetch_hourly_power(
            start, start + pd.Timedelta(hours=2), client=client, cache_dir=tmp_path
        )

    asked = [(request.url.path, dict(request.url.params)) for request in requests]
    assert asked == [
        ("/v2/public_power", {"country": "de", "start": "2025-06-01", "end": "2025-06-30"}),
        ("/v2/public_power", {"country": "de", "start": "2025-07-01", "end": "2025-07-31"}),
        ("/v2/total_power", {"country": "de", "start": "2025-06-01", "end": "2025-06-30"}),
        ("/v2/total_power", {"country": "de", "start": "2025-07-01", "end": "2025-07-31"}),
    ]


def test_settled_months_are_cached_and_the_last_two_are_downloaded_again(tmp_path: Path) -> None:
    this_month = pd.Timestamp.now(tz="Europe/Berlin").tz_localize(None).to_period("M")
    months = [this_month - 2, this_month - 1, this_month]
    contents = {
        ("public_power", str(month.start_time.date())): month_content("public_power", month)
        for month in months
    }
    requests: list[httpx.Request] = []

    for _ in range(2):
        with power_server(contents, requests) as client:
            for month in months:
                energy_charts.fetch_power_month("public_power", month, client, tmp_path)

    starts = [request.url.params["start"] for request in requests]
    assert starts.count(str(months[0].start_time.date())) == 1
    assert starts.count(str(months[1].start_time.date())) == 2
    assert starts.count(str(months[2].start_time.date())) == 2
    assert [path.name for path in tmp_path.iterdir()] == [f"public_power_de_{months[0]}.json"]


def test_a_settled_month_with_a_missing_value_is_downloaded_again(tmp_path: Path) -> None:
    payload = json.loads(month_content("public_power", pd.Period("2020-01")))
    payload["data"][48]["values"]["solar"] = None
    contents = {("public_power", "2020-01-01"): json.dumps(payload).encode()}
    requests: list[httpx.Request] = []

    with power_server(contents, requests) as client:
        for _ in range(2):
            energy_charts.fetch_power_month("public_power", pd.Period("2020-01"), client, tmp_path)

    assert len(requests) == 2
    assert list(tmp_path.iterdir()) == []


def test_a_settled_month_with_missing_quarter_hours_is_downloaded_again(tmp_path: Path) -> None:
    # The API leaves out the quarter-hours it has no values for: here, all of June but the 21st.
    contents = {("public_power", "2025-06-01"): power_content("public_power", "2025-06-21")}
    requests: list[httpx.Request] = []

    with power_server(contents, requests) as client:
        for _ in range(2):
            energy_charts.fetch_power_month("public_power", pd.Period("2025-06"), client, tmp_path)

    assert len(requests) == 2
    assert list(tmp_path.iterdir()) == []


def test_power_under_another_licence_is_rejected_and_not_cached(tmp_path: Path) -> None:
    payload = json.loads(power_content("public_power", "2025-06-21"))
    payload["license"] = "private and internal use only"
    contents = {("public_power", "2020-01-01"): json.dumps(payload).encode()}

    with power_server(contents, []) as client, pytest.raises(ValueError, match="licence"):
        energy_charts.fetch_power_month("public_power", pd.Period("2020-01"), client, tmp_path)

    assert list(tmp_path.iterdir()) == []


def test_a_cached_response_is_checked_for_its_licence_too(tmp_path: Path) -> None:
    payload = json.loads(power_content("public_power", "2025-06-21"))
    payload["license"] = "private and internal use only"
    (tmp_path / "public_power_de_2020-01.json").write_text(json.dumps(payload))

    with power_server({}, []) as client, pytest.raises(ValueError, match="licence"):
        energy_charts.fetch_power_month("public_power", pd.Period("2020-01"), client, tmp_path)


@pytest.mark.parametrize(
    ("change", "message"),
    [({"interval_minutes": 60}, "interval"), ({"unit": "GW"}, "units")],
)
def test_power_in_another_interval_or_unit_is_rejected(
    change: dict[str, Any], message: str
) -> None:
    payload = {**json.loads(power_content("public_power", "2025-06-21")), **change}

    with pytest.raises(ValueError, match=message):
        energy_charts.parse_power(payload)
