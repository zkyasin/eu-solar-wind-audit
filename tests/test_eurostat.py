from collections.abc import Callable
from pathlib import Path

import httpx
import pandas as pd
import pytest
from pandera.errors import SchemaError

from eu_solar_wind_audit.sources import eurostat

# Fixture: https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/nrg_bal_peh/A.GEP.RA300+RA420.GWH.AT+BE+DE+DK+ES+FR+IT+NL+PL+PT/?format=SDMX-CSV&startPeriod=2019&compressed=false,
# retrieved 2026-11-24.
FIXTURE = Path(__file__).parent / "fixtures" / "eurostat" / "nrg_bal_peh.csv"

HEADER = FIXTURE.read_text().splitlines()[0]

Handler = Callable[[httpx.Request], httpx.Response]


def download(*rows: dict[str, str]) -> bytes:
    """An SDMX-CSV download of German solar in 2024, with each row's fields replaced."""
    lines = [HEADER]
    for fields in rows:
        row = {
            "freq": "A",
            "nrg_bal": "GEP",
            "siec": "RA420",
            "unit": "GWH",
            "geo": "DE",
            "year": "2024",
            "value": "75379.0",
            "flag": "",
            **fields,
        }
        lines.append(
            f"ESTAT:NRG_BAL_PEH(1.0),02/06/26 23:00:00,{row['freq']},{row['nrg_bal']},"
            f"{row['siec']},{row['unit']},{row['geo']},{row['year']},{row['value']},{row['flag']},"
        )
    return "\n".join(lines).encode()


def gwh(release: eurostat.Release, country: str, technology: str, year: int) -> float:
    production = release.production
    (value,) = production.loc[
        (production["country"] == country)
        & (production["technology"] == technology)
        & (production["year"] == year),
        "gwh",
    ]
    return float(value)


def test_real_download_gives_gross_production_in_gwh() -> None:
    release = eurostat.parse_gross_production(FIXTURE.read_bytes())

    assert gwh(release, "DE", "solar", 2024) == pytest.approx(75_379, abs=1)
    assert gwh(release, "FR", "wind", 2023) == pytest.approx(51_457, abs=1)
    assert gwh(release, "NL", "solar", 2019) == pytest.approx(5_399, abs=1)


def test_real_download_covers_every_country_and_technology_from_2019_to_2024() -> None:
    production = eurostat.parse_gross_production(FIXTURE.read_bytes()).production

    counts = production.groupby(["country", "technology"])["year"].agg(["min", "max", "size"])

    assert len(counts) == len(eurostat.COUNTRIES) * len(eurostat.FUELS)
    assert (counts["min"] == 2019).all()
    assert (counts["max"] == 2024).all()
    assert (counts["size"] == 6).all()
    assert production["flag"].isna().all()


def test_updated_is_the_last_update_of_the_dataset() -> None:
    release = eurostat.parse_gross_production(FIXTURE.read_bytes())

    assert release.updated == pd.Timestamp("2026-06-02 23:00")


def test_observations_without_a_value_are_left_out() -> None:
    release = eurostat.parse_gross_production(
        download({"year": "2023"}, {"year": "2024", "value": ""})
    )

    assert release.production["year"].tolist() == [2023]


def test_flags_are_kept() -> None:
    release = eurostat.parse_gross_production(download({"flag": "p"}))

    assert release.production["flag"].tolist() == ["p"]


@pytest.mark.parametrize(("column", "value"), [("freq", "M"), ("nrg_bal", "NEP"), ("unit", "KTOE")])
def test_another_frequency_balance_or_unit_is_rejected(column: str, value: str) -> None:
    with pytest.raises(ValueError, match=column):
        eurostat.parse_gross_production(download({}, {column: value, "year": "2023"}))


def test_an_unknown_fuel_is_rejected() -> None:
    with pytest.raises(ValueError, match="RA100"):
        eurostat.parse_gross_production(download({"siec": "RA100"}))


def test_a_country_outside_the_plan_is_rejected() -> None:
    with pytest.raises(SchemaError):
        eurostat.parse_gross_production(download({"geo": "SE"}))


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
) -> eurostat.Release:
    transport = httpx.MockTransport(replay(responses, requests))
    with httpx.Client(transport=transport) as client:
        return eurostat.fetch_gross_production(client=client, cache_dir=cache_dir, refresh=refresh)


def test_fetch_asks_for_the_plan_countries_and_caches_the_raw_download(
    tmp_path: Path, sleeps: list[float]
) -> None:
    requests: list[httpx.Request] = []

    release = fetch([httpx.Response(200, content=FIXTURE.read_bytes())], requests, tmp_path)

    (request,) = requests
    assert request.url.path.endswith(
        "/nrg_bal_peh/A.GEP.RA420+RA300.GWH.AT+BE+DE+DK+ES+FR+IT+NL+PL+PT/"
    )
    assert dict(request.url.params) == {
        "format": "SDMX-CSV",
        "startPeriod": "2019",
        "compressed": "false",
    }
    assert (tmp_path / "nrg_bal_peh.csv").read_bytes() == FIXTURE.read_bytes()
    assert gwh(release, "DE", "solar", 2024) == pytest.approx(75_379, abs=1)
    assert sleeps == []


def test_cached_download_is_read_without_a_request_unless_refreshed(tmp_path: Path) -> None:
    (tmp_path / "nrg_bal_peh.csv").write_bytes(FIXTURE.read_bytes())
    requests: list[httpx.Request] = []

    cached = fetch([], requests, tmp_path)
    refreshed = fetch(
        [httpx.Response(200, content=FIXTURE.read_bytes())], requests, tmp_path, refresh=True
    )

    assert len(requests) == 1
    assert cached.production.equals(refreshed.production)
    assert cached.updated == refreshed.updated


def test_server_errors_and_timeouts_are_retried(tmp_path: Path, sleeps: list[float]) -> None:
    requests: list[httpx.Request] = []
    responses: list[httpx.Response | Exception] = [
        httpx.Response(503),
        httpx.ReadTimeout("timed out"),
        httpx.Response(200, content=FIXTURE.read_bytes()),
    ]

    fetch(responses, requests, tmp_path)

    assert len(requests) == 3
    assert sleeps == list(eurostat.RETRY_DELAYS_SECONDS)


def test_persistent_server_errors_raise_and_cache_nothing(
    tmp_path: Path, sleeps: list[float]
) -> None:
    attempts = len(eurostat.RETRY_DELAYS_SECONDS) + 1
    requests: list[httpx.Request] = []

    with pytest.raises(httpx.HTTPStatusError):
        fetch([httpx.Response(502)] * attempts, requests, tmp_path)

    assert len(requests) == attempts
    assert list(tmp_path.iterdir()) == []


def test_a_wrong_query_fails_at_once(tmp_path: Path, sleeps: list[float]) -> None:
    requests: list[httpx.Request] = []

    with pytest.raises(httpx.HTTPStatusError):
        fetch([httpx.Response(404)], requests, tmp_path)

    assert len(requests) == 1
    assert sleeps == []


def test_a_download_that_does_not_parse_is_not_cached(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="RA100"):
        fetch([httpx.Response(200, content=download({"siec": "RA100"}))], [], tmp_path)

    assert list(tmp_path.iterdir()) == []
