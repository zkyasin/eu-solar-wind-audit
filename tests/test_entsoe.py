from collections.abc import Callable
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import pytest

from eu_solar_wind_audit.sources import entsoe

GENERATION = "urn:iec62325.351:tc57wg16:451-6:generationloaddocument:3:0"
ACKNOWLEDGEMENT = "urn:iec62325.351:tc57wg16:451-1:acknowledgementdocument:7:0"
START = pd.Timestamp("2024-06-01 00:00", tz="UTC")
HOUR = pd.Timedelta(hours=1)


def period(
    powers: list[float],
    start: str,
    end: str,
    resolution: str = "PT15M",
    positions: list[int] | None = None,
) -> str:
    """A period with the power in MW at `positions`, by default 1, 2, ..."""
    listed = "".join(
        f"<Point><position>{position}</position><quantity>{power}</quantity></Point>"
        for position, power in zip(positions or range(1, len(powers) + 1), powers, strict=True)
    )
    return (
        f"<Period><timeInterval><start>{start}</start><end>{end}</end></timeInterval>"
        f"<resolution>{resolution}</resolution>{listed}</Period>"
    )


def time_series(
    *periods: str,
    curve: str = "A01",
    domain: str = "inBiddingZone_Domain.mRID",
    unit: str = "MAW",
) -> str:
    return (
        f"<TimeSeries><{domain}>10YNL----------L</{domain}>"
        f"<quantity_Measure_Unit.name>{unit}</quantity_Measure_Unit.name>"
        f"<curveType>{curve}</curveType>{''.join(periods)}</TimeSeries>"
    )


def document(*series: str) -> bytes:
    return (
        f'<?xml version="1.0" encoding="UTF-8"?><GL_MarketDocument xmlns="{GENERATION}">'
        f"<type>A75</type>{''.join(series)}</GL_MarketDocument>"
    ).encode()


def acknowledgement(code: str, text: str) -> bytes:
    return (
        f'<?xml version="1.0" encoding="UTF-8"?><Acknowledgement_MarketDocument '
        f'xmlns="{ACKNOWLEDGEMENT}"><Reason><code>{code}</code><text>{text}</text></Reason>'
        "</Acknowledgement_MarketDocument>"
    ).encode()


def serving(answer: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(answer))


def solar(cache_dir: Path, hours: int, *series: str) -> pd.DataFrame:
    """Dutch solar generation over the first `hours` of 2024-06-01, answered by `series`."""
    client = serving(lambda request: httpx.Response(200, content=document(*series)))
    return entsoe.fetch_hourly_generation(
        "NL", "solar", START, START + hours * HOUR, client=client, cache_dir=cache_dir
    )


def test_a_point_is_the_average_power_over_its_position(tmp_path: Path) -> None:
    quarters = [100.0, 100.0, 200.0, 200.0, 400.0, 400.0, 400.0, 400.0]

    hourly = solar(
        tmp_path, 2, time_series(period(quarters, "2024-06-01T00:00Z", "2024-06-01T02:00Z"))
    )

    assert hourly["energy_mwh"].tolist() == [150.0, 400.0]
    assert hourly["complete"].all()


def test_a_variable_sized_block_lasts_until_the_next_listed_position(tmp_path: Path) -> None:
    # Curve type A03 lists positions 1 and 3 only: 100 MW for half an hour, then 300 MW.
    blocks = period([100.0, 300.0], "2024-06-01T00:00Z", "2024-06-01T02:00Z", positions=[1, 3])

    hourly = solar(tmp_path, 2, time_series(blocks, curve="A03"))

    assert hourly["energy_mwh"].tolist() == [200.0, 300.0]


def test_half_hourly_and_hourly_positions(tmp_path: Path) -> None:
    hourly = solar(
        tmp_path,
        2,
        time_series(period([40.0, 80.0], "2024-06-01T00:00Z", "2024-06-01T01:00Z", "PT30M")),
        time_series(period([70.0], "2024-06-01T01:00Z", "2024-06-01T02:00Z", "PT60M")),
    )

    assert hourly["energy_mwh"].tolist() == [60.0, 70.0]


def test_the_consumption_series_of_a_generation_document_are_left_out(tmp_path: Path) -> None:
    bounds = ("2024-06-01T00:00Z", "2024-06-01T02:00Z", "PT60M")
    generation = time_series(period([100.0, 100.0], *bounds))
    pumping = time_series(period([999.0, 999.0], *bounds), domain="outBiddingZone_Domain.mRID")

    hourly = solar(tmp_path, 2, generation, pumping)

    assert hourly["energy_mwh"].tolist() == [100.0, 100.0]


def test_the_hours_between_two_periods_are_a_gap(tmp_path: Path) -> None:
    before = period([10.0], "2024-06-01T00:00Z", "2024-06-01T01:00Z", "PT60M")
    after = period([30.0], "2024-06-01T02:00Z", "2024-06-01T03:00Z", "PT60M")

    hourly = solar(tmp_path, 3, time_series(before, after, curve="A03"))

    assert hourly["complete"].tolist() == [True, False, True]
    assert np.isnan(hourly["energy_mwh"].iloc[1])


def test_a_query_without_data_gives_incomplete_hours(tmp_path: Path) -> None:
    no_data = acknowledgement("999", "No matching data found for Data item [16.1.B&amp;C]")
    client = serving(lambda request: httpx.Response(200, content=no_data))

    hourly = entsoe.fetch_hourly_generation(
        "NL", "solar", START, START + 3 * HOUR, client=client, cache_dir=tmp_path
    )

    assert len(hourly) == 3
    assert not hourly["complete"].any()


def test_a_refused_query_raises_its_reason_and_is_not_cached(tmp_path: Path) -> None:
    refused = acknowledgement("A41", "The query is not valid")
    client = serving(lambda request: httpx.Response(400, content=refused))

    with pytest.raises(entsoe.EntsoeError, match="A41: The query is not valid"):
        entsoe.fetch_hourly_load("NL", START, START + HOUR, client=client, cache_dir=tmp_path)
    assert not any(tmp_path.iterdir())


def test_quantities_must_be_in_megawatts() -> None:
    kilowatts = time_series(period([1.0], "2024-06-01T00:00Z", "2024-06-01T01:00Z"), unit="KWT")

    with pytest.raises(entsoe.EntsoeError, match="unit KWT"):
        entsoe.parse_energy(document(kilowatts), entsoe.GENERATION_DOMAIN)


def test_requests_cover_one_year_at_most_and_settled_answers_are_cached(tmp_path: Path) -> None:
    start = pd.Timestamp("2023-12-31 22:00", tz="UTC")
    end = pd.Timestamp("2024-01-01 02:00", tz="UTC")
    queries: list[httpx.QueryParams] = []

    def answer(request: httpx.Request) -> httpx.Response:
        queries.append(request.url.params)
        # The first answer runs past the end of its query, into the next one.
        first, last, hours = {
            "202312312200": ("2023-12-31T22:00Z", "2024-01-01T02:00Z", 4),
            "202401010000": ("2024-01-01T00:00Z", "2024-01-01T02:00Z", 2),
        }[request.url.params["periodStart"]]
        content = document(time_series(period([40.0] * hours, first, last, "PT60M")))
        return httpx.Response(200, content=content)

    for _ in range(2):
        hourly = entsoe.fetch_hourly_generation(
            "NL", "solar", start, end, client=serving(answer), cache_dir=tmp_path
        )

    assert [(query["periodStart"], query["periodEnd"]) for query in queries] == [
        ("202312312200", "202401010000"),
        ("202401010000", "202401010200"),
    ]
    assert hourly["energy_mwh"].tolist() == [40.0] * 4
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "A75_A16_B16_10YNL----------L_202312312200_202401010000.xml",
        "A75_A16_B16_10YNL----------L_202401010000_202401010200.xml",
    ]
    assert all("securityToken" not in query for query in queries)


def test_recent_answers_are_not_cached(tmp_path: Path) -> None:
    start = pd.Timestamp.now(tz="UTC").floor("h") - HOUR
    client = serving(lambda request: httpx.Response(200, content=acknowledgement("999", "")))

    entsoe.fetch_hourly_load("NL", start, start + HOUR, client=client, cache_dir=tmp_path)

    assert not any(tmp_path.iterdir())


def test_the_token_travels_in_a_header(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(entsoe.TOKEN_VARIABLE, "a-token")

    with entsoe.make_client() as client:
        assert client.headers["SECURITY_TOKEN"] == "a-token"


def test_a_missing_token_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(entsoe.TOKEN_VARIABLE, raising=False)

    with pytest.raises(entsoe.EntsoeError, match=entsoe.TOKEN_VARIABLE):
        entsoe.make_client()
