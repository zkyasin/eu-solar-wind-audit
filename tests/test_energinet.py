import json
import time
from collections.abc import Callable
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import pytest

from eu_solar_wind_audit.sources import energinet

START = pd.Timestamp("2024-06-01 10:00", tz="UTC")
HOUR = pd.Timedelta(hours=1)


def record(hour: str, area: str, **values: float | None) -> dict[str, object]:
    """A settlement record of `hour` (UTC) in `area`: every column of the quantities is 1 MWh
    unless given."""
    row: dict[str, object] = {"HourUTC": hour, "PriceArea": area}
    row.update(dict.fromkeys(energinet.ENERGY_COLUMNS, 1.0))
    row.update(values)
    return row


def serving(*payloads: dict[str, object]) -> tuple[httpx.Client, list[httpx.Request]]:
    """A client answering each request with the next payload, and the requests it received."""
    requests: list[httpx.Request] = []

    def answer(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, content=json.dumps(payloads[len(requests) - 1]).encode())

    return httpx.Client(transport=httpx.MockTransport(answer)), requests


def settlement(
    tmp_path: Path, area: energinet.Area, quantity: energinet.Quantity, *records: dict[str, object]
) -> pd.DataFrame:
    """`quantity` in `area` over two hours from START, answered by `records`."""
    client, _ = serving({"records": list(records)})
    return energinet.fetch_hourly_settlement(
        area, quantity, START, START + 2 * HOUR, client=client, cache_dir=tmp_path
    )


def test_solar_is_the_sum_of_the_three_plant_sizes(tmp_path: Path) -> None:
    hourly = settlement(
        tmp_path,
        "DK1",
        "solar",
        record("2024-06-01T10:00:00", "DK1", SolarPowerGe40kW_MWh=500.0),
        record("2024-06-01T11:00:00", "DK1", SolarPowerGe40kW_MWh=600.0),
        record("2024-06-01T10:00:00", "DK2", SolarPowerGe40kW_MWh=900.0),
    )

    assert hourly["energy_mwh"].tolist() == [502.0, 602.0]
    assert hourly["complete"].all()


def test_total_solar_adds_the_self_consumed_output(tmp_path: Path) -> None:
    hourly = settlement(
        tmp_path,
        "DK2",
        "solar_total",
        record("2024-06-01T10:00:00", "DK2", SolarPowerGe40kW_MWh=500.0, SolarPowerSelfConMWh=80.0),
        record("2024-06-01T11:00:00", "DK2", SolarPowerSelfConMWh=None),
    )

    assert hourly["energy_mwh"].iloc[0] == 582.0
    assert hourly["complete"].tolist() == [True, False]


def test_denmark_is_complete_when_both_areas_are(tmp_path: Path) -> None:
    hourly = settlement(
        tmp_path,
        "DK",
        "wind_offshore",
        record("2024-06-01T10:00:00", "DK1"),
        record("2024-06-01T10:00:00", "DK2"),
        record("2024-06-01T11:00:00", "DK1"),
    )

    assert hourly["energy_mwh"].iloc[0] == 4.0
    assert hourly["complete"].tolist() == [True, False]


def test_an_hour_is_incomplete_when_a_column_is_missing_and_negatives_are_kept(
    tmp_path: Path,
) -> None:
    hourly = settlement(
        tmp_path,
        "DK2",
        "wind_onshore",
        record("2024-06-01T10:00:00", "DK2", OnshoreWindLt50kW_MWh=None),
        record("2024-06-01T11:00:00", "DK2", OnshoreWindGe50kW_MWh=-3.0),
    )

    assert hourly["complete"].tolist() == [False, True]
    assert np.isnan(hourly["energy_mwh"].iloc[0])
    assert hourly["energy_mwh"].iloc[1] == -2.0


def test_requests_take_local_years_and_settled_years_are_cached(tmp_path: Path) -> None:
    # New Year 2024 in Copenhagen is 23:00 UTC on 31 December 2023.
    start = pd.Timestamp("2023-12-31 22:00", tz="UTC")
    end = pd.Timestamp("2024-01-01 00:00", tz="UTC")
    client, requests = serving(
        {"records": [record("2023-12-31T22:00:00", "DK1")]},
        {"records": [record("2023-12-31T23:00:00", "DK1")]},
    )

    for _ in range(2):
        hourly = energinet.fetch_hourly_settlement(
            "DK1", "gross_consumption", start, end, client=client, cache_dir=tmp_path
        )

    assert [dict(request.url.params) for request in requests] == [
        {"start": "2023-01-01", "end": "2024-01-01", "limit": "0"},
        {"start": "2024-01-01", "end": "2025-01-01", "limit": "0"},
    ]
    assert hourly["energy_mwh"].tolist() == [1.0, 1.0]
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "ProductionConsumptionSettlement_2023.json",
        "ProductionConsumptionSettlement_2024.json",
    ]


def test_the_current_year_is_not_cached(tmp_path: Path) -> None:
    start = pd.Timestamp.now(tz="UTC").floor("h") - HOUR
    client, _ = serving({"records": []})

    energinet.fetch_hourly_settlement(
        "DK1", "solar", start, start + HOUR, client=client, cache_dir=tmp_path
    )

    assert not any(tmp_path.iterdir())


def test_a_year_still_corrected_is_downloaded_once_per_process(tmp_path: Path) -> None:
    start = pd.Timestamp.now(tz="UTC").floor("h") - HOUR
    hour = start.strftime("%Y-%m-%dT%H:%M:%S")
    client, requests = serving({"records": [record(hour, "DK1"), record(hour, "DK2")]})

    for quantity in ("solar", "wind_onshore"):
        hourly = energinet.fetch_hourly_settlement(
            "DK", quantity, start, start + HOUR, client=client, cache_dir=tmp_path
        )
        assert hourly["energy_mwh"].tolist() == [len(energinet.QUANTITIES[quantity]) * 2.0]

    assert len(requests) == 1


def test_a_rate_limit_waits_the_time_the_api_asks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    waits: list[float] = []
    monkeypatch.setattr(time, "sleep", waits.append)
    answers: list[Callable[[], httpx.Response]] = [
        lambda: httpx.Response(
            429, json={"message": "Rate limit is exceeded. Try again in 42 seconds."}
        ),
        lambda: httpx.Response(200, json={"records": []}),
    ]
    client = httpx.Client(transport=httpx.MockTransport(lambda request: answers.pop(0)()))

    energinet.fetch_hourly_settlement(
        "DK1", "solar", START, START + HOUR, client=client, cache_dir=tmp_path
    )

    assert waits == [42.0]


def test_capacity_per_municipality_and_its_price_area() -> None:
    payload = {
        "records": [
            {
                "Month": "2025-12-01T00:00:00",
                "MunicipalityNo": number,
                "SolarPowerCapacity": 10.0,
                "OnshoreWindCapacity": 5.0,
                "OffshoreWindCapacity": None,
            }
            for number in ("101", "400", "410", "1")
        ]
    }

    capacity = energinet.parse_capacity(payload)

    assert capacity["municipality"].tolist() == [1, 101, 400, 410]
    assert capacity["price_area"].tolist()[1:] == ["DK2", "DK2", "DK1"]
    assert capacity["price_area"].isna().iloc[0]
    assert (capacity["month_start"] == pd.Timestamp("2025-12-01")).all()
    assert capacity["wind_offshore_mw"].isna().all()


def test_price_area_of_municipality_numbers() -> None:
    numbers = pd.Series([101, 400, 410, 860, 1, None], dtype="Int64")

    areas = energinet.price_area(numbers)

    assert areas.tolist()[:4] == ["DK2", "DK2", "DK1", "DK1"]
    assert areas.iloc[4:].isna().all()


def test_settlement_values_keep_each_record_of_one_area(tmp_path: Path) -> None:
    client, _ = serving(
        {
            "records": [
                record("2024-06-01T11:00:00", "DK1", OnshoreWindGe50kW_MWh=300.0),
                record("2024-06-01T10:00:00", "DK1", OnshoreWindLt50kW_MWh=None),
                record("2024-06-01T10:00:00", "DK1"),
                record("2024-06-01T10:00:00", "DK2"),
                record("2024-06-01T12:00:00", "DK1"),
            ]
        }
    )

    values = energinet.fetch_settlement_values(
        "DK1", "wind_onshore", START, START + 2 * HOUR, client=client, cache_dir=tmp_path
    )

    # The repeated hour keeps both records, in the order they came; the hour after the period goes.
    assert values.index.tolist() == [START, START, START + HOUR]
    assert np.isnan(values.iloc[0])
    assert values.iloc[1:].tolist() == [2.0, 301.0]


def capacity_of(month_starts: list[str], solar: list[float]) -> pd.DataFrame:
    return energinet.parse_capacity(
        {
            "records": [
                {
                    "Month": f"{month}T00:00:00",
                    "MunicipalityNo": number,
                    "SolarPowerCapacity": mw / 2,
                    "OnshoreWindCapacity": 1.0,
                    "OffshoreWindCapacity": 0.0,
                }
                for month, mw in zip(month_starts, solar, strict=True)
                for number in ("730", "751", "101")
            ]
        }
    )


def test_capacity_is_linear_between_the_first_days_of_months_in_danish_time() -> None:
    capacity = capacity_of(["2024-06-01", "2024-07-01"], [100.0, 130.0])
    # Midnight on 1 June 2024 in Copenhagen is 22:00 UTC on 31 May.
    start = pd.Timestamp("2024-05-31 22:00", tz="UTC")

    hourly = energinet.hourly_capacity(capacity, "DK1", "solar", start, start + 720 * HOUR)

    # Two municipalities of DK1 hold half of the figure each; Copenhagen is in DK2.
    assert hourly.iloc[0] == pytest.approx(100.0)
    assert hourly.iloc[360] == pytest.approx(115.0)
    assert (
        energinet.hourly_capacity(capacity, "DK2", "wind_onshore", start, start + HOUR).iloc[0]
        == 1.0
    )
    with pytest.raises(ValueError, match="no solar capacity of DK1"):
        energinet.hourly_capacity(capacity, "DK1", "solar", start - HOUR, start)
