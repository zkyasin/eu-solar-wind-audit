import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from pandera.errors import SchemaError

from eu_solar_wind_audit.sources import energy_charts

# Fixture: https://api.energy-charts.info/v2/installed_power?country=de&time_step=monthly&installation_decommission=false,
# retrieved 2026-11-22.
FIXTURE = Path(__file__).parent / "fixtures" / "energy_charts" / "installed_power_de_monthly.json"


@pytest.fixture
def german_payload() -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(FIXTURE.read_bytes())
    return payload


def payload(
    data: list[dict[str, Any]],
    *,
    units: dict[str, str | None],
    generated_at: str = "2026-11-22T14:57:13+01:00",
) -> dict[str, Any]:
    """A /v2/installed_power response with the envelope unit set to GW."""
    return {
        "timezone": "Europe/Berlin",
        "unit": "GW",
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


def test_negative_capacity_is_rejected() -> None:
    data = [{"timestamp": "2025-01-01T00:00:00+01:00", "values": {"wind_onshore": -0.1}}]

    with pytest.raises(SchemaError, match="capacity_mw"):
        energy_charts.parse_installed_power(payload(data, units={"wind_onshore": None}))
