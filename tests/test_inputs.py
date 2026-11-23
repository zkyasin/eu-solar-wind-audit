from pathlib import Path

import httpx
import pandas as pd
import pytest

from eu_solar_wind_audit import eligibility, grid, physics
from eu_solar_wind_audit.anomalies import inputs
from eu_solar_wind_audit.config import CONFIG_DIR, load_country
from eu_solar_wind_audit.sources import energy_charts, odre, sdes, smard

# The SDES wind dashboard of 2026-11-22 cut to Bretagne, Normandie, Corsica and the overseas
# departments (tests/test_sdes.py).
SDES_WIND = Path(__file__).parent / "fixtures" / "sdes" / "eolien-2026-t2.xlsx"
FIRST_POWER = odre.FIRST_OFFSHORE_POWER


def utc(stamp: str) -> pd.Timestamp:
    return pd.Timestamp(stamp, tz="UTC")


def test_offshore_output_starts_with_the_first_farm_of_each_area() -> None:
    starts = inputs.offshore_starts(sdes.read_capacity(SDES_WIND))

    assert starts == {
        "FR": FIRST_POWER,
        "FR-52": FIRST_POWER,
        # 497 MW in Normandie from the third quarter of 2023, 496 MW in Bretagne from the second:
        # from 1 July and 1 April, midnight in Paris.
        "FR-28": utc("2023-06-30 22:00"),
        "FR-53": utc("2023-03-31 22:00"),
    }


def offshore_rows(code: str, region: str, capacity_mw: list[float]) -> list[dict[str, object]]:
    """Quarterly panel rows of one department, offshore and onshore, from the third quarter of
    2023 on, with `capacity_mw` offshore and 1000 MW onshore."""
    quarter_ends = pd.date_range("2023-09-30", periods=len(capacity_mw), freq="QE")
    return [
        {
            "department_code": code,
            "region": region,
            "technology": technology,
            "quarter_end": quarter_end,
            "capacity_mw": value if technology == "wind_offshore" else 1000.0,
        }
        for quarter_end, value in zip(quarter_ends, capacity_mw, strict=True)
        for technology in ("wind_offshore", "wind_onshore")
    ]


def test_a_region_counts_from_its_first_quarter_above_10_mw() -> None:
    panel = pd.DataFrame(
        [
            # Aude: 10 MW, then 10.5 MW from the first quarter of 2024, in winter time.
            *offshore_rows("11", "Occitanie", [10.0, 10.0, 10.5]),
            # Two departments of 6 MW: 12 MW in Normandie from the fourth quarter of 2023.
            *offshore_rows("14", "Normandie", [0.0, 6.0, 6.0]),
            *offshore_rows("76", "Normandie", [0.0, 6.0, 6.0]),
            # Corsica is not in the éCO2mix series.
            *offshore_rows("2A", "Corse", [50.0, 50.0, 50.0]),
            # Pays de la Loire starts with its first farm, whatever the panel says.
            *offshore_rows("44", "Pays de la Loire", [0.0, 0.0, 500.0]),
        ]
    )

    starts = inputs.offshore_starts(panel)

    assert starts == {
        "FR": FIRST_POWER,
        "FR-52": FIRST_POWER,
        "FR-28": utc("2023-09-30 22:00"),
        "FR-76": utc("2023-12-31 23:00"),
    }


ODRE = Path(__file__).parent / "fixtures" / "odre"
# The switch from consolidated to real-time values on 31 August and 1 September 2026, as in
# tests/test_odre_hourly.py.
SWITCH = {
    odre.NATIONAL_CONSOLIDATED: ODRE / "eco2mix-national-cons-def_2026-08-31_2026-09-01.parquet",
    odre.NATIONAL_REALTIME: ODRE / "eco2mix-national-tr_2026-08-31_2026-09-01.parquet",
    odre.REGIONAL_CONSOLIDATED: ODRE / "eco2mix-regional-cons-def_2026-08-31_2026-09-01.parquet",
}
ENERGY_CHARTS = Path(__file__).parent / "fixtures" / "energy_charts"


def hours_of(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    return pd.date_range(start, end, freq="h", inclusive="left", unit="ns", name="time")


def reference_classes(start: pd.Timestamp, end: pd.Timestamp, **_: object) -> pd.DataFrame:
    """Price classes as eligibility.price_classes gives them, every hour of reference class."""
    return pd.DataFrame(
        {
            "mean_price": 50.0,
            "has_negative_quarter": False,
            "near_zero_negative": False,
            "regime": "reference",
        },
        index=hours_of(start, end),
    )


def test_france_inputs_give_each_series_its_datasets_model_and_offshore_hours(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def modelled_regions(
        model: str, first: pd.Timestamp, last: pd.Timestamp, *_: object
    ) -> dict[str, pd.DataFrame]:
        frame = pd.DataFrame(
            {
                "energy_mwh": 1.0,
                "complete": True,
                "capacity_mw": 10.0,
                "capacity_provisional": False,
                "era5_source": "ERA5",
            },
            index=hours_of(first, last),
        )
        return {code: physics.Modelled.validate(frame) for code in odre.REGIONS}

    def capacity(
        monthly: pd.DataFrame, series: str, start: pd.Timestamp, end: pd.Timestamp, **_: object
    ) -> pd.Series:
        return pd.Series(100.0, index=hours_of(start, end))

    monkeypatch.setattr(physics, "model_france_regions", modelled_regions)
    monkeypatch.setattr(physics, "load_monthly_capacity", lambda *_: pd.DataFrame())
    monkeypatch.setattr(sdes, "hourly_capacity", capacity)
    monkeypatch.setattr(sdes, "download_dashboard", lambda *_: SDES_WIND)
    monkeypatch.setattr(eligibility, "price_classes", reference_classes)

    def export(request: httpx.Request) -> httpx.Response:
        dataset = request.url.path.split("/")[-3]
        return httpx.Response(200, content=SWITCH[dataset].read_bytes())

    start, end = utc("2026-08-31 00:00"), utc("2026-09-02 00:00")
    with httpx.Client(transport=httpx.MockTransport(export)) as client:
        found = inputs.france_inputs(
            client, start, end, load_country(CONFIG_DIR / "fr.toml"), odre_cache_dir=tmp_path
        )

    technologies = ("solar", "wind", "wind_onshore", "wind_offshore")
    assert set(found.series) == {
        f"odre.{area}.{technology}" for area in inputs.AREAS for technology in technologies
    }
    assert found.hours.equals(hours_of(start - inputs.MARGIN, end + inputs.MARGIN))
    # A national series: consolidated rows, then real-time rows from the hour they stop.
    consolidated, realtime = found.series["odre.FR.solar"].published
    assert consolidated.provenance == "RTE, éCO2mix, ODRÉ dataset eco2mix-national-cons-def"
    assert realtime.provenance == "RTE, éCO2mix, ODRÉ dataset eco2mix-national-tr"
    assert (consolidated.provisional, realtime.provisional) == (False, True)
    assert consolidated.hourly.index[-1] + grid.HOUR == realtime.hourly.index[0]
    assert set(realtime.hourly["status"].dropna()) == {"realtime"}
    # National onshore wind is the sum of the regions until real time.
    onshore = found.series["odre.FR.wind_onshore"].published[0]
    assert onshore.provenance.endswith(odre.REGIONAL_CONSOLIDATED)
    # A regional series has consolidated and definitive rows only.
    (regional,) = found.series["odre.FR-52.wind"].published
    assert not regional.provisional
    # Mainland France is modelled as the sum of its 12 regions; offshore wind has no model.
    national_model = found.series["odre.FR.solar"].modelled
    assert national_model is not None
    assert (national_model["energy_mwh"] == 12.0).all()
    assert found.series["odre.FR.wind_offshore"].modelled is None
    # Every hour of 2026 holds offshore output in mainland and Bretagne wind, none in Île-de-France
    # wind, onshore wind or solar.
    for series, with_offshore in [
        ("odre.FR.wind", True),
        ("odre.FR-53.wind", True),
        ("odre.FR-11.wind", False),
        ("odre.FR.wind_onshore", False),
        ("odre.FR.solar", False),
    ]:
        offshore = found.series[series].offshore
        assert offshore is not None
        assert offshore.index.equals(found.hours)
        assert (offshore == with_offshore).all(), series
    assert [(rows.dataset, rows.provisional) for rows in found.rows] == [
        ("odre.eco2mix-national-cons-def", False),
        ("odre.eco2mix-regional-cons-def", False),
        ("odre.eco2mix-national-tr", True),
    ]
    assert "mean_price" not in found.price_classes.columns


def test_germany_inputs_give_smard_and_energy_charts_series_without_a_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    start, end = utc("2025-06-01 00:00"), utc("2025-06-02 00:00")
    first, last = start - inputs.MARGIN, end + inputs.MARGIN
    quarter_hours = pd.date_range(first, last, freq="15min", inclusive="left")

    def feed_in(request: httpx.Request) -> httpx.Response:
        # One weekly file from the first hour on, with 2.5 MWh in every quarter-hour.
        first_ms = int(first.timestamp() * 1000)
        if request.url.path.endswith("index_quarterhour.json"):
            return httpx.Response(200, json={"timestamps": [first_ms]})
        series = [[int(stamp.timestamp() * 1000), 2.5] for stamp in quarter_hours]
        return httpx.Response(200, json={"meta_data": {}, "series": series})

    def power_month(endpoint: str, month: pd.Period, *_: object) -> pd.DataFrame:
        local = pd.date_range(
            month.start_time, month.end_time, freq="15min", tz=energy_charts.TIMEZONE
        )
        stamps = pd.DatetimeIndex(local.tz_convert("UTC"), name="time")
        return pd.DataFrame({"solar": 100.0, "wind_onshore": 200.0, "wind_offshore": 50.0}, stamps)

    cache = tmp_path / "energy_charts"
    cache.mkdir()
    installed = "installed_power_de_monthly.json"
    (cache / installed).write_bytes((ENERGY_CHARTS / installed).read_bytes())
    monkeypatch.setattr(energy_charts, "fetch_power_month", power_month)
    monkeypatch.setattr(eligibility, "price_classes", reference_classes)

    with httpx.Client(transport=httpx.MockTransport(feed_in)) as client:
        found = inputs.germany_inputs(
            client, start, end, smard_cache_dir=tmp_path / "smard", energy_charts_cache_dir=cache
        )

    smard_series = {
        f"smard.{area}.{technology}"
        for area in inputs.SMARD_AREAS
        for technology in ("solar", "wind_onshore")
    } | {f"smard.{area}.wind_offshore" for area in smard.OFFSHORE_REGIONS}
    energy_charts_series = {
        f"energy_charts.DE.{endpoint}.{technology}"
        for endpoint in ("public_power", "total_power")
        for technology in energy_charts.POWER_SERIES
    }
    assert set(found.series) == smard_series | energy_charts_series
    assert all(item.modelled is None and item.offshore is None for item in found.series.values())
    # Capacity is published for Germany as a whole only.
    assert found.series["smard.DE.solar"].capacity_mw is not None
    assert found.series["smard.Amprion.solar"].capacity_mw is None
    # Quarter-hour energies of 2.5 MWh, and a mean power of 200 MW, over whole hours.
    (feed_in_part,) = found.series["smard.DE.solar"].published
    assert feed_in_part.provenance == "Bundesnetzagentur | SMARD.de, filter 4068, region DE"
    assert (feed_in_part.hourly["energy_mwh"] == 10.0).all()
    (power_part,) = found.series["energy_charts.DE.public_power.wind_onshore"].published
    assert (power_part.hourly["energy_mwh"] == 200.0).all()
    assert [rows.dataset for rows in found.rows] == [
        "smard.4068",
        "smard.4067",
        "smard.1225",
        "energy_charts.public_power",
        "energy_charts.total_power",
    ]
