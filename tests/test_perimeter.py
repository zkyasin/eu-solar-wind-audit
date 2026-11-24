import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandera.typing import DataFrame

from eu_solar_wind_audit import cli, perimeter
from eu_solar_wind_audit.detectability import Z_SUM
from eu_solar_wind_audit.sources import energy_charts, eurostat, mastr, odre, sdes, smard

FIXTURES = Path(__file__).parent / "fixtures"


def hourly(start: str, end: str, energy_mwh: float) -> pd.DataFrame:
    hours = pd.date_range(start, end, freq="h", inclusive="left", tz="UTC", name="time")
    return pd.DataFrame({"energy_mwh": energy_mwh, "complete": True}, index=hours)


def test_annual_energy_sums_paris_calendar_years_in_gwh() -> None:
    # The last UTC hour of 2019 is the first hour of 2020 in Paris.
    frame = hourly("2019-12-31 22:00", "2020-01-01 01:00", 1000.0)

    energy = perimeter.annual_energy(frame)

    assert energy.to_dict() == {2019: 1.0, 2020: 2.0}


def test_annual_energy_leaves_out_incomplete_hours() -> None:
    frame = hourly("2020-06-01", "2020-06-01 03:00", 1000.0)
    frame.loc[frame.index[1], ["energy_mwh", "complete"]] = [np.nan, False]

    assert perimeter.annual_energy(frame).to_dict() == {2020: 2.0}


def test_year_bounds_are_the_utc_hours_of_paris_new_years() -> None:
    start, end = perimeter.year_bounds(2019, 2024)

    assert start == pd.Timestamp("2018-12-31 23:00", tz="UTC")
    assert end == pd.Timestamp("2024-12-31 23:00", tz="UTC")


def sdes_rows(code: str, capacities: dict[str, float]) -> list[dict[str, object]]:
    return [
        {"department_code": code, "quarter_end": pd.Timestamp(day), "capacity_mw": capacity}
        for day, capacity in capacities.items()
    ]


def test_inside_shares_leave_out_corsica_and_overseas_departments() -> None:
    quarters = ["2024-06-30", "2024-12-31"]
    panel = pd.DataFrame(
        [
            *sdes_rows("75", {"2024-06-30": 80.0, "2024-12-31": 90.0}),
            *sdes_rows("2A", {"2024-06-30": 5.0, "2024-12-31": 5.0}),
            *sdes_rows("974", {"2024-06-30": 15.0, "2024-12-31": 5.0}),
        ]
    )
    totals = pd.DataFrame(
        [
            {"total": "Total France", "quarter_end": pd.Timestamp(day), "capacity_mw": 100.0}
            for day in quarters
        ]
        + [
            {"total": "Total métropole", "quarter_end": pd.Timestamp(day), "capacity_mw": 0.0}
            for day in quarters
        ]
    )

    shares = perimeter.inside_shares(panel, totals)

    assert shares.to_dict() == {2024: pytest.approx(0.85)}


def inputs(
    solar_gwh: float, wind_gwh: float, share: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, "pd.Series[float]"]:
    """One year of one German series against a gross production of 100 GWh of each technology,
    with 85 % of wind capacity on the distribution grids."""
    published = pd.DataFrame(
        {"series": ["smard"], "year": [2024], "solar": [solar_gwh], "wind": [wind_gwh]}
    )
    gross = pd.DataFrame({"country": ["DE"], "year": [2024], "solar": [100.0], "wind": [100.0]})
    inside = pd.DataFrame({"series": ["smard"], "year": [2024], "solar": [1.0], "wind": [1.0]})
    self_consumption = pd.DataFrame({"country": ["DE"], "year": [2024], "share": [share]})
    distribution = pd.Series({"DE": 0.85})
    return published, gross, inside, self_consumption, distribution


def test_the_gap_is_the_log_of_the_solar_ratio_over_the_wind_ratio() -> None:
    table = perimeter.levels(*inputs(solar_gwh=80.0, wind_gwh=95.0, share=0.2))

    (row,) = table.to_dict(orient="records")
    assert row["solar_ratio"] == pytest.approx(0.8)
    assert row["wind_ratio"] == pytest.approx(0.95)
    assert row["gap"] == pytest.approx(np.log(0.8 / 0.95))
    assert row["gap_if_excluded"] == pytest.approx(np.log(0.8))
    assert row["log_wind_ratio_if_not_covered"] == pytest.approx(np.log(0.15))


def test_gross_production_is_cut_to_the_capacity_inside_the_series() -> None:
    published, gross, inside, self_consumption, distribution = inputs(
        solar_gwh=90.0, wind_gwh=99.0, share=0.1
    )
    inside["solar"] = 0.9

    table = perimeter.levels(published, gross, inside, self_consumption, distribution)

    (row,) = table.to_dict(orient="records")

    assert row["solar_ratio"] == pytest.approx(1.0)


def levels_of(rows: list[dict[str, object]]) -> DataFrame[perimeter.Levels]:
    """Levels with only the columns the decisions read, the others at neutral values."""
    frame = pd.DataFrame(rows)
    frame["gap_if_excluded"] = np.log1p(-frame["self_consumption_share"])
    if "distribution_share" not in frame:
        frame["distribution_share"] = 0.85
    frame["log_wind_ratio_if_not_covered"] = np.log1p(-frame["distribution_share"])
    for column in [
        "published_solar_gwh",
        "published_wind_gwh",
        "gross_solar_gwh",
        "gross_wind_gwh",
    ]:
        frame[column] = 100.0
    for column in ["inside_solar", "inside_wind", "solar_ratio"]:
        frame[column] = 1.0
    if "wind_ratio" not in frame:
        frame["wind_ratio"] = 1.0
    return perimeter.Levels.validate(frame[list(perimeter.Levels.to_schema().columns)])


def test_errors_are_the_gap_minus_that_of_the_known_content() -> None:
    table = levels_of(
        [
            {"series": "eco2mix", "year": 2024, "gap": 0.02, "self_consumption_share": 0.07},
            {"series": "smard", "year": 2024, "gap": -0.2, "self_consumption_share": 0.2},
            # Its errors are those of SMARD, so they do not count.
            {
                "series": "energy_charts_total",
                "year": 2024,
                "gap": 0.5,
                "self_consumption_share": 0.2,
            },
        ]
    )

    errors = perimeter.gap_errors(table)

    assert errors.to_dict() == {
        ("eco2mix", 2024): pytest.approx(0.02),
        ("smard", 2024): pytest.approx(-0.2 - np.log(0.8)),
    }


def test_the_error_is_the_root_mean_square() -> None:
    assert perimeter.error_rms(pd.Series([0.03, -0.04])) == pytest.approx(np.sqrt(0.00125))


def test_the_threshold_is_the_share_detected_with_80_percent_power() -> None:
    threshold = perimeter.detection_threshold(0.02)

    assert -np.log1p(-threshold) == pytest.approx(Z_SUM * 0.02)
    assert threshold == pytest.approx(0.0485, abs=1e-4)


def test_years_below_the_threshold_are_inconclusive_and_others_take_the_closer_content() -> None:
    table = levels_of(
        [
            {"series": "smard", "year": 2021, "gap": -0.2, "self_consumption_share": 0.04},
            {"series": "smard", "year": 2022, "gap": -0.06, "self_consumption_share": 0.1},
            {"series": "smard", "year": 2023, "gap": -0.04, "self_consumption_share": 0.1},
        ]
    )

    assert perimeter.decide_self_consumption(table, threshold=0.05).tolist() == [
        "inconclusive",
        "excluded",
        "included",
    ]


def test_the_wind_error_is_the_log_wind_ratio_of_series_that_cover_the_grids() -> None:
    table = levels_of(
        [
            {
                "series": "eco2mix",
                "year": 2024,
                "gap": 0.0,
                "self_consumption_share": 0.07,
                "wind_ratio": 0.99,
            },
            {
                "series": "smard",
                "year": 2024,
                "gap": 0.0,
                "self_consumption_share": 0.2,
                "wind_ratio": 1.01,
            },
        ]
    )

    errors = perimeter.wind_errors(table)

    assert errors.to_dict() == {
        ("eco2mix", 2024): pytest.approx(np.log(0.99)),
        ("smard", 2024): pytest.approx(np.log(1.01)),
    }


def test_grids_are_covered_when_wind_is_closer_to_eurostat_than_to_its_distribution_share() -> None:
    table = levels_of(
        [
            {
                "series": "smard",
                "year": 2022,
                "gap": 0.0,
                "self_consumption_share": 0.1,
                "wind_ratio": 0.98,
                "distribution_share": 0.85,
            },
            {
                "series": "smard",
                "year": 2023,
                "gap": 0.0,
                "self_consumption_share": 0.1,
                "wind_ratio": 0.2,
                "distribution_share": 0.85,
            },
            {
                "series": "smard",
                "year": 2024,
                "gap": 0.0,
                "self_consumption_share": 0.1,
                "wind_ratio": 0.2,
                "distribution_share": 0.03,
            },
        ]
    )

    assert perimeter.decide_distribution(table, threshold=0.04).tolist() == [
        "covered",
        "not_covered",
        "inconclusive",
    ]


@pytest.mark.parametrize(
    ("decisions", "expected"),
    [
        (["inconclusive", "excluded", "excluded"], "excluded"),
        (["included", "inconclusive"], "included"),
        (["included", "excluded"], "ambiguous"),
        (["inconclusive", "inconclusive"], "inconclusive"),
    ],
)
def test_a_series_takes_the_content_its_decided_years_agree_on(
    decisions: list[str], expected: str
) -> None:
    assert perimeter.verdict(pd.Series(decisions)) == expected


def test_the_command_writes_the_levels_and_the_content_of_each_series(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def constant(start: pd.Timestamp, end: pd.Timestamp, energy_mwh: float) -> pd.DataFrame:
        return hourly(str(start.tz_convert(None)), str(end.tz_convert(None)), energy_mwh)

    def gross(**_: object) -> eurostat.Release:
        return eurostat.parse_gross_production(
            (FIXTURES / "eurostat" / "nrg_bal_peh.csv").read_bytes()
        )

    def national_hourly(
        client: object, series: str, start: pd.Timestamp, end: pd.Timestamp, **_: object
    ) -> pd.DataFrame:
        return constant(start, end, 2000.0)

    def download_dashboard(client: object, media_id: int, **_: object) -> Path:
        name = (
            "solaire-pv-2026-t2.xlsx" if media_id == sdes.SOLAR_2026_Q2 else "eolien-2026-t2.xlsx"
        )
        return FIXTURES / "sdes" / name

    def register(client: object, **_: object) -> pd.DataFrame:
        cut = FIXTURES / "odre" / f"{odre.REGISTER}_cut.parquet"
        return pd.read_parquet(cut).rename(columns=odre.REGISTER_FIELDS)

    def feed_in(
        region: str, technology: str, start: pd.Timestamp, end: pd.Timestamp, **_: object
    ) -> pd.DataFrame:
        return constant(start, end, 5000.0)

    def power(start: pd.Timestamp, end: pd.Timestamp, **_: object) -> dict[str, pd.DataFrame]:
        series = {
            f"{perimeter_name}.{technology}": constant(start, end, 5000.0)
            for perimeter_name in ["public", "total"]
            for technology in ["solar", "wind_onshore", "wind_offshore"]
        }
        series["self_consumption.solar"] = constant(start, end, 500.0)
        return series

    units_dir = tmp_path / "mastr"
    units_dir.mkdir()
    mastr.read_units(FIXTURES / "mastr" / "Gesamtdatenexport_20261001_26.1.zip").to_parquet(
        units_dir / "mastr_units.parquet"
    )
    (units_dir / "mastr_units.json").write_text(json.dumps({"export_date": "2026-10-01"}))
    monkeypatch.setattr(mastr, "OUTPUT_DIR", units_dir)
    monkeypatch.setattr(eurostat, "fetch_gross_production", gross)
    monkeypatch.setattr(odre, "national_hourly", national_hourly)
    monkeypatch.setattr(odre, "fetch_register", register)
    monkeypatch.setattr(sdes, "download_dashboard", download_dashboard)
    monkeypatch.setattr(smard, "fetch_hourly_feed_in", feed_in)
    monkeypatch.setattr(energy_charts, "fetch_hourly_power", power)

    cli.main(["audit-perimeter", str(tmp_path / "out")])

    table = pd.read_parquet(tmp_path / "out" / "perimeter-levels.parquet")
    summary = json.loads((tmp_path / "out" / "perimeter.json").read_text())
    assert set(table["series"]) == set(perimeter.SERIES)
    assert set(table["year"]) == set(range(2019, 2025))
    assert set(table["self_consumption"]) <= {"included", "excluded", "inconclusive"}
    assert set(table["distribution"]) <= {"covered", "not_covered", "inconclusive"}
    assert set(summary["content"]) == set(perimeter.SERIES)
    assert set(summary["errors"]) == set(perimeter.DIMENSIONS)
    assert all(len(errors) == 12 for errors in summary["errors"].values())
    assert all(0 < value < 1 for value in summary["detection_threshold"].values())
    # Wind of the register cut: 78,800 kW of 100,550 on the grid of RTE.
    assert summary["distribution_share"]["values"]["FR"] == pytest.approx(21_750 / 100_550)
    assert summary["eurostat"]["updated"] == "2026-06-02T23:00:00"
