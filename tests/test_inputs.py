from pathlib import Path

import pandas as pd

from eu_solar_wind_audit.anomalies import inputs
from eu_solar_wind_audit.sources import odre, sdes

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
