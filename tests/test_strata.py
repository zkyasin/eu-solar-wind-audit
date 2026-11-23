import numpy as np
import pandas as pd
import pytest
from numpy.typing import ArrayLike
from pandera.typing import DataFrame

from eu_solar_wind_audit import days, precompute, strata

CAPACITY_MW = 100.0
# Capacity factors 0.1 to 0.9.
EDGES = tuple(np.arange(1, 10) / 10)


def utc_hours(start: str, periods: int) -> pd.DatetimeIndex:
    return pd.date_range(start, periods=periods, freq="h", tz="UTC", unit="ns", name="time")


def covariates(
    hours: pd.DatetimeIndex, sun_elevation_deg: ArrayLike = 30.0
) -> DataFrame[precompute.Covariates]:
    """Covariates of 100 MW on `hours`, with the given sun elevation."""
    frame = pd.DataFrame(
        {
            "complete": True,
            "capacity_mw": CAPACITY_MW,
            "capacity_provisional": False,
            "era5_source": "ERA5",
            "sun_elevation_deg": sun_elevation_deg,
            "ghi_w_m2": 500.0,
            "clear_sky_ghi_w_m2": 700.0,
        },
        index=hours,
    )
    return precompute.Covariates.validate(frame)


def test_elevation_bands_start_at_their_lower_edge_and_night_holds_zero() -> None:
    elevations = [-20.0, 0.0, 0.01, 9.99, 10.0, 19.99, 20.0, 39.99, 40.0, 49.99, 50.0, 70.0]
    hours = utc_hours("2021-06-01 00:00", len(elevations))

    found = strata.classify_hours(
        pd.Series(50.0, index=hours), covariates(hours, elevations), EDGES
    )

    assert found["elevation_band"].tolist() == [
        "night",
        "night",
        "0-10",
        "0-10",
        "10-20",
        "10-20",
        "20-30",
        "30-40",
        "40-50",
        "40-50",
        "50+",
        "50+",
    ]


def test_hours_below_or_above_the_edges_go_to_the_first_or_last_decile() -> None:
    # Capacity factors 0, 0.05, 0.1, ..., 1.3 on 100 MW.
    potential_mwh = [0.0, 5.0, 10.0, 15.0, 50.0, 85.0, 90.0, 130.0]
    hours = utc_hours("2021-06-01 08:00", len(potential_mwh))

    found = strata.classify_hours(pd.Series(potential_mwh, index=hours), covariates(hours), EDGES)

    # An hour on an edge goes to the decile above it.
    assert found["decile"].tolist() == [1, 1, 2, 2, 6, 9, 10, 10]


def test_hours_take_the_type_of_their_day_in_paris() -> None:
    # From Friday 3 June 2022 at 21:00 UTC, 23:00 in Paris: Saturday in Paris starts at 22:00 UTC.
    hours = utc_hours("2022-06-03 21:00", 30)

    found = strata.classify_hours(pd.Series(50.0, index=hours), covariates(hours), EDGES)

    pd.testing.assert_series_equal(found["day_type"], days.day_types(hours))
    expected = ["working_day"] + ["saturday"] * 24 + ["sunday_or_holiday"] * 5
    assert found["day_type"].tolist() == expected


def test_hours_without_elevation_or_capacity_factor_have_no_band_or_decile() -> None:
    hours = utc_hours("2021-06-01 10:00", 3)
    potential_mwh = pd.Series([50.0, 50.0, np.nan], index=hours)

    found = strata.classify_hours(potential_mwh, covariates(hours, [35.0, np.nan, 35.0]), EDGES)

    pd.testing.assert_series_equal(
        found["elevation_band"],
        pd.Series(["30-40", None, "30-40"], index=hours, name="elevation_band", dtype="str"),
    )
    pd.testing.assert_series_equal(
        found["decile"], pd.Series([6, 6, None], index=hours, name="decile", dtype="Int64")
    )


@pytest.mark.parametrize(
    "edges",
    [EDGES[:-1], (*EDGES, 0.95), (0.2, 0.1, *EDGES[2:])],
    ids=["eight", "ten", "decreasing"],
)
def test_nine_increasing_decile_edges_are_required(edges: tuple[float, ...]) -> None:
    hours = utc_hours("2021-06-01 10:00", 2)

    with pytest.raises(ValueError, match="9 increasing decile edges"):
        strata.classify_hours(pd.Series(50.0, index=hours), covariates(hours), edges)


def test_the_potential_must_be_on_the_hours_of_the_covariates() -> None:
    hours = utc_hours("2021-06-01 10:00", 2)

    with pytest.raises(ValueError, match="not on the hours of the covariates"):
        strata.classify_hours(pd.Series(50.0, index=hours[1:]), covariates(hours), EDGES)
