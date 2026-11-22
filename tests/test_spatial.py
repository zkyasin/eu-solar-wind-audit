import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from shapely.geometry import box

from eu_solar_wind_audit.spatial import capacity_weights, region_fractions

STEP = 0.25
LATITUDE = np.arange(45.0, 46.0, STEP)
LONGITUDE = np.arange(2.0, 3.0, STEP)


def cell(latitude: float, longitude: float) -> tuple[float, float, float, float]:
    """Bounds (west, south, east, north) of the grid cell centred on the point."""
    half = STEP / 2
    return (longitude - half, latitude - half, longitude + half, latitude + half)


def squares(polygons: dict[str, tuple[float, float, float, float]]) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {"region": list(polygons)},
        geometry=[box(*bounds) for bounds in polygons.values()],
        crs="EPSG:4326",
    )


def hand_fractions(values: dict[str, list[list[float]]], latitude: list[float]) -> xr.DataArray:
    """Fractions on a grid with one longitude column per value in each row."""
    data = np.array(list(values.values()), dtype=float)
    return xr.DataArray(
        data,
        coords={
            "region": list(values),
            "latitude": latitude,
            "longitude": np.arange(data.shape[2]) * STEP,
        },
        dims=("region", "latitude", "longitude"),
    )


def capacity_of(values: dict[str, float]) -> xr.DataArray:
    return xr.DataArray(list(values.values()), coords={"region": list(values)}, dims="region")


def test_fractions_of_full_half_and_quarter_cells() -> None:
    west, south, east, north = cell(45.5, 2.5)
    regions = squares(
        {
            "full": cell(45.0, 2.0),
            "half": (west, south, (west + east) / 2, north),
            "quarter": (*cell(45.75, 2.75)[:2], 2.75, 45.75),
        }
    )

    fractions = region_fractions(regions, LATITUDE, LONGITUDE)

    expected = {"full": (45.0, 2.0, 1.0), "half": (45.5, 2.5, 0.5), "quarter": (45.75, 2.75, 0.25)}
    for region, (latitude, longitude, fraction) in expected.items():
        covered = fractions.sel(region=region)
        assert float(covered.sel(latitude=latitude, longitude=longitude)) == pytest.approx(fraction)
        assert float(covered.sum()) == pytest.approx(fraction)


def test_fractions_use_era5_names_and_region_codes() -> None:
    regions = squares({"75": cell(45.25, 2.5), "2A": cell(45.5, 2.25)})

    fractions = region_fractions(regions, LATITUDE, LONGITUDE)

    assert fractions.dims == ("region", "latitude", "longitude")
    assert fractions["region"].to_numpy().tolist() == ["75", "2A"]
    np.testing.assert_array_equal(fractions["latitude"], LATITUDE)
    np.testing.assert_array_equal(fractions["longitude"], LONGITUDE)


def test_neighbours_sharing_a_cell_cover_it_once() -> None:
    west, south, east, north = cell(45.25, 2.25)
    middle = west + 0.3 * STEP
    regions = squares({"a": (west - 0.5, south, middle, north), "b": (middle, south, east, north)})

    fractions = region_fractions(regions, LATITUDE, LONGITUDE)

    shared = fractions.sel(latitude=45.25, longitude=2.25)
    assert shared.sel(region="a") == pytest.approx(0.3)
    assert shared.sum() == pytest.approx(1.0)


def test_region_outside_the_grid_keeps_an_empty_slice() -> None:
    regions = squares({"inside": cell(45.0, 2.0), "outside": cell(50.0, 2.0)})

    fractions = region_fractions(regions, LATITUDE, LONGITUDE)

    assert fractions["region"].to_numpy().tolist() == ["inside", "outside"]
    assert float(fractions.sel(region="outside").sum()) == 0.0


def test_capacity_follows_cell_area_shrinking_with_cos_latitude() -> None:
    fractions = hand_fractions({"a": [[1.0], [1.0]]}, latitude=[0.0, 60.0])

    weights = capacity_weights(fractions, capacity_of({"a": 30.0}))

    # cos(60°) = 0.5, so the equatorial cell holds twice the area.
    np.testing.assert_allclose(weights.sel(longitude=0.0), [20.0, 10.0])


def test_capacity_follows_the_covered_fraction() -> None:
    fractions = hand_fractions({"a": [[1.0, 0.5, 0.0]]}, latitude=[45.0])

    weights = capacity_weights(fractions, capacity_of({"a": 9.0}))

    np.testing.assert_allclose(weights.sel(latitude=45.0), [6.0, 3.0, 0.0])


def test_each_region_keeps_its_total() -> None:
    regions = squares({"a": (1.8, 44.8, 2.4, 45.6), "b": (2.4, 45.1, 3.2, 46.2)})
    fractions = region_fractions(regions, LATITUDE, LONGITUDE)

    for region, capacity in {"a": 120.0, "b": 75.0}.items():
        weights = capacity_weights(fractions, capacity_of({region: capacity}))
        assert float(weights.sum()) == pytest.approx(capacity)
        assert float(weights.where(fractions.sel(region=region) == 0, 0.0).sum()) == 0.0

    both = capacity_weights(fractions, capacity_of({"a": 120.0, "b": 75.0}))
    assert float(both.sum()) == pytest.approx(195.0)


def test_extra_dims_are_kept_and_conserved_per_step() -> None:
    fractions = hand_fractions(
        {"a": [[1.0, 0.5], [0.0, 0.0]], "b": [[0.0, 0.5], [1.0, 1.0]]}, latitude=[0.0, 60.0]
    )
    months = pd.DatetimeIndex(["2025-01-31", "2025-02-28"], name="month_end")
    capacity = xr.DataArray(
        [[90.0, 30.0], [180.0, 0.0]],
        coords={"month_end": months, "region": ["a", "b"]},
        dims=("month_end", "region"),
    )

    weights = capacity_weights(fractions, capacity)

    assert weights.dims == ("month_end", "latitude", "longitude")
    # Areas: a covers 1 and 0.5 at the equator; b covers 0.5 there and 2 x 0.5 at 60°N.
    np.testing.assert_allclose(weights.sel(month_end="2025-01-31"), [[60.0, 40.0], [10.0, 10.0]])
    np.testing.assert_allclose(weights.sel(month_end="2025-02-28"), [[120.0, 60.0], [0.0, 0.0]])


def test_capacity_in_a_region_without_covered_cell_raises() -> None:
    fractions = hand_fractions({"a": [[1.0]], "empty": [[0.0]]}, latitude=[45.0])

    with pytest.raises(ValueError, match="empty"):
        capacity_weights(fractions, capacity_of({"a": 1.0, "empty": 2.0}))


def test_zero_capacity_in_a_region_without_covered_cell_is_allowed() -> None:
    fractions = hand_fractions({"a": [[1.0]], "empty": [[0.0]]}, latitude=[45.0])

    weights = capacity_weights(fractions, capacity_of({"a": 1.0, "empty": 0.0}))

    assert float(weights.sum()) == pytest.approx(1.0)


def test_capacity_in_a_region_without_polygon_raises() -> None:
    fractions = hand_fractions({"a": [[1.0]]}, latitude=[45.0])

    with pytest.raises(ValueError, match="971"):
        capacity_weights(fractions, capacity_of({"a": 1.0, "971": 2.0}))
