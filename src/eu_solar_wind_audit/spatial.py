"""Installed capacity spread over the cells of a regular latitude-longitude grid (ERA5, 0.25
degree). A weight is the capacity in MW attributed to a cell, so a capacity-weighted sum of cell
capacity factors gives the power of the fleet.

Within a region, capacity is spread in proportion to land area: the true distribution inside a
region is not known here.
"""

import geopandas as gpd
import numpy as np
import numpy.typing as npt
import pandas as pd
import regionmask
import xarray as xr

GRID = ("latitude", "longitude")


def region_fractions(
    regions: gpd.GeoDataFrame, latitude: npt.ArrayLike, longitude: npt.ArrayLike
) -> xr.DataArray:
    """Fraction (0 to 1) of each grid cell, centred on the given coordinates, covered by each
    region of `regions` (columns `region` and `geometry`, EPSG:4326).

    The grid must be equally spaced. Every region gets a slice, even one that covers no cell.
    """
    mask = regionmask.from_geopandas(
        regions.reset_index(drop=True), names="region"
    ).mask_3D_frac_approx(np.asarray(longitude), np.asarray(latitude), drop=False)
    return (
        mask.rename(lat="latitude", lon="longitude")
        .assign_coords(region=mask["names"].to_numpy())
        .drop_vars(["abbrevs", "names"])
    )


def capacity_weights(fractions: xr.DataArray, capacity: xr.DataArray) -> xr.DataArray:
    """Capacity in MW per cell, with dims (..., latitude, longitude).

    Each region's capacity (`capacity` has a `region` dim and possibly others, such as
    `month_end`) is spread over its cells in proportion to the covered area, the covered fraction
    times cos(latitude), so the cells of a region add up to its capacity.
    """
    unknown = sorted(set(capacity["region"].to_numpy()) - set(fractions["region"].to_numpy()))
    if unknown:
        raise ValueError(f"regions with capacity but no polygon: {unknown}")
    area = fractions.sel(region=capacity["region"]) * np.cos(np.deg2rad(fractions["latitude"]))
    total = area.sum(GRID)
    empty = [
        str(region)
        for region in total["region"].to_numpy()
        if total.sel(region=region) == 0 and (capacity.sel(region=region) > 0).any()
    ]
    if empty:
        raise ValueError(f"regions with capacity but no covered cell: {empty}")
    # A region without a covered cell has zero area everywhere, so any divisor gives zero shares.
    share = area / total.where(total > 0, 1.0)
    weights: xr.DataArray = xr.dot(capacity, share, dim="region")
    return weights.transpose(..., *GRID)


def point_weights(
    points: pd.DataFrame, latitude: npt.ArrayLike, longitude: npt.ArrayLike
) -> xr.DataArray:
    """Capacity in MW per cell of point sources (columns `latitude`, `longitude`, `capacity_mw`),
    each added to the cell whose centre is nearest. The grid must be ascending and equally spaced.
    """
    weights = xr.DataArray(
        0.0,
        coords={"latitude": np.asarray(latitude), "longitude": np.asarray(longitude)},
        dims=GRID,
    )
    rows = _nearest_cell(weights.indexes["latitude"], points["latitude"])
    columns = _nearest_cell(weights.indexes["longitude"], points["longitude"])
    np.add.at(weights.values, (rows, columns), points["capacity_mw"].to_numpy(dtype=float))
    return weights


def _nearest_cell(centres: pd.Index, values: "pd.Series[float]") -> npt.NDArray[np.intp]:
    """Index of the cell holding each value; a value beyond the grid's outer edges raises."""
    half_step = float(np.diff(centres.to_numpy()).max()) / 2
    cells = centres.get_indexer(pd.Index(values), method="nearest", tolerance=half_step)
    if (cells < 0).any():
        raise ValueError(f"points outside the grid: {values[cells < 0].to_list()}")
    return cells
