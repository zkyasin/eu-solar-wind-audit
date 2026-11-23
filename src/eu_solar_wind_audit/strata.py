"""Strata of the calibrated potential (evaluation plan, "Two model objects", item 2, and
Deviations): the published output over the potential is estimated in each stratum of hours.

The stratum of an hour is the band of its sun elevation, the decile of the capacity factor of the
potential and the type of its Europe/Paris day (days.day_types). The elevation is the
`sun_elevation_deg` covariate of precompute: capacity-weighted for solar, at the capacity centroid
of the fleet for onshore wind. The decile edges are the 10 % to 90 % quantiles of the capacity
factor over the eligible hours of the training years; hours below or above them go to the first
or last decile. The published output never defines a stratum, since it depends on modulation.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd
import pandera.pandas as pa
from pandera.typing import DataFrame, Index, Series

from eu_solar_wind_audit import days, grid, precompute

# Sun elevation in degrees: the night band up to 0, then a band from each edge to the next, the
# last without an upper edge.
ELEVATION_EDGES = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]
ELEVATION_BANDS = ["night", "0-10", "10-20", "20-30", "30-40", "40-50", "50+"]
DECILES = list(range(1, 11))
# Quantile levels of the 9 decile edges.
DECILE_LEVELS = np.arange(1, 10) / 10


class HourStrata(pa.DataFrameModel):
    """Stratum of the hour starting at `time`: the band of its sun elevation, the decile of the
    capacity factor of the potential, from 1 to 10, and the type of its day. The band is null
    without a sun elevation, the decile without a capacity factor."""

    time: Index[grid.UtcTime] = pa.Field(unique=True, check_name=True)
    elevation_band: Series[str] = pa.Field(isin=ELEVATION_BANDS, nullable=True)
    decile: Series[pd.Int64Dtype] = pa.Field(isin=DECILES, nullable=True)
    day_type: Series[str] = pa.Field(isin=days.DAY_TYPES)

    class Config:
        strict = True


def classify_hours(
    potential_mwh: pd.Series, covariates: pd.DataFrame, decile_edges: Sequence[float]
) -> DataFrame[HourStrata]:
    """Stratum of each hour of `covariates`, a precompute.Covariates frame, for the potential
    `potential_mwh` in MWh on the same hours and the 9 increasing `decile_edges`. An hour on an
    edge goes to the band or decile above it, except at 0 degrees, which is night."""
    precompute.Covariates.validate(covariates)
    hours = pd.DatetimeIndex(covariates.index)
    if not potential_mwh.index.equals(hours):
        raise ValueError("the potential is not on the hours of the covariates")
    edges = np.asarray(decile_edges, dtype=float)
    if edges.shape != DECILE_LEVELS.shape or (np.diff(edges) < 0).any():
        raise ValueError(f"expected {DECILE_LEVELS.size} increasing decile edges")

    elevation = covariates["sun_elevation_deg"].to_numpy()
    band = np.where(elevation <= 0, 0, np.searchsorted(ELEVATION_EDGES, elevation, side="right"))
    factor = (potential_mwh / covariates["capacity_mw"]).to_numpy()
    decile = np.searchsorted(edges, factor, side="right") + 1
    strata = pd.DataFrame(
        {
            "elevation_band": pd.Series(
                np.take(ELEVATION_BANDS, band), index=hours, dtype="str"
            ).where(~np.isnan(elevation)),
            "decile": pd.Series(decile, index=hours, dtype="Int64").where(~np.isnan(factor)),
            "day_type": days.day_types(hours),
        }
    )
    return HourStrata.validate(strata)
