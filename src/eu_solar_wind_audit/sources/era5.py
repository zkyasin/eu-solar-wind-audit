"""ERA5 hourly single-level fields from two anonymous cloud copies.

The Earthmover Icechunk store on AWS holds final ERA5 in time-series chunks. Google ARCO-ERA5 adds
the ERA5T tail, released about six days after real time. Times are naive UTC.
"""

from collections.abc import Sequence
from typing import NamedTuple

import icechunk
import numpy as np
import pandas as pd
import xarray as xr

ARCO_URL = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
# Final ERA5 from the time-series group of the Earthmover store.
ICECHUNK_BUCKET = "earthmover-icechunk-era5"
ICECHUNK_PREFIX = "icechunkV2"
# Connections to the bucket sometimes take longer than the 3 s that icechunk waits by default, and
# three runs of 2026-11-25 stopped on them: wait up to 30 s, and retry with a backoff of up to a
# minute.
STORAGE_SETTINGS = icechunk.StorageSettings(
    timeouts=icechunk.StorageTimeoutSettings(connect_timeout_ms=30_000, read_timeout_ms=120_000),
    retries=icechunk.StorageRetriesSettings(
        max_tries=12, initial_backoff_ms=1_000, max_backoff_ms=60_000
    ),
)
ICECHUNK_GROUP = "single/temporal"
ARCO_NAMES = {
    "surface_solar_radiation_downwards": "ssrd",
    "total_sky_direct_solar_radiation_at_surface": "fdir",
    "toa_incident_solar_radiation": "tisr",
    "2m_temperature": "t2m",
    "total_cloud_cover": "tcc",
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
    "100m_u_component_of_wind": "u100",
    "100m_v_component_of_wind": "v100",
}
VARIABLES = tuple(ARCO_NAMES.values())
# Accumulated in J m-2 over the hour ending at the timestamp; the others are instantaneous.
ACCUMULATED = frozenset({"ssrd", "fdir", "tisr"})
HOUR = pd.Timedelta(hours=1)


class Box(NamedTuple):
    """Bounds in degrees, inclusive, with longitudes in [-180, 180)."""

    west: float
    south: float
    east: float
    north: float


def _standard_grid(ds: xr.Dataset) -> xr.Dataset:
    # ARCO stores its coordinates as float32, Icechunk as float64.
    latitude = ds["latitude"].astype("float64")
    longitude = (ds["longitude"].astype("float64") + 180) % 360 - 180
    return ds.assign_coords(latitude=latitude, longitude=longitude).sortby(
        ["latitude", "longitude"]
    )


def from_icechunk(store: xr.Dataset) -> xr.Dataset:
    ds = store[list(VARIABLES)].reset_coords(drop=True).rename(valid_time="time")
    source = np.full(ds.sizes["time"], "ERA5")
    return _standard_grid(ds).assign_coords(source=("time", source))


def open_icechunk() -> xr.Dataset:
    """Final ERA5, lazily, from the time-series group of the Earthmover store."""
    storage = icechunk.s3_storage(
        bucket=ICECHUNK_BUCKET, prefix=ICECHUNK_PREFIX, region="us-east-1", anonymous=True
    )
    config = icechunk.RepositoryConfig(storage=STORAGE_SETTINGS)
    session = icechunk.Repository.open(storage, config=config).readonly_session(branch="main")
    store = xr.open_zarr(session.store, group=ICECHUNK_GROUP, consolidated=False, chunks=None)
    return from_icechunk(store)


def from_arco(store: xr.Dataset) -> xr.Dataset:
    # The attributes name a day; its 23:00 stamp is the last one with data.
    final_end = pd.Timestamp(store.attrs["valid_time_stop"]) + 23 * HOUR
    era5t_end = pd.Timestamp(store.attrs["valid_time_stop_era5t"]) + 23 * HOUR
    ds = store[list(ARCO_NAMES)].rename(ARCO_NAMES)
    # The time axis is padded with NaN from 1900 to 2050.
    ds = ds.sel(time=slice(pd.Timestamp(store.attrs["valid_time_start"]), era5t_end))
    # Whole ERA5T days can be missing (all NaN) inside that range; they stay missing.
    source = np.where(ds.indexes["time"] <= final_end, "ERA5", "ERA5T")
    return _standard_grid(ds).assign_coords(source=("time", source))


def open_arco() -> xr.Dataset:
    """Final ERA5 then ERA5T, lazily, from Google ARCO-ERA5."""
    store = xr.open_zarr(ARCO_URL, chunks=None, storage_options={"token": "anon"})
    return from_arco(store)


def _subset(
    ds: xr.Dataset, variables: Sequence[str], box: Box, start: pd.Timestamp, end: pd.Timestamp
) -> xr.Dataset:
    return ds[list(variables)].sel(
        time=slice(start, end),
        latitude=slice(box.south, box.north),
        longitude=slice(box.west, box.east),
    )


def combine_stores(
    final: xr.Dataset,
    recent: xr.Dataset | None,
    variables: Sequence[str],
    box: Box,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> xr.Dataset:
    """Hours [start, end) labelled at their start, from `final` up to its last time and from
    `recent` after it. Radiation becomes the mean power over the hour in W m-2."""
    last_final = pd.Timestamp(final.indexes["time"][-1])
    parts = [_subset(final, variables, box, start, min(end, last_final))]
    if recent is not None:
        parts.append(_subset(recent, variables, box, max(start, last_final + HOUR), end))
    # The hour starting at h needs the stamp h + 1 h, and h as well for instantaneous fields.
    raw = xr.concat(parts, dim="time", join="exact")
    expected = pd.date_range(start, end, freq="h")
    if not raw.indexes["time"].equals(expected):
        raise ValueError(f"ERA5 stores do not hold every hour from {start} to {end}")

    hourly = {}
    for name in variables:
        values = raw[name]
        if name in ACCUMULATED:
            hourly[name] = (values / 3600).assign_attrs(units="W m-2")
        else:
            hourly[name] = values.rolling(time=2).mean(keep_attrs=True)
    return xr.Dataset(hourly).assign_coords(time=expected - HOUR).isel(time=slice(1, None))


def read_era5(
    variables: Sequence[str], box: Box, start: pd.Timestamp, end: pd.Timestamp
) -> xr.Dataset:
    """Hourly fields for hours [start, end), naive UTC, with a per-hour `source` of ERA5 or ERA5T.

    ARCO is opened only when the period goes past the end of the Icechunk store.
    """
    final = open_icechunk()
    recent = open_arco() if end > final.indexes["time"][-1] else None
    return combine_stores(final, recent, variables, box, start, end)
