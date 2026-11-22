import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
import xarray as xr

from eu_solar_wind_audit.sources import era5

FRANCE = era5.Box(west=1.0, south=46.0, east=2.0, north=47.0)
# Store layout of both copies: latitudes from north to south, longitudes from 0 eastwards.
LATITUDES = np.array([47.0, 46.0, 45.0])
LONGITUDES = np.array([0.0, 1.0, 2.0, 358.0, 359.0])


def _store(
    time_dim: str, times: pd.DatetimeIndex, values: npt.ArrayLike, names: list[str]
) -> xr.Dataset:
    """`values` per hour, repeated over the grid."""
    shape = (len(times), len(LATITUDES), len(LONGITUDES))
    data = np.broadcast_to(np.asarray(values, dtype="float32")[:, None, None], shape)
    dims = (time_dim, "latitude", "longitude")
    return xr.Dataset(
        {name: (dims, data.copy()) for name in names},
        coords={time_dim: times, "latitude": LATITUDES, "longitude": LONGITUDES},
    )


def icechunk_store(times: pd.DatetimeIndex, values: npt.ArrayLike) -> xr.Dataset:
    store = _store("valid_time", times, values, list(era5.VARIABLES))
    return store.assign_coords(lsm=(("latitude", "longitude"), np.ones((3, 5))))


def arco_store(
    times: pd.DatetimeIndex, values: npt.ArrayLike, final_stop: str, era5t_stop: str
) -> xr.Dataset:
    store = _store("time", times, values, list(era5.ARCO_NAMES))
    return store.assign_attrs(
        valid_time_start="1940-01-01",
        valid_time_stop=final_stop,
        valid_time_stop_era5t=era5t_stop,
    )


def hours(start: str, end: str) -> pd.DatetimeIndex:
    return pd.date_range(start, end, freq="h")


def test_radiation_is_the_mean_power_of_the_hour_before_its_stamp() -> None:
    times = hours("2026-06-01 00:00", "2026-06-01 06:00")
    joules = 3600.0 * np.arange(len(times))
    final = era5.from_icechunk(icechunk_store(times, joules))

    out = era5.combine_stores(
        final,
        None,
        ["ssrd"],
        FRANCE,
        pd.Timestamp("2026-06-01 02:00"),
        pd.Timestamp("2026-06-01 05:00"),
    )

    assert out.indexes["time"].equals(hours("2026-06-01 02:00", "2026-06-01 04:00"))
    # The 03:00 stamp holds the energy of [02:00, 03:00).
    assert out["ssrd"].isel(latitude=0, longitude=0).values.tolist() == [3.0, 4.0, 5.0]
    assert out["ssrd"].attrs["units"] == "W m-2"


def test_instantaneous_field_is_the_mean_of_both_ends_of_the_hour() -> None:
    times = hours("2026-06-01 00:00", "2026-06-01 06:00")
    final = era5.from_icechunk(icechunk_store(times, 2.0 * np.arange(len(times))))

    out = era5.combine_stores(
        final,
        None,
        ["u100"],
        FRANCE,
        pd.Timestamp("2026-06-01 02:00"),
        pd.Timestamp("2026-06-01 05:00"),
    )

    assert out.indexes["time"].equals(hours("2026-06-01 02:00", "2026-06-01 04:00"))
    assert out["u100"].isel(latitude=0, longitude=0).values.tolist() == [5.0, 7.0, 9.0]


def test_period_needing_a_stamp_before_the_store_is_refused() -> None:
    times = hours("2026-06-01 00:00", "2026-06-01 06:00")
    final = era5.from_icechunk(icechunk_store(times, np.zeros(len(times))))

    with pytest.raises(ValueError, match="every hour"):
        era5.combine_stores(
            final, None, ["ssrd"], FRANCE, times[0] - pd.Timedelta(hours=1), times[3]
        )


def test_stores_join_at_the_end_of_final_era5() -> None:
    final_times = hours("2026-06-30 00:00", "2026-06-30 23:00")
    final = era5.from_icechunk(icechunk_store(final_times, np.full(24, 3600.0)))
    # ARCO also holds the last final hours, with other values, to show that Icechunk wins there.
    recent_times = hours("2026-06-30 20:00", "2026-07-01 23:00")
    recent = era5.from_arco(
        arco_store(recent_times, np.full(28, 7200.0), "2026-06-30", "2026-07-01")
    )

    out = era5.combine_stores(
        final,
        recent,
        ["ssrd", "u100"],
        FRANCE,
        pd.Timestamp("2026-06-30 21:00"),
        pd.Timestamp("2026-07-01 02:00"),
    )

    expected = hours("2026-06-30 21:00", "2026-07-01 01:00")
    assert out.indexes["time"].equals(expected)
    assert out["source"].values.tolist() == ["ERA5", "ERA5", "ERA5T", "ERA5T", "ERA5T"]
    point = out.isel(latitude=0, longitude=0)
    assert point["ssrd"].values.tolist() == [1.0, 1.0, 2.0, 2.0, 2.0]
    # The hour from 23:00 averages the last final stamp and the first ERA5T stamp.
    assert point["u100"].values.tolist() == [3600.0, 3600.0, 5400.0, 7200.0, 7200.0]


def test_arco_labels_final_hours_and_drops_the_padding() -> None:
    times = hours("2026-06-30 22:00", "2026-07-02 01:00")
    store = arco_store(times, np.arange(len(times), dtype="float64"), "2026-06-30", "2026-07-01")
    store["2m_temperature"][-2:] = np.nan

    recent = era5.from_arco(store)

    assert recent.indexes["time"][-1] == pd.Timestamp("2026-07-01 23:00")
    assert recent["source"].sel(time="2026-06-30 23:00").item() == "ERA5"
    assert recent["source"].sel(time="2026-07-01 00:00").item() == "ERA5T"
    assert not recent["t2m"].isnull().any()


def test_period_past_the_era5t_tail_is_refused() -> None:
    final = era5.from_icechunk(
        icechunk_store(hours("2026-06-30 00:00", "2026-06-30 23:00"), np.zeros(24))
    )
    recent = era5.from_arco(
        arco_store(
            hours("2026-07-01 00:00", "2026-07-02 23:00"), np.zeros(48), "2026-06-30", "2026-07-01"
        )
    )

    with pytest.raises(ValueError, match="every hour"):
        era5.combine_stores(
            final,
            recent,
            ["ssrd"],
            FRANCE,
            pd.Timestamp("2026-07-01 12:00"),
            pd.Timestamp("2026-07-02 00:00"),
        )


def test_missing_hour_inside_a_store_is_refused() -> None:
    times = hours("2026-06-01 00:00", "2026-06-01 06:00").delete(3)
    final = era5.from_icechunk(icechunk_store(times, np.zeros(len(times))))

    with pytest.raises(ValueError, match="every hour"):
        era5.combine_stores(
            final,
            None,
            ["u100"],
            FRANCE,
            pd.Timestamp("2026-06-01 01:00"),
            pd.Timestamp("2026-06-01 05:00"),
        )


def test_box_across_longitude_zero_is_contiguous() -> None:
    times = hours("2026-06-01 00:00", "2026-06-01 02:00")
    store = icechunk_store(times, np.zeros(len(times)))
    store["t2m"] = store["t2m"] + xr.DataArray(LONGITUDES.astype("float32"), dims="longitude")

    final = era5.from_icechunk(store)
    out = era5.combine_stores(
        final,
        None,
        ["t2m"],
        era5.Box(west=-1.0, south=45.0, east=1.0, north=47.0),
        times[0],
        times[-1],
    )

    assert out["longitude"].values.tolist() == [-1.0, 0.0, 1.0]
    assert out["t2m"].isel(time=0, latitude=0).values.tolist() == [359.0, 0.0, 1.0]


def test_latitudes_ascend_with_their_values() -> None:
    times = hours("2026-06-01 00:00", "2026-06-01 02:00")
    store = icechunk_store(times, np.zeros(len(times)))
    store["t2m"] = store["t2m"] + xr.DataArray(LATITUDES.astype("float32"), dims="latitude")

    final = era5.from_icechunk(store)
    out = era5.combine_stores(
        final,
        None,
        ["t2m"],
        era5.Box(west=0.0, south=45.0, east=0.0, north=47.0),
        times[0],
        times[-1],
    )

    assert out["latitude"].values.tolist() == [45.0, 46.0, 47.0]
    assert out["t2m"].isel(time=0, longitude=0).values.tolist() == [45.0, 46.0, 47.0]


def test_arco_is_not_opened_when_final_era5_covers_the_period(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    times = hours("2026-06-01 00:00", "2026-06-01 06:00")
    final = era5.from_icechunk(icechunk_store(times, np.zeros(len(times))))
    monkeypatch.setattr(era5, "open_icechunk", lambda: final)

    def fail() -> xr.Dataset:
        raise AssertionError("ARCO opened")

    monkeypatch.setattr(era5, "open_arco", fail)

    out = era5.read_era5(["ssrd"], FRANCE, times[1], times[-1])

    assert out.sizes["time"] == 5
    assert set(out["source"].values) == {"ERA5"}
