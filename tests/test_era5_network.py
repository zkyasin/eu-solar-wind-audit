import os

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from eu_solar_wind_audit.sources import era5

pytestmark = pytest.mark.skipif(os.environ.get("RUN_NETWORK_TESTS") != "1", reason="network")

CENTRE_OF_FRANCE = era5.Box(west=2.0, south=46.0, east=3.0, north=47.0)


@pytest.fixture(scope="module")
def final() -> xr.Dataset:
    return era5.open_icechunk()


@pytest.fixture(scope="module")
def recent() -> xr.Dataset:
    return era5.open_arco()


def test_arco_names_and_radiation_units(final: xr.Dataset, recent: xr.Dataset) -> None:
    for name in era5.VARIABLES:
        assert recent[name].attrs["short_name"] == name
    for name in era5.ACCUMULATED:
        assert final[name].attrs["units"] == recent[name].attrs["units"] == "J m**-2"


def test_icechunk_day_of_final_era5(final: xr.Dataset) -> None:
    start = pd.Timestamp("2025-06-21")

    out = era5.combine_stores(
        final, None, ["ssrd", "tisr", "u100"], CENTRE_OF_FRANCE, start, start + pd.Timedelta(days=1)
    )

    assert dict(out.sizes) == {"time": 24, "latitude": 5, "longitude": 5}
    assert not out.to_array().isnull().any()
    assert set(out["source"].values) == {"ERA5"}
    assert 600 < out["ssrd"].max() < 1100
    assert (out["ssrd"].sel(time="2025-06-21 22:00") == 0).all()
    # Solar noon is near 11:50 UTC here, so the hour starting at 11:00 gets the most sunlight.
    peak = out["tisr"].mean(["latitude", "longitude"]).idxmax().item()
    assert pd.Timestamp(peak) == pd.Timestamp("2025-06-21 11:00")


def test_arco_last_day_of_era5t(final: xr.Dataset, recent: xr.Dataset) -> None:
    end = pd.Timestamp(recent.indexes["time"][-1])
    start = end - pd.Timedelta(days=1)

    out = era5.combine_stores(final, recent, ["ssrd", "u100"], CENTRE_OF_FRANCE, start, end)

    assert dict(out.sizes) == {"time": 24, "latitude": 5, "longitude": 5}
    assert not out.to_array().isnull().any()
    assert set(out["source"].values) == {"ERA5T"}


def test_stores_agree_on_final_era5(final: xr.Dataset, recent: xr.Dataset) -> None:
    start, end = pd.Timestamp("2026-06-10"), pd.Timestamp("2026-06-13")

    icechunk = era5.combine_stores(final, None, ["ssrd", "u100"], CENTRE_OF_FRANCE, start, end)
    arco = era5.combine_stores(recent, None, ["ssrd", "u100"], CENTRE_OF_FRANCE, start, end)

    # Icechunk compresses with a relative error of about 1e-3.
    for name in ["ssrd", "u100"]:
        scale = float(np.abs(arco[name]).max())
        np.testing.assert_allclose(icechunk[name], arco[name], rtol=0, atol=2e-3 * scale)


def test_read_joins_the_stores_without_gap_or_duplicate(
    final: xr.Dataset, recent: xr.Dataset
) -> None:
    last_final = pd.Timestamp(final.indexes["time"][-1])
    start, end = last_final - pd.Timedelta(hours=3), last_final + pd.Timedelta(hours=4)

    out = era5.read_era5(["ssrd", "u100"], CENTRE_OF_FRANCE, start, end)

    assert out.indexes["time"].equals(pd.date_range(start, end, freq="h", inclusive="left"))
    assert not out.to_array().isnull().any()
    assert out["source"].values.tolist()[:3] == ["ERA5"] * 3
    # The hour starting at the last final stamp ends on the first ARCO stamp and takes its label.
    after = recent["source"].sel(time=slice(last_final + pd.Timedelta(hours=1), end))
    assert out["source"].values.tolist()[3:] == after.values.tolist()
