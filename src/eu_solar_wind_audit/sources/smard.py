"""Client for the chart data files of SMARD, the electricity market data portal of the
Bundesnetzagentur (licence CC BY 4.0, attribution "Bundesnetzagentur | SMARD.de").

Generation and load values are energy per interval in MWh, prices are in EUR/MWh. Timestamps are
the start of the interval, in UTC.
"""

import json
import time
from pathlib import Path
from typing import Literal

import httpx
import pandas as pd

BASE_URL = "https://www.smard.de/app/chart_data"
CACHE_DIR = Path("data/raw/smard")
TIMEOUT_SECONDS = 30.0
RETRY_DELAYS_SECONDS = (2.0, 10.0)
# Recent weeks can still be filled in or revised, so they are downloaded again on each call.
SETTLED_AFTER = pd.Timedelta(days=7)

# Generation is published for the regions DE, 50Hertz, Amprion, TenneT and TransnetBW; offshore
# wind only for 50Hertz and TenneT.
SOLAR = 4068
WIND_ONSHORE = 4067
WIND_OFFSHORE = 1225
LOAD = 410

# Day-ahead price filter and region by bidding zone. Foreign zones are served under region DE.
PRICES = {
    "DE-LU": (4169, "DE-LU"),
    "FR": (254, "DE"),
    "BE": (4996, "DE"),
    "NL": (256, "DE"),
    "DK1": (252, "DE"),
    "DK2": (253, "DE"),
}

Resolution = Literal["quarterhour", "hour"]

_EPOCH = pd.Timestamp(0, tz="UTC")
_MILLISECOND = pd.Timedelta(milliseconds=1)


def fetch_series(
    filter_id: int,
    region: str,
    resolution: Resolution,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    client: httpx.Client | None = None,
    cache_dir: Path = CACHE_DIR,
) -> pd.Series:
    """Values on [start, end) with a UTC index; null values become NaN.

    `start` and `end` must be timezone-aware. Only the weekly files that overlap the interval are
    downloaded.
    """
    if client is None:
        with httpx.Client(timeout=TIMEOUT_SECONDS) as own_client:
            return fetch_series(
                filter_id, region, resolution, start, end, client=own_client, cache_dir=cache_dir
            )

    start_ms = (start - _EPOCH) // _MILLISECOND
    end_ms = (end - _EPOCH) // _MILLISECOND
    now_ms = (pd.Timestamp.now(tz="UTC") - _EPOCH) // _MILLISECOND
    folder = f"{BASE_URL}/{filter_id}/{region}"
    listing = json.loads(_get(client, f"{folder}/index_{resolution}.json"))
    stamps: list[int] = sorted(listing["timestamps"])
    # A weekly file runs until the next one starts; the last one is still open.
    week_ends: list[int | None] = [*stamps[1:], None]

    rows: list[tuple[int, float | None]] = []
    for week_start, week_end in zip(stamps, week_ends, strict=True):
        if week_start >= end_ms or (week_end is not None and week_end <= start_ms):
            continue
        name = f"{filter_id}_{region}_{resolution}_{week_start}.json"
        settled = week_end is not None and week_end + SETTLED_AFTER // _MILLISECOND <= now_ms
        content = _weekly_file(client, f"{folder}/{name}", cache_dir / name, settled)
        rows += json.loads(content)["series"]

    index = pd.to_datetime([ms for ms, _ in rows], unit="ms", utc=True)
    series = pd.Series([value for _, value in rows], index=index, dtype=float).sort_index()
    return series[(series.index >= start) & (series.index < end)]


def _weekly_file(client: httpx.Client, url: str, path: Path, settled: bool) -> bytes:
    if path.exists():
        return path.read_bytes()
    content = _get(client, url)
    if settled:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write under a temporary name so that an interrupted write never looks complete.
        partial = path.with_name(path.name + ".part")
        partial.write_bytes(content)
        partial.replace(path)
    return content


def _get(client: httpx.Client, url: str) -> bytes:
    """Retries timeouts, connection errors and server errors; other HTTP errors raise at once."""
    for delay in RETRY_DELAYS_SECONDS:
        try:
            response = client.get(url)
            if response.status_code < 500:
                break
        except httpx.TransportError:
            pass
        time.sleep(delay)
    else:
        response = client.get(url)
    response.raise_for_status()
    return response.content
