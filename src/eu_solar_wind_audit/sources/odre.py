"""Client for dataset exports of ODRÉ, the open data portal of the French energy networks."""

import hashlib
import json
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path

import httpx

BASE_URL = "https://odre.opendatasoft.com/api/explore/v2.1/catalog/datasets"
CACHE_DIR = Path("data/raw/odre")


def make_client() -> httpx.Client:
    # Transport retries cover failed connections only; HTTP error statuses are raised.
    return httpx.Client(
        transport=httpx.HTTPTransport(retries=3),
        timeout=httpx.Timeout(30, read=300),
        follow_redirects=True,
    )


def export_url(dataset: str, export_format: str) -> str:
    return f"{BASE_URL}/{dataset}/exports/{export_format}"


def utc_days(start: date, end: date) -> str:
    """ODSQL filter on the timestamps from `start` 00:00 UTC to `end` + 1 day 00:00 UTC, both
    included, so that every UTC day can be rebuilt from end-labelled or instantaneous values."""
    if end < start:
        raise ValueError(f"end {end} is before start {start}")
    first = datetime.combine(start, time(), UTC).isoformat()
    last = datetime.combine(end + timedelta(days=1), time(), UTC).isoformat()
    return f"date_heure >= date'{first}' AND date_heure <= date'{last}'"


def download_export(
    client: httpx.Client,
    dataset: str,
    *,
    select: str | None = None,
    where: str | None = None,
    export_format: str = "parquet",
    cache_dir: Path = CACHE_DIR,
    refresh: bool = False,
) -> Path:
    """Download an export once and return its path under `cache_dir`/`dataset`.

    The file name is a hash of the query, so a different selection is downloaded again.
    """
    params = {name: value for name, value in [("select", select), ("where", where)] if value}
    key = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:16]
    dest = cache_dir / dataset / f"{key}.{export_format}"
    if refresh or not dest.exists():
        _stream_to(client, export_url(dataset, export_format), params, dest)
    return dest


def _stream_to(client: httpx.Client, url: str, params: dict[str, str], dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Write under a temporary name so that an interrupted download never looks complete.
    partial = dest.with_name(dest.name + ".part")
    with client.stream("GET", url, params=params) as response:
        response.raise_for_status()
        with partial.open("wb") as file:
            for chunk in response.iter_bytes():
                file.write(chunk)
    partial.replace(dest)
