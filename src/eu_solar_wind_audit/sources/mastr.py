"""Installed solar and wind capacity of Germany by district (Kreis), from the full public export of
the Marktstammdatenregister (MaStR) of the Bundesnetzagentur (licence dl-de/by-2-0).

Powers are in kW in the export and in MW in the outputs. For solar units the gross power
(Bruttoleistung) is the module power in kWp, so DC, and the net rated power (Nettonennleistung) is
the lesser of the inverter active power and the gross power, so AC (MaStR field definitions,
version 26.2). For wind units both are the same rated power.
"""

import hashlib
import json
import re
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pandas as pd

DOWNLOAD_PAGE = "https://www.marktstammdatenregister.de/MaStR/Datendownload"
CACHE_DIR = Path("data/raw/mastr")
# The daily export gets a new name each day; the quarterly exports are under /Stichtag/.
EXPORT_LINK = re.compile(
    r'href="(https://download\.marktstammdatenregister\.de/Gesamtdatenexport_\d{8}_[\d.]+\.zip)"'
)
ZIP_TYPES = ("application/zip", "application/x-zip-compressed")
TIMEOUT = httpx.Timeout(30, read=300)
RETRY_DELAYS_SECONDS = (10.0, 60.0)


def find_export_url(client: httpx.Client) -> str:
    """URL of the current daily export, read from the download page."""
    response = client.get(DOWNLOAD_PAGE)
    response.raise_for_status()
    links = {match.group(1) for match in EXPORT_LINK.finditer(response.text)}
    if len(links) != 1:
        raise ValueError(f"{DOWNLOAD_PAGE}: expected one daily export link, found {sorted(links)}")
    return links.pop()


def download_export(client: httpx.Client, url: str, cache_dir: Path = CACHE_DIR) -> Path:
    """Path of the export zip, downloaded unless already cached.

    An interrupted download is resumed from its .part file. Once complete, a JSON file next to the
    zip records the URL, retrieval time, size and SHA-256.
    """
    dest = cache_dir / url.rsplit("/", 1)[1]
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".part")
    for delay in (*RETRY_DELAYS_SECONDS, None):
        try:
            _resume(client, url, partial)
            break
        except httpx.TransportError:
            if delay is None:
                raise
            time.sleep(delay)

    with partial.open("rb") as file:
        sha256 = hashlib.file_digest(file, "sha256").hexdigest()
    metadata = {
        "url": url,
        "file": dest.name,
        "retrieved_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "size": partial.stat().st_size,
        "sha256": sha256,
    }
    # Written before the zip gets its final name, so that a cached zip always has its metadata.
    _metadata_path(dest).write_text(json.dumps(metadata, indent=2) + "\n")
    partial.replace(dest)
    return dest


def export_date_of(path: Path) -> pd.Timestamp:
    match = re.fullmatch(r"Gesamtdatenexport_(\d{8})_[\d.]+\.zip", path.name)
    if match is None:
        raise ValueError(f"{path.name}: not a daily export file name")
    return pd.Timestamp(match.group(1))


def _resume(client: httpx.Client, url: str, partial: Path) -> None:
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    with client.stream("GET", url, headers=headers) as response:
        response.raise_for_status()
        # Never cache an error or maintenance page under the zip name.
        content_type = response.headers.get("content-type", "")
        if not content_type.startswith(ZIP_TYPES):
            raise ValueError(f"{url}: not a zip file ({content_type})")
        # A server that ignores the range sends the whole file again.
        mode = "ab" if response.status_code == 206 else "wb"
        with partial.open(mode) as file:
            for chunk in response.iter_bytes():
                file.write(chunk)


def _metadata_path(path: Path) -> Path:
    return path.with_name(path.name + ".json")
