"""Files pinned by their SHA-256: downloaded once, checked, then read from the cache."""

import hashlib
import time
from pathlib import Path

import httpx

TIMEOUT_SECONDS = 300.0
# Some servers close the connection mid-transfer.
RETRY_DELAYS_SECONDS = (2.0, 10.0)


def download(url: str, sha256: str, dest: Path, client: httpx.Client | None = None) -> Path:
    """`dest`, downloaded from `url` and checked against `sha256` unless already cached. A
    dropped connection is retried; an HTTP error raises at once, and a file with another checksum
    raises ValueError. Nothing is cached on failure."""
    if dest.exists():
        return dest
    if client is None:
        with httpx.Client(timeout=TIMEOUT_SECONDS) as own_client:
            return download(url, sha256, dest, own_client)

    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".part")
    for delay in RETRY_DELAYS_SECONDS:
        try:
            digest = _stream_to(client, url, partial)
            break
        except httpx.TransportError:
            time.sleep(delay)
    else:
        digest = _stream_to(client, url, partial)
    if digest != sha256:
        partial.unlink()
        raise ValueError(f"{url}: SHA-256 {digest}, expected {sha256}")
    partial.replace(dest)
    return dest


def _stream_to(client: httpx.Client, url: str, dest: Path) -> str:
    digest = hashlib.sha256()
    with client.stream("GET", url) as response:
        response.raise_for_status()
        with dest.open("wb") as file:
            for chunk in response.iter_bytes():
                digest.update(chunk)
                file.write(chunk)
    return digest.hexdigest()
