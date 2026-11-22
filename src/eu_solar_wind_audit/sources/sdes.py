"""Installed solar and wind capacity of France by department, from the quarterly dashboards of the
SDES, the statistics service of the French ministry for the environment."""

from pathlib import Path

import httpx

BASE_URL = "https://www.statistiques.developpement-durable.gouv.fr"
# Each quarterly publication gets new media ids; these are the second quarter 2026 files.
SOLAR_2026_Q2 = 9714
WIND_2026_Q2 = 9715
RAW_DIR = Path("data/raw/sdes")
XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


def download_dashboard(client: httpx.Client, media_id: int, raw_dir: Path = RAW_DIR) -> Path:
    """Path of the dashboard workbook `media_id`, downloaded unless already cached."""
    dest = raw_dir / f"sdes-media-{media_id}.xlsx"
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".part")
    url = f"{BASE_URL}/media/{media_id}/download?inline"
    with client.stream("GET", url) as response:
        response.raise_for_status()
        # Never cache an error or maintenance page under the workbook name.
        if not response.headers.get("content-type", "").startswith(XLSX):
            raise ValueError(f"{url}: not an xlsx file ({response.headers.get('content-type')})")
        with partial.open("wb") as file:
            for chunk in response.iter_bytes():
                file.write(chunk)
    partial.replace(dest)
    return dest
