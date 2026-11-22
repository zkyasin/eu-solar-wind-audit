"""German districts (Kreise and kreisfreie Städte) and Länder from VG5000, the 1:5,000,000
administrative areas of the Bundesamt für Kartographie und Geodäsie (BKG), licence dl-de/by-2-0.

The edition is the state of 1 January 2026: 401 districts, since Hanau left the Main-Kinzig-Kreis
on that day. The district code is the first five digits of the Amtlicher Gemeindeschlüssel (AGS).
"""

import hashlib
import time
from pathlib import Path

import geopandas as gpd
import httpx

EDITION = "2026-01-01"
URL = (
    "https://daten.gdz.bkg.bund.de/produkte/vg/vg5000_0101/2026/vg5000_01-01.utm32s.gpkg.ebenen.zip"
)
# Computed on the download of 2026-11-22; its MD5 matches the .md5 file published next to it.
SHA256 = "ceb0419ed572ec11763e999f97776ba08727a131fc6c8a65a418cc495c310dde"
MEMBER = "vg5000_ebenen_0101/DE_VG5000.gpkg"
RAW_DIR = Path("data/raw/bkg")
TIMEOUT_SECONDS = 120.0
RETRY_DELAYS_SECONDS = (2.0, 10.0)

# GF (Geofaktor) 9 is the part of a unit that has lower levels, that is its land area. On the Land
# level, GF 8 rows are the 12-nautical-mile coastal sea and the Bodensee, which hold no districts.
LAND_AREA = 9


def download_vg5000(client: httpx.Client | None = None, raw_dir: Path = RAW_DIR) -> Path:
    """Path of the pinned VG5000 zip, downloaded and checked unless already cached."""
    dest = raw_dir / f"vg5000_{EDITION}.utm32s.gpkg.ebenen.zip"
    if dest.exists():
        return dest
    if client is None:
        with httpx.Client(timeout=TIMEOUT_SECONDS) as own_client:
            return download_vg5000(own_client, raw_dir)

    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".part")
    # The server sometimes closes the connection mid-transfer.
    for delay in RETRY_DELAYS_SECONDS:
        try:
            digest = _stream_to(client, partial)
            break
        except httpx.TransportError:
            time.sleep(delay)
    else:
        digest = _stream_to(client, partial)
    if digest != SHA256:
        partial.unlink()
        raise ValueError(f"{URL}: SHA-256 {digest}, expected {SHA256}")
    partial.replace(dest)
    return dest


def load_districts(path: Path | None = None) -> gpd.GeoDataFrame:
    """Districts in EPSG:4326: `region` is the 5-digit AGS ("09162"), `name` the geographic name
    and `kind` the type ("Kreisfreie Stadt", "Landkreis"), which tells apart same-name pairs."""
    districts = _read_land_area(path, "vg5000_krs", ["AGS", "GEN", "BEZ", "GF"])
    districts = districts.rename(columns={"AGS": "region", "GEN": "name", "BEZ": "kind"})
    return _checked(districts[["region", "name", "kind", "geometry"]], digits=5)


def load_laender(path: Path | None = None) -> gpd.GeoDataFrame:
    """Länder in EPSG:4326: `region` is the 2-digit AGS ("09" Bayern), `name` the name."""
    laender = _read_land_area(path, "vg5000_lan", ["AGS", "GEN", "GF"])
    laender = laender.rename(columns={"AGS": "region", "GEN": "name"})
    return _checked(laender[["region", "name", "geometry"]], digits=2)


def _stream_to(client: httpx.Client, dest: Path) -> str:
    digest = hashlib.sha256()
    with client.stream("GET", URL) as response:
        response.raise_for_status()
        with dest.open("wb") as file:
            for chunk in response.iter_bytes():
                digest.update(chunk)
                file.write(chunk)
    return digest.hexdigest()


def _read_land_area(path: Path | None, layer: str, columns: list[str]) -> gpd.GeoDataFrame:
    if path is None:
        path = download_vg5000()
    frame = gpd.read_file(f"zip://{path}!{MEMBER}", layer=layer, columns=columns)
    return frame[frame["GF"] == LAND_AREA]


def _checked(units: gpd.GeoDataFrame, digits: int) -> gpd.GeoDataFrame:
    codes = units["region"]
    if not codes.str.fullmatch(rf"\d{{{digits}}}").all() or not codes.is_unique:
        raise ValueError(f"region codes are not unique {digits}-digit keys")
    return units.sort_values("region").reset_index(drop=True).to_crs("EPSG:4326")
