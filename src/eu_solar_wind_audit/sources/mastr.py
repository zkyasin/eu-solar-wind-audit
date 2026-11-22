"""Installed solar and wind capacity of Germany by district (Kreis), from the full public export of
the Marktstammdatenregister (MaStR) of the Bundesnetzagentur (licence dl-de/by-2-0).

Powers are in kW in the export and in MW in the outputs. For solar units the gross power
(Bruttoleistung) is the module power in kWp, so DC (MaStR registration help for solar units), and
the net rated power (Nettonennleistung) is the lesser of the gross power and the inverter active
power, so AC: this holds for every solar unit with an inverter power in the export of 2026-11-22.
"""

import hashlib
import json
import re
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import ClassVar
from xml.etree import ElementTree

import httpx
import pandas as pd
import pandera.pandas as pa
from pandera.typing import DataFrame, Series

DOWNLOAD_PAGE = "https://www.marktstammdatenregister.de/MaStR/Datendownload"
CACHE_DIR = Path("data/raw/mastr")
OUTPUT_DIR = Path("data/processed/mastr")
# The daily export gets a new name each day; the quarterly exports are under /Stichtag/.
EXPORT_LINK = re.compile(
    r'href="(https://download\.marktstammdatenregister\.de/Gesamtdatenexport_\d{8}_[\d.]+\.zip)"'
)
ZIP_TYPES = ("application/zip", "application/x-zip-compressed")
TIMEOUT = httpx.Timeout(30, read=300)
RETRY_DELAYS_SECONDS = (10.0, 60.0)

# The solar table is split into numbered members; the EEG plant tables (AnlagenEegSolar_*.xml) are
# a different object.
SOLAR_MEMBER = re.compile(r"EinheitenSolar(_\d+)?\.xml")
WIND_MEMBER = "EinheitenWind.xml"
CATALOGUE_MEMBER = "Katalogwerte.xml"

TECHNOLOGIES = ("solar", "wind_onshore", "wind_offshore")
STATUSES = ("planned", "operating", "temporarily_shut_down", "decommissioned")

# Catalogue values of the export, checked against Katalogwerte.xml at each read.
STATUS_CODES = {
    31: ("In Planung", "planned"),
    35: ("In Betrieb", "operating"),
    37: ("Vorübergehend stillgelegt", "temporarily_shut_down"),
    38: ("Endgültig stillgelegt", "decommissioned"),
}
WIND_LOCATION_CODES = {
    888: ("Windenergie an Land", "wind_onshore"),
    889: ("Windenergie auf See", "wind_offshore"),
}
SEA_CODES = {639: ("Ostsee", "baltic"), 640: ("Nordsee", "north")}

COMMON_FIELDS = {
    "EinheitMastrNummer": "unit_id",
    "EinheitBetriebsstatus": "status",
    "Bruttoleistung": "gross_kw",
    "Nettonennleistung": "net_kw",
    "Inbetriebnahmedatum": "commissioned",
    "DatumEndgueltigeStilllegung": "decommissioned",
    "DatumBeginnVoruebergehendeStilllegung": "shutdown_start",
    "DatumWiederaufnahmeBetrieb": "shutdown_end",
    "Gemeindeschluessel": "municipality_key",
    "Laengengrad": "longitude",
    "Breitengrad": "latitude",
}
WIND_FIELDS = {
    **COMMON_FIELDS,
    "WindAnLandOderAufSee": "location",
    "Seelage": "sea",
    "GebietNachDemFlaechenentwicklungsplanNordsee": "cluster_north",
    "GebietNachDemFlaechenentwicklungsplanOstsee": "cluster_baltic",
    "NameWindpark": "wind_farm",
}
DATES = ["commissioned", "decommissioned", "shutdown_start", "shutdown_end"]
# Plausibility bounds, gross power for solar and net power for wind. The largest units in the
# export of 2026-11-22 are a 200 MWp solar park and 15 MW offshore turbines.
MAX_KW = {"solar": 200_000.0, "wind_onshore": 20_000.0, "wind_offshore": 20_000.0}
# Districts merged since: Göttingen and Osterode am Harz into the new Göttingen district
# (2016-11-01), Eisenach into the Wartburgkreis (2021-07-01). Some units still carry former keys.
FORMER_DISTRICTS = {"03152": "03159", "03156": "03159", "16056": "16063"}
# Earlier dates are placeholders, such as 1900-01-01.
EARLIEST_COMMISSIONING = pd.Timestamp("1950-01-01")
# The evaluation period starts in 2019.
PANEL_START = pd.Timestamp("2019-01-31")
# ERA5 grid spacing in degrees.
CELL_DEGREES = 0.25


class Units(pa.DataFrameModel):
    unit_id: Series[str] = pa.Field(unique=True, str_matches=r"^SEE\d{12}$")
    technology: Series[str] = pa.Field(isin=TECHNOLOGIES)
    status: Series[str] = pa.Field(isin=STATUSES)
    # Not checked for sign or range: implausible_units reports such values.
    gross_kw: Series[float] = pa.Field(nullable=True)
    net_kw: Series[float] = pa.Field(nullable=True)
    commissioned: Series[pa.DateTime] = pa.Field(nullable=True)
    decommissioned: Series[pa.DateTime] = pa.Field(nullable=True)
    shutdown_start: Series[pa.DateTime] = pa.Field(nullable=True)
    shutdown_end: Series[pa.DateTime] = pa.Field(nullable=True)
    municipality_key: Series[str] = pa.Field(nullable=True)
    district: Series[str] = pa.Field(nullable=True, str_matches=r"^\d{5}$")
    longitude: Series[float] = pa.Field(nullable=True)
    latitude: Series[float] = pa.Field(nullable=True)
    sea: Series[str] = pa.Field(nullable=True, isin=["north", "baltic"])
    cluster: Series[str] = pa.Field(nullable=True)
    wind_farm: Series[str] = pa.Field(nullable=True)

    class Config:
        strict = True


class MonthlyCapacity(pa.DataFrameModel):
    month_end: Series[pa.DateTime]
    # Null for offshore wind and for units without a municipality key.
    district: Series[str] = pa.Field(nullable=True, str_matches=r"^\d{5}$")
    technology: Series[str] = pa.Field(isin=TECHNOLOGIES)
    capacity_mw_ac: Series[float] = pa.Field(ge=0)
    # Solar only.
    capacity_mw_dc: Series[float] = pa.Field(nullable=True, ge=0)

    @pa.check("month_end")
    @classmethod
    def last_day_of_month(cls, month_end: "pd.Series[pd.Timestamp]") -> "pd.Series[bool]":
        return month_end.dt.is_month_end & (month_end == month_end.dt.normalize())

    class Config:
        strict = True
        unique: ClassVar[list[str]] = ["month_end", "district", "technology"]


class OffshoreWind(pa.DataFrameModel):
    sea: Series[str] = pa.Field(isin=["north", "baltic"])
    cluster: Series[str] = pa.Field(nullable=True)
    cell_latitude: Series[float] = pa.Field(nullable=True)
    cell_longitude: Series[float] = pa.Field(nullable=True)
    wind_farms: Series[str]
    units: Series[int] = pa.Field(ge=1)
    capacity_mw: Series[float]
    first_commissioning_month: Series[pa.DateTime]
    last_commissioning_month: Series[pa.DateTime]

    class Config:
        strict = True


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
    zip records the URL, retrieval time, file name, size and SHA-256.
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

    # Renamed before hashing: a complete .part file left behind cannot be resumed, since the server
    # rejects a range past the end of the file.
    partial.replace(dest)
    metadata = {
        "url": url,
        "retrieved_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        **_describe(dest),
    }
    _metadata_path(dest).write_text(json.dumps(metadata, indent=2) + "\n")
    return dest


def read_units(path: Path) -> DataFrame[Units]:
    """Solar and wind units of the export, one row per unit."""
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        solar = sorted(name for name in names if SOLAR_MEMBER.fullmatch(name))
        if not solar or WIND_MEMBER not in names or CATALOGUE_MEMBER not in names:
            raise ValueError(f"{path}: missing solar, wind or catalogue members")
        catalogue = _read_catalogue(archive)
        frames = [
            _solar_units(_read_member(archive, name, "EinheitSolar", COMMON_FIELDS))
            for name in solar
        ]
        frames.append(
            _wind_units(_read_member(archive, WIND_MEMBER, "EinheitWind", WIND_FIELDS), catalogue)
        )
    units = pd.concat(frames, ignore_index=True).astype(
        {"sea": "str", "cluster": "str", "wind_farm": "str"}
    )
    return Units.validate(units[list(Units.to_schema().columns)])


def monthly_capacity(
    units: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp
) -> DataFrame[MonthlyCapacity]:
    """Capacity at each month end from `start` to `end`, by district and technology.

    A unit counts from the end of its commissioning month until its final decommissioning: it is
    counted at a month end on or after its commissioning date and before its decommissioning date.
    Planned units have no commissioning date and never count, nor do units decommissioned before
    their commissioning date. Temporarily shut down units keep counting: the export holds only
    their latest shutdown, and they remain installed. Recent months are incomplete: units must be
    registered within a month of commissioning (MaStRV section 5(5)), and some are registered later.
    """
    counted = _countable(units)
    starts = _events(counted, "commissioned", sign=1.0)
    ends = _events(counted[counted["decommissioned"].notna()], "decommissioned", sign=-1.0)
    events = pd.concat([starts, ends], ignore_index=True)
    # Group keys cannot be null; "" stands for no district until the end.
    events["district"] = events["district"].fillna("")
    changes = events.groupby(["month_end", "district", "technology"])[
        ["capacity_mw_ac", "capacity_mw_dc"]
    ].sum()

    first = min(changes.index.get_level_values("month_end").min(), start)
    months = pd.date_range(first, end, freq="ME", name="month_end")
    wide = changes.unstack(["district", "technology"], fill_value=0.0)
    # Powers have at most 3 decimals in kW, so 6 in MW; rounding removes the residue of sums.
    cumulative = wide.reindex(months, fill_value=0.0).cumsum().loc[start:].round(6)
    panel = cumulative.stack(["district", "technology"], future_stack=True).reset_index()
    panel["district"] = panel["district"].replace("", None)
    panel.loc[panel["technology"].ne("solar"), "capacity_mw_dc"] = float("nan")
    panel["month_end"] = panel["month_end"].astype("datetime64[ns]")
    panel = panel.sort_values(["month_end", "technology", "district"], ignore_index=True)
    return MonthlyCapacity.validate(panel)


def offshore_wind(units: pd.DataFrame, as_of: pd.Timestamp) -> DataFrame[OffshoreWind]:
    """Offshore wind units in operation at `as_of`, by sea, development plan cluster and ERA5 cell
    (coordinates rounded to the nearest 0.25 degree)."""
    offshore = _countable(units[units["technology"].eq("wind_offshore")])
    active = offshore[
        (offshore["commissioned"] <= as_of) & ~(offshore["decommissioned"] <= as_of)
    ].assign(
        cell_latitude=lambda frame: _to_cell(frame["latitude"]),
        cell_longitude=lambda frame: _to_cell(frame["longitude"]),
        commissioning_month=lambda frame: frame["commissioned"] + pd.offsets.MonthEnd(0),
    )
    groups = active.groupby(["sea", "cluster", "cell_latitude", "cell_longitude"], dropna=False)
    table = groups.agg(
        wind_farms=("wind_farm", lambda names: "; ".join(sorted(set(names.dropna())))),
        units=("unit_id", "size"),
        capacity_mw=("net_kw", "sum"),
        first_commissioning_month=("commissioning_month", "min"),
        last_commissioning_month=("commissioning_month", "max"),
    ).reset_index()
    table["capacity_mw"] = table["capacity_mw"] / 1000
    return OffshoreWind.validate(table)


def implausible_units(units: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """Units with an impossible value, one row per unit and problem."""
    maximum = units["technology"].map(MAX_KW)
    capped = units["gross_kw"].where(units["technology"].eq("solar"), units["net_kw"])
    district_in_germany = units["district"].str.match(r"^(0[1-9]|1[0-6])\d{3}$", na=False)
    checks = {
        "power not positive": ~(units["net_kw"] > 0) | ~(units["gross_kw"] > 0),
        "net power above gross power": units["net_kw"] > units["gross_kw"],
        "power above the plausible maximum": capped > maximum,
        "commissioning date in the future": units["commissioned"] > as_of,
        "commissioning date before 1950": units["commissioned"] < EARLIEST_COMMISSIONING,
        "decommissioning date in the future": units["decommissioned"] > as_of,
        "decommissioned before commissioning": units["decommissioned"] < units["commissioned"],
        "decommissioned without a date": units["status"].eq("decommissioned")
        & units["decommissioned"].isna(),
        "not planned and without a commissioning date": units["status"].ne("planned")
        & units["commissioned"].isna(),
        "municipality key malformed": units["municipality_key"].notna()
        & ~units["municipality_key"].str.fullmatch(r"\d{5}|\d{8}", na=False),
        "district outside the German state codes": units["district"].notna() & ~district_in_germany,
        "on land without a municipality key": units["technology"].ne("wind_offshore")
        & units["municipality_key"].isna(),
    }
    flagged = [units[mask].assign(problem=problem) for problem, mask in checks.items()]
    return pd.concat(flagged, ignore_index=True)


def extract_capacity(
    *,
    start: pd.Timestamp = PANEL_START,
    cache_dir: Path = CACHE_DIR,
    output_dir: Path = OUTPUT_DIR,
    client: httpx.Client | None = None,
) -> Path:
    """Download the current export unless cached, then write its tables with `write_capacity`.
    Returns the panel path."""
    if client is None:
        with httpx.Client(timeout=TIMEOUT, follow_redirects=True) as own_client:
            return extract_capacity(
                start=start, cache_dir=cache_dir, output_dir=output_dir, client=own_client
            )
    export = download_export(client, find_export_url(client), cache_dir)
    return write_capacity(export, output_dir, start=start)


def write_capacity(
    export: Path, output_dir: Path = OUTPUT_DIR, *, start: pd.Timestamp = PANEL_START
) -> Path:
    """Write the monthly capacity panel, the offshore wind table and the unit table of a
    downloaded export as parquet files under `output_dir`, and return the panel path.

    The panel ends at the last month end before the export date. Each parquet file has a JSON
    sidecar with the export file name, its SHA-256, and the retrieval and extraction times.
    """
    export_date = export_date_of(export)
    units = read_units(export)
    end = export_date - pd.offsets.MonthEnd(1)
    outputs: dict[str, pd.DataFrame] = {
        "mastr_capacity_monthly": monthly_capacity(units, start, end),
        "mastr_offshore_wind": offshore_wind(units, end),
        "mastr_units": units,
    }
    record = _metadata_path(export)
    # An export downloaded by hand has no download record: its URL and retrieval time are unknown.
    download = (
        json.loads(record.read_text())
        if record.exists()
        else {"url": None, "retrieved_at": None, **_describe(export)}
    )
    metadata = {
        **download,
        "export_date": export_date.date().isoformat(),
        "extracted_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in outputs.items():
        frame.to_parquet(output_dir / f"{name}.parquet", index=False)
        (output_dir / f"{name}.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return output_dir / "mastr_capacity_monthly.parquet"


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


def _describe(path: Path) -> dict[str, str | int]:
    with path.open("rb") as file:
        sha256 = hashlib.file_digest(file, "sha256").hexdigest()
    return {"file": path.name, "size": path.stat().st_size, "sha256": sha256}


def _read_member(
    archive: zipfile.ZipFile, name: str, record: str, fields: dict[str, str]
) -> pd.DataFrame:
    """Text of the fields of each `record` element, streamed from the member.

    Members are UTF-16 with a byte order mark, which the XML parser reads on its own.
    """
    columns: dict[str, list[str | None]] = {column: [] for column in fields.values()}
    with archive.open(name) as member:
        for _, element in ElementTree.iterparse(member):
            if element.tag == record:
                values = {child.tag: child.text for child in element}
                for tag, column in fields.items():
                    columns[column].append(values.get(tag))
                element.clear()
    return pd.DataFrame(columns, dtype="str")


def _read_catalogue(archive: zipfile.ZipFile) -> dict[int, str]:
    values = _read_member(archive, CATALOGUE_MEMBER, "Katalogwert", {"Id": "id", "Wert": "value"})
    catalogue = dict(zip(values["id"].astype(int), values["value"], strict=True))
    expected = {
        code: name
        for codes in (STATUS_CODES, WIND_LOCATION_CODES, SEA_CODES)
        for code, (name, _) in codes.items()
    }
    changed = {
        code: catalogue.get(code) for code, name in expected.items() if catalogue.get(code) != name
    }
    if changed:
        raise ValueError(f"catalogue values changed: {changed}")
    return catalogue


def _solar_units(raw: pd.DataFrame) -> pd.DataFrame:
    return _common(raw).assign(technology="solar", sea=None, cluster=None, wind_farm=None)


def _wind_units(raw: pd.DataFrame, catalogue: dict[int, str]) -> pd.DataFrame:
    cluster = raw["cluster_north"].fillna(raw["cluster_baltic"])
    return _common(raw).assign(
        technology=_decode(raw["location"], WIND_LOCATION_CODES),
        sea=_decode(raw["sea"], SEA_CODES),
        cluster=cluster.map(lambda code: catalogue[int(code)], na_action="ignore"),
        wind_farm=raw["wind_farm"],
    )


def _common(raw: pd.DataFrame) -> pd.DataFrame:
    key = raw["municipality_key"]
    frame = pd.DataFrame(
        {
            "unit_id": raw["unit_id"],
            "status": _decode(raw["status"], STATUS_CODES),
            "gross_kw": raw["gross_kw"].astype("float64"),
            "net_kw": raw["net_kw"].astype("float64"),
            **{
                column: pd.to_datetime(raw[column], format="%Y-%m-%d").astype("datetime64[ns]")
                for column in DATES
            },
            "municipality_key": key,
            # The key is Land (2 digits), Regierungsbezirk (1), Kreis (2) and Gemeinde (3); some
            # units carry only the first five.
            "district": key.where(key.str.fullmatch(r"\d{5}|\d{8}", na=False))
            .str[:5]
            .replace(FORMER_DISTRICTS),
            "longitude": raw["longitude"].astype("float64"),
            "latitude": raw["latitude"].astype("float64"),
        }
    )
    return frame


def _decode(codes: "pd.Series[str]", table: dict[int, tuple[str, str]]) -> "pd.Series[str]":
    names = {str(code): name for code, (_, name) in table.items()}
    unknown = set(codes.dropna()) - names.keys()
    if unknown:
        raise ValueError(f"unknown codes: {sorted(unknown)}")
    return codes.map(names)


def _countable(units: pd.DataFrame) -> pd.DataFrame:
    return units[
        units["commissioned"].notna()
        & units["status"].ne("planned")
        # The date is required once a unit is decommissioned; without it the end is unknown.
        & ~(units["status"].eq("decommissioned") & units["decommissioned"].isna())
        & ~(units["decommissioned"] < units["commissioned"])
    ]


def _events(units: pd.DataFrame, date: str, sign: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            # The first month end on or after the date.
            "month_end": units[date] + pd.offsets.MonthEnd(0),
            "district": units["district"],
            "technology": units["technology"],
            "capacity_mw_ac": sign * units["net_kw"] / 1000,
            "capacity_mw_dc": sign * units["gross_kw"] / 1000,
        }
    )


def _to_cell(degrees: "pd.Series[float]") -> "pd.Series[float]":
    return (degrees / CELL_DEGREES).round() * CELL_DEGREES
