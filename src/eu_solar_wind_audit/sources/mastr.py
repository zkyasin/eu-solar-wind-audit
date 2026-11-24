"""Installed solar and wind capacity of Germany by district (Kreis), from the full public export of
the Marktstammdatenregister (MaStR) of the Bundesnetzagentur (licence dl-de/by-2-0).

The daily export gets a new file name each day, which the export documentation (revision 26.1.2,
section 2.3) says is meant to prevent automated retrieval. Quarterly exports of a fixed date
(Stichtag) are kept under a stable name for analyses (section 3.3), so one of them is pinned here
with its checksum.

Powers are in kW in the export and in MW in the outputs. For solar units the gross power
(Bruttoleistung) is the module power in kWp, so DC (MaStR registration help for solar units), and
the net rated power (Nettonennleistung) is the lesser of the gross power and the inverter active
power, so AC: this holds for every solar unit with an inverter power in the pinned export.

The control area of a unit is that of the grid connection points of its location
(Netzanschlusspunkte), null when its location has none or has points in several control areas.
"""

import hashlib
import json
import re
import time
import zipfile
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import ClassVar
from xml.etree import ElementTree

import httpx
import numpy as np
import pandas as pd
import pandera.pandas as pa
from pandera.typing import DataFrame, Series

EXPORT_URL = (
    "https://download.marktstammdatenregister.de/Stichtag/Gesamtdatenexport_20261001_26.1.zip"
)
# Computed on the download of 2026-11-22.
SHA256 = "3084c3849082b3fda1e21e0067e8b678542dc6f35980bf1edb63996d2578ebbf"
CACHE_DIR = Path("data/raw/mastr")
OUTPUT_DIR = Path("data/processed/mastr")
ZIP_TYPES = ("application/zip", "application/x-zip-compressed")
TIMEOUT = httpx.Timeout(30, read=300)
RETRY_DELAYS_SECONDS = (10.0, 60.0)

# The solar table is split into numbered members; the EEG plant tables (AnlagenEegSolar_*.xml) are
# a different object.
SOLAR_MEMBER = re.compile(r"EinheitenSolar(_\d+)?\.xml")
WIND_MEMBER = "EinheitenWind.xml"
CATALOGUE_MEMBER = "Katalogwerte.xml"
CONNECTION_MEMBER = re.compile(r"Netzanschlusspunkte(_\d+)?\.xml")

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
    888: ("Windkraft an Land", "wind_onshore"),
    889: ("Windkraft auf See", "wind_offshore"),
}
SEA_CODES = {639: ("Ostsee", "baltic"), 640: ("Nordsee", "north")}
# The control areas of the four German transmission system operators.
CONTROL_AREA_CODES = {
    1000001: ("50Hertz", "50Hertz"),
    1000010: ("Amprion", "Amprion"),
    1001564: ("TenneT", "TenneT"),
    1001572: ("TransnetBW", "TransnetBW"),
}
CONTROL_AREAS = tuple(name for _, name in CONTROL_AREA_CODES.values())
# Voltage levels of the grid connection points. In Germany the four transmission system operators
# run the extra-high voltage grid; the 110 kV grid and below are distribution grids.
VOLTAGE_CODES = {
    348: ("Höchstspannung", "extra_high"),
    349: ("Umspannebene Höchstspannung/Hochspannung", "extra_high_to_high"),
    350: ("Hochspannung", "high"),
    351: ("Umspannebene Hochspannung/Mittelspannung", "high_to_medium"),
    352: ("Mittelspannung", "medium"),
    353: ("Umspannebene Mittelspannung/Niederspannung", "medium_to_low"),
    354: ("Niederspannung (= Hausanschluss/Haushaltsstrom)", "low"),
}
TRANSMISSION_VOLTAGES = ("extra_high", "extra_high_to_high")
# Main orientation and tilt of the modules of a solar unit, whether a unit feeds all its output
# or only what its owner does not consume, and the limit on its feed-in (EEG 2012 to 2021: 70 %
# of the module power for small units without remote control).
ORIENTATION_CODES = {
    695: ("Nord", "north"),
    696: ("Nord-Ost", "north_east"),
    697: ("Ost", "east"),
    698: ("Süd-Ost", "south_east"),
    699: ("Süd", "south"),
    700: ("Süd-West", "south_west"),
    701: ("West", "west"),
    702: ("Nord-West", "north_west"),
    703: ("nachgeführt", "tracked"),
    704: ("Ost-West", "east_west"),
}
TILT_CODES = {
    3179: ("unter 5 Grad (horizontal)", "below_5"),
    810: ("5 - 20 Grad", "5_to_20"),
    809: ("21 - 40 Grad", "21_to_40"),
    808: ("41 - 60 Grad", "41_to_60"),
    807: ("61 - 89 Grad", "61_to_89"),
    # The catalogue value ends with a space.
    806: ("90 Grad (vertikal) ", "vertical"),
    811: ("Nachgeführt", "tracked"),
}
FEED_IN_CODES = {
    688: ("Volleinspeisung", "full"),
    689: ("Teileinspeisung (einschließlich Eigenverbrauch)", "partial"),
}
FEED_IN_LIMIT_CODES = {
    802: ("Nein", "none"),
    803: ("Ja, auf 70%", "70_percent"),
    804: ("Ja, auf 60%", "60_percent"),
    805: ("Ja, auf 50%", "50_percent"),
    1535: ("Ja, sonstige", "other"),
}

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
    "Einspeisungsart": "feed_in",
    "LokationMaStRNummer": "location_id",
}
CONNECTION_FIELDS = {
    "LokationMaStRNummer": "location_id",
    "RegelzoneNetzanschlusspunkt": "control_area",
    "Spannungsebene": "voltage",
}
SOLAR_FIELDS = {
    **COMMON_FIELDS,
    "Hauptausrichtung": "orientation",
    "HauptausrichtungNeigungswinkel": "tilt",
    "Leistungsbegrenzung": "feed_in_limit",
}
WIND_FIELDS = {
    **COMMON_FIELDS,
    "Nabenhoehe": "hub_height_m",
    "Rotordurchmesser": "rotor_diameter_m",
    "WindAnLandOderAufSee": "location",
    "Seelage": "sea",
    "GebietNachDemFlaechenentwicklungsplanNordsee": "cluster_north",
    "GebietNachDemFlaechenentwicklungsplanOstsee": "cluster_baltic",
    "NameWindpark": "wind_farm",
}
DATES = ["commissioned", "decommissioned", "shutdown_start", "shutdown_end"]
# Plausibility bounds, gross power for solar and net power for wind. The largest units in the
# pinned export are a 200 MWp solar park and 15 MW offshore turbines.
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


def _names(codes: dict[int, tuple[str, str]]) -> list[str]:
    return [name for _, name in codes.values()]


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
    feed_in: Series[str] = pa.Field(nullable=True, isin=_names(FEED_IN_CODES))
    orientation: Series[str] = pa.Field(nullable=True, isin=_names(ORIENTATION_CODES))
    tilt: Series[str] = pa.Field(nullable=True, isin=_names(TILT_CODES))
    feed_in_limit: Series[str] = pa.Field(nullable=True, isin=_names(FEED_IN_LIMIT_CODES))
    hub_height_m: Series[float] = pa.Field(nullable=True)
    rotor_diameter_m: Series[float] = pa.Field(nullable=True)
    # Most units point to a generation location (SEL), some to a consumption location (SVL).
    location_id: Series[str] = pa.Field(nullable=True, str_matches=r"^[A-Z]{3}\d{12}$")
    control_area: Series[str] = pa.Field(nullable=True, isin=_names(CONTROL_AREA_CODES))
    voltage: Series[str] = pa.Field(nullable=True, isin=_names(VOLTAGE_CODES))

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


class SolarFleet(pa.DataFrameModel):
    """DC capacity in MW of solar units by main orientation and tilt class of their modules, and
    its share of the total; units without an orientation or a tilt are in null groups."""

    orientation: Series[str] = pa.Field(nullable=True)
    tilt: Series[str] = pa.Field(nullable=True)
    capacity_mw_dc: Series[float] = pa.Field(ge=0)
    share: Series[float] = pa.Field(ge=0, le=1)

    class Config:
        strict = True


class WindGenerations(pa.DataFrameModel):
    """Onshore wind units commissioned up to the end of `last_commissioning_year`, after the
    previous row, null in the last row: their capacity in MW, their hub height in m weighted by
    capacity, and their specific power, the capacity over the area swept by their rotors."""

    last_commissioning_year: Series[pd.Int64Dtype] = pa.Field(nullable=True)
    capacity_mw: Series[float] = pa.Field(ge=0)
    hub_height_m: Series[float]
    specific_power_w_m2: Series[float]

    class Config:
        strict = True


def solar_fleet(units: pd.DataFrame, as_of: pd.Timestamp) -> DataFrame[SolarFleet]:
    """`SolarFleet` of the solar units in operation at `as_of`, by their gross power."""
    active = _active(units, "solar", as_of)
    capacity = active.groupby(["orientation", "tilt"], dropna=False)["gross_kw"].sum() / 1000
    table = capacity.rename("capacity_mw_dc").reset_index()
    table["share"] = table["capacity_mw_dc"] / table["capacity_mw_dc"].sum()
    table = table.sort_values("capacity_mw_dc", ascending=False, ignore_index=True)
    return SolarFleet.validate(table.astype({"orientation": "str", "tilt": "str"}))


def dc_ac_ratio(units: pd.DataFrame, as_of: pd.Timestamp) -> float:
    """Gross over net power of the solar units in operation at `as_of` that have both: their
    module power over their inverter-limited power."""
    active = _active(units, "solar", as_of).dropna(subset=["gross_kw", "net_kw"])
    return float(active["gross_kw"].sum() / active["net_kw"].sum())


def wind_generations(
    units: pd.DataFrame, as_of: pd.Timestamp, last_years: Sequence[int | None]
) -> DataFrame[WindGenerations]:
    """`WindGenerations` of the onshore wind units in operation at `as_of`, one row per window of
    `last_years`, the last of which is None. Hub heights and rotor diameters count where given."""
    active = _active(units, "wind_onshore", as_of)
    year = active["commissioned"].dt.year.to_numpy()
    ends = [np.inf if last is None else float(last) for last in last_years]
    rows = []
    for last, low, high in zip(last_years, [-np.inf, *ends], ends, strict=False):
        window = active[(year > low) & (year <= high)]
        with_hub = window.dropna(subset=["hub_height_m"])
        with_rotor = window.dropna(subset=["rotor_diameter_m"])
        swept_m2 = np.pi * (with_rotor["rotor_diameter_m"] / 2) ** 2
        rows.append(
            {
                "last_commissioning_year": last,
                "capacity_mw": window["net_kw"].sum() / 1000,
                "hub_height_m": float(
                    np.average(with_hub["hub_height_m"], weights=with_hub["net_kw"])
                ),
                "specific_power_w_m2": float(with_rotor["net_kw"].sum() * 1000 / swept_m2.sum()),
            }
        )
    table = pd.DataFrame(rows).astype({"last_commissioning_year": "Int64"})
    return WindGenerations.validate(table)


def download_export(client: httpx.Client, cache_dir: Path = CACHE_DIR) -> Path:
    """Path of the pinned export zip, downloaded and checked unless already cached.

    An interrupted download is resumed from its .part file. Once the checksum matches, a JSON file
    next to the zip records the URL, retrieval time, file name, size and SHA-256; a zip without
    that record is checked again.
    """
    dest = cache_dir / EXPORT_URL.rsplit("/", 1)[1]
    record = _metadata_path(dest)
    if record.exists():
        return dest
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        partial = dest.with_name(dest.name + ".part")
        for delay in (*RETRY_DELAYS_SECONDS, None):
            try:
                _resume(client, EXPORT_URL, partial)
                break
            except httpx.TransportError:
                if delay is None:
                    raise
                time.sleep(delay)
        # Renamed before hashing: a complete .part file left behind cannot be resumed, since the
        # server rejects a range past the end of the file.
        partial.replace(dest)

    description = _describe(dest)
    if description["sha256"] != SHA256:
        dest.unlink()
        raise ValueError(f"{EXPORT_URL}: SHA-256 {description['sha256']}, expected {SHA256}")
    metadata = {
        "url": EXPORT_URL,
        "retrieved_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        **description,
    }
    record.write_text(json.dumps(metadata, indent=2) + "\n")
    return dest


def read_units(path: Path) -> DataFrame[Units]:
    """Solar and wind units of the export, one row per unit, with their control area."""
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        solar = sorted(name for name in names if SOLAR_MEMBER.fullmatch(name))
        connections = sorted(name for name in names if CONNECTION_MEMBER.fullmatch(name))
        if not solar or not connections or {WIND_MEMBER, CATALOGUE_MEMBER} - set(names):
            raise ValueError(f"{path}: missing solar, wind, connection or catalogue members")
        catalogue = _read_catalogue(archive)
        frames = [
            _solar_units(_read_member(archive, name, "EinheitSolar", SOLAR_FIELDS))
            for name in solar
        ]
        frames.append(
            _wind_units(_read_member(archive, WIND_MEMBER, "EinheitWind", WIND_FIELDS), catalogue)
        )
        points = pd.concat(
            _read_member(archive, name, "Netzanschlusspunkt", CONNECTION_FIELDS)
            for name in connections
        )
    text = ["sea", "cluster", "wind_farm", "feed_in", "orientation", "tilt", "feed_in_limit"]
    units = pd.concat(frames, ignore_index=True).astype(dict.fromkeys(text, "str"))
    for column, codes in [("control_area", CONTROL_AREA_CODES), ("voltage", VOLTAGE_CODES)]:
        found = _location_values(points, column, codes)
        units[column] = units["location_id"].map(found).astype("str")
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


def transmission_share(
    units: pd.DataFrame, technologies: Sequence[str], as_of: pd.Timestamp
) -> float:
    """Share of the net power of the units of `technologies` in operation at `as_of` that feeds
    the transmission grid: offshore wind, which the transmission system operators connect whatever
    the voltage of its connection point, and the units at TRANSMISSION_VOLTAGES. Units on land
    whose location has no single voltage are left out."""
    active = pd.concat([_active(units, technology, as_of) for technology in technologies])
    offshore = active["technology"].eq("wind_offshore")
    known = active[offshore | active["voltage"].notna()]
    transmission = known["technology"].eq("wind_offshore") | known["voltage"].isin(
        TRANSMISSION_VOLTAGES
    )
    return float(known.loc[transmission, "net_kw"].sum() / known["net_kw"].sum())


def offshore_wind(units: pd.DataFrame, as_of: pd.Timestamp) -> DataFrame[OffshoreWind]:
    """Offshore wind units in operation at `as_of`, by sea, development plan cluster and ERA5 cell
    (coordinates rounded to the nearest 0.25 degree)."""
    active = _active(units, "wind_offshore", as_of).assign(
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
    """Download the pinned export unless cached, then write its tables with `write_capacity`.
    Returns the panel path."""
    if client is None:
        with httpx.Client(timeout=TIMEOUT, follow_redirects=True) as own_client:
            return extract_capacity(
                start=start, cache_dir=cache_dir, output_dir=output_dir, client=own_client
            )
    export = download_export(client, cache_dir)
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
        raise ValueError(f"{path.name}: not an export file name")
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
    tables = (
        STATUS_CODES,
        WIND_LOCATION_CODES,
        SEA_CODES,
        ORIENTATION_CODES,
        TILT_CODES,
        FEED_IN_CODES,
        FEED_IN_LIMIT_CODES,
        CONTROL_AREA_CODES,
        VOLTAGE_CODES,
    )
    expected = {code: name for codes in tables for code, (name, _) in codes.items()}
    changed = {
        code: catalogue.get(code) for code, name in expected.items() if catalogue.get(code) != name
    }
    if changed:
        raise ValueError(f"catalogue values changed: {changed}")
    return catalogue


def _solar_units(raw: pd.DataFrame) -> pd.DataFrame:
    return _common(raw).assign(
        technology="solar",
        sea=None,
        cluster=None,
        wind_farm=None,
        orientation=_decode(raw["orientation"], ORIENTATION_CODES),
        tilt=_decode(raw["tilt"], TILT_CODES),
        feed_in_limit=_decode(raw["feed_in_limit"], FEED_IN_LIMIT_CODES),
        hub_height_m=float("nan"),
        rotor_diameter_m=float("nan"),
    )


def _wind_units(raw: pd.DataFrame, catalogue: dict[int, str]) -> pd.DataFrame:
    cluster = raw["cluster_north"].fillna(raw["cluster_baltic"])
    return _common(raw).assign(
        technology=_decode(raw["location"], WIND_LOCATION_CODES),
        sea=_decode(raw["sea"], SEA_CODES),
        cluster=cluster.map(lambda code: catalogue[int(code)], na_action="ignore"),
        wind_farm=raw["wind_farm"],
        orientation=None,
        tilt=None,
        feed_in_limit=None,
        hub_height_m=raw["hub_height_m"].astype("float64"),
        rotor_diameter_m=raw["rotor_diameter_m"].astype("float64"),
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
            # units carry only the first five (daily export of 2026-11-22).
            "district": key.where(key.str.fullmatch(r"\d{5}|\d{8}", na=False))
            .str[:5]
            .replace(FORMER_DISTRICTS),
            "longitude": raw["longitude"].astype("float64"),
            "latitude": raw["latitude"].astype("float64"),
            "feed_in": _decode(raw["feed_in"], FEED_IN_CODES),
            "location_id": raw["location_id"],
        }
    )
    return frame


def _location_values(
    points: pd.DataFrame, column: str, codes: dict[int, tuple[str, str]]
) -> "pd.Series[str]":
    """The decoded `column` of each location of the grid connection `points`, for the locations
    whose points all give the same value."""
    found = points[["location_id", column]].assign(**{column: _decode(points[column], codes)})
    found = found.dropna().drop_duplicates()
    single = found[~found["location_id"].duplicated(keep=False)]
    values: pd.Series[str] = single.set_index("location_id")[column]
    return values


def _decode(codes: "pd.Series[str]", table: dict[int, tuple[str, str]]) -> "pd.Series[str]":
    names = {str(code): name for code, (_, name) in table.items()}
    unknown = set(codes.dropna()) - names.keys()
    if unknown:
        raise ValueError(f"unknown codes: {sorted(unknown)}")
    return codes.map(names)


def _active(units: pd.DataFrame, technology: str, as_of: pd.Timestamp) -> pd.DataFrame:
    """Units of `technology` in operation at `as_of`: commissioned, and not decommissioned, by
    then."""
    counted = _countable(units[units["technology"].eq(technology)])
    return counted[(counted["commissioned"] <= as_of) & ~(counted["decommissioned"] <= as_of)]


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
