"""Client for the RESTful API of the ENTSO-E Transparency Platform: actual generation per
production type (16.1.B&C), actual total load (6.1.A) and the day-ahead forecast of wind and solar
generation (14.1.D) of a bidding zone.

The API needs a personal security token, read from the environment variable ENTSOE_API_TOKEN. It
is sent in the SECURITY_TOKEN header, as the API's Postman collection allows, so that it never
appears in a URL, an error or a cache file name. One request covers one year at most.

Quantities are average power in MW over each position of a period. The terms of use do not allow
republishing these values: they stay in the local cache, and the project publishes only values
derived from them (docs/data-licences.md).
"""

import itertools
import os
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Literal

import httpx
import numpy as np
import pandas as pd
from pandera.typing import DataFrame

from eu_solar_wind_audit import grid

URL = "https://web-api.tp.entsoe.eu/api"
TOKEN_VARIABLE = "ENTSOE_API_TOKEN"
CACHE_DIR = Path("data/raw/entsoe")
# A request for one day took 14 s on 2026-11-24; a year takes longer.
TIMEOUT_SECONDS = 600.0
RETRY_DELAYS_SECONDS = (10.0, 60.0)
RETRIED_STATUSES = (429, 500, 502, 503, 504)
# Data can be resubmitted for weeks after delivery, so recent years are downloaded again on each
# call. Settled years are cached as received, gaps included: the platform answers slowly.
SETTLED_AFTER = pd.Timedelta(days=30)
# Acknowledgement reason when no data matches a query.
NO_DATA = "999"

# EIC codes of the bidding zones, from the Transparency Platform's "Area List with Energy
# Identification Code (EIC)", updated 2026-09-21:
# https://transparencyplatform.zendesk.com/hc/en-us/articles/15885757676308
Area = Literal["AT", "BE", "DE-LU", "DK1", "DK2", "ES", "FR", "NL", "PL", "PT"]
AREAS: dict[Area, str] = {
    "AT": "10YAT-APG------L",
    "BE": "10YBE----------2",
    "DE-LU": "10Y1001A1001A82H",
    "DK1": "10YDK-1--------W",
    "DK2": "10YDK-2--------M",
    "ES": "10YES-REE------0",
    "FR": "10YFR-RTE------C",
    "NL": "10YNL----------L",
    "PL": "10YPL-AREA-----S",
    "PT": "10YPT-REN------W",
}
Technology = Literal["solar", "wind_onshore", "wind_offshore"]
PRODUCTION_TYPES: dict[Technology, str] = {
    "solar": "B16",
    "wind_onshore": "B19",
    "wind_offshore": "B18",
}
RESOLUTIONS = {
    "PT15M": pd.Timedelta(minutes=15),
    "PT30M": pd.Timedelta(minutes=30),
    "PT60M": pd.Timedelta(hours=1),
}
QUARTER = pd.Timedelta(minutes=15)
# Element that a time series sets for the area of its quantity: generation flows into a bidding
# zone, load out of it. Generation documents also hold consumption series, such as pumping.
GENERATION_DOMAIN = "inBiddingZone_Domain.mRID"
LOAD_DOMAIN = "outBiddingZone_Domain.mRID"


class EntsoeError(Exception):
    """The platform refused a query; the message is the reason it gave."""


def make_client() -> httpx.Client:
    """An HTTP client that sends the security token of ENTSOE_API_TOKEN with each request."""
    token = os.environ.get(TOKEN_VARIABLE)
    if not token:
        raise EntsoeError(f"set {TOKEN_VARIABLE} to an ENTSO-E Transparency Platform token")
    return httpx.Client(timeout=TIMEOUT_SECONDS, headers={"SECURITY_TOKEN": token})


def fetch_hourly_generation(
    area: Area,
    technology: Technology,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    client: httpx.Client | None = None,
    cache_dir: Path = CACHE_DIR,
) -> DataFrame[grid.Hourly]:
    """Actual generation of `technology` in `area` on the UTC hours of [start, end)."""
    query = {
        "documentType": "A75",
        "processType": "A16",
        "psrType": PRODUCTION_TYPES[technology],
        "in_Domain": AREAS[area],
    }
    return _hourly(query, GENERATION_DOMAIN, start, end, client, cache_dir)


def fetch_hourly_load(
    area: Area,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    client: httpx.Client | None = None,
    cache_dir: Path = CACHE_DIR,
) -> DataFrame[grid.Hourly]:
    """Actual total load of `area` on the UTC hours of [start, end)."""
    query = {"documentType": "A65", "processType": "A16", "outBiddingZone_Domain": AREAS[area]}
    return _hourly(query, LOAD_DOMAIN, start, end, client, cache_dir)


def fetch_hourly_forecast(
    area: Area,
    technology: Technology,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    client: httpx.Client | None = None,
    cache_dir: Path = CACHE_DIR,
) -> DataFrame[grid.Hourly]:
    """Day-ahead forecast of the generation of `technology` in `area` on the UTC hours of
    [start, end)."""
    query = {
        "documentType": "A69",
        "processType": "A01",
        "psrType": PRODUCTION_TYPES[technology],
        "in_Domain": AREAS[area],
    }
    return _hourly(query, GENERATION_DOMAIN, start, end, client, cache_dir)


def parse_energy(content: bytes, domain: str) -> pd.Series:
    """Energy in MWh of each quarter-hour of the time series of a market document that set the
    `domain` element, indexed by the UTC start of the quarter-hour. An acknowledgement that no
    data matches the query gives an empty series; any other acknowledgement raises EntsoeError.

    A point is the average power over its position. Curve type A01 lists every position; A03
    lists only the positions where the power changes, each block lasting until the next one
    (ENTSO-E, "The introduction of different time series possibilities (CurveType) within ENTSO-E
    electronic documents", version 1.4, section 4.3). Positions missing from a period, and
    intervals between disjoint periods, are gaps (section 5)."""
    root = ET.fromstring(content)
    namespace = {"m": root.tag[1:].split("}")[0]}
    if root.tag.endswith("Acknowledgement_MarketDocument"):
        reasons = root.findall("m:Reason", namespace)
        codes = [_text(reason, "m:code", namespace) for reason in reasons]
        if codes == [NO_DATA]:
            return _energy([])
        texts = [reason.findtext("m:text", default="", namespaces=namespace) for reason in reasons]
        raise EntsoeError(f"reasons {', '.join(codes)}: {' '.join(texts)}")
    periods = []
    for series in root.findall("m:TimeSeries", namespace):
        if series.find(f"m:{domain}", namespace) is None:
            continue
        unit = _text(series, "m:quantity_Measure_Unit.name", namespace)
        if unit != "MAW":
            raise EntsoeError(f"unexpected unit {unit}")
        curve = series.findtext("m:curveType", default="A01", namespaces=namespace)
        for period in series.findall("m:Period", namespace):
            periods.append(_period_energy(period, curve, namespace))
    return _energy(periods)


def _hourly(
    query: dict[str, str],
    domain: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    client: httpx.Client | None,
    cache_dir: Path,
) -> DataFrame[grid.Hourly]:
    grid.check_period(start, end)
    if client is None:
        with make_client() as own_client:
            return _hourly(query, domain, start, end, own_client, cache_dir)
    parts = []
    for part_start, part_end in _years(start, end):
        energy = _answer(client, query, domain, part_start, part_end, cache_dir)
        # A period can extend beyond the query; neighbouring requests would then overlap.
        parts.append(energy[(energy.index >= part_start) & (energy.index < part_end)])
    hourly = grid.hourly_from_period_start_energy(_energy(parts), "15min")
    return grid.Hourly.validate(grid.on_hours(hourly, start, end))


def _years(start: pd.Timestamp, end: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """[start, end) cut at the UTC year starts, so that each part fits in one request."""
    cuts = [pd.Timestamp(year, 1, 1, tz="UTC") for year in range(start.year + 1, end.year + 1)]
    return list(itertools.pairwise([start, *(cut for cut in cuts if cut < end), end]))


def _answer(
    client: httpx.Client,
    query: dict[str, str],
    domain: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cache_dir: Path,
) -> pd.Series:
    """Quarter-hour energies of the answer to `query` over [start, end), read from the cache
    when it holds them. A settled answer is cached once it parses, so a refusal never is."""
    params = {**query, "periodStart": f"{start:%Y%m%d%H%M}", "periodEnd": f"{end:%Y%m%d%H%M}"}
    path = cache_dir / f"{'_'.join(params.values())}.xml"
    if path.exists():
        return parse_energy(path.read_bytes(), domain)
    content = _get(client, params)
    energy = parse_energy(content, domain)
    if end + SETTLED_AFTER <= pd.Timestamp.now(tz="UTC"):
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write under a temporary name so that an interrupted write never looks complete.
        partial = path.with_name(path.name + ".part")
        partial.write_bytes(content)
        partial.replace(path)
    return energy


def _get(client: httpx.Client, params: dict[str, str]) -> bytes:
    """Retries timeouts, connection errors, rate limits and server errors. A refused query
    answers with an acknowledgement document, which parse_energy reads."""
    for delay in RETRY_DELAYS_SECONDS:
        try:
            response = client.get(URL, params=params)
            if response.status_code not in RETRIED_STATUSES:
                break
        except httpx.TransportError:
            pass
        time.sleep(delay)
    else:
        response = client.get(URL, params=params)
    if response.status_code == 400 and b"Acknowledgement_MarketDocument" in response.content:
        return response.content
    response.raise_for_status()
    return response.content


def _period_energy(period: ET.Element, curve: str, namespace: dict[str, str]) -> pd.Series:
    """Energy in MWh of each quarter-hour of one period."""
    start = pd.Timestamp(_text(period, "m:timeInterval/m:start", namespace))
    end = pd.Timestamp(_text(period, "m:timeInterval/m:end", namespace))
    resolution = _text(period, "m:resolution", namespace)
    if resolution not in RESOLUTIONS:
        raise EntsoeError(f"unexpected resolution {resolution}")
    step = RESOLUTIONS[resolution]
    points = {
        int(_text(point, "m:position", namespace)): float(_text(point, "m:quantity", namespace))
        for point in period.findall("m:Point", namespace)
    }
    power = pd.Series(points, dtype=float).reindex(range(1, (end - start) // step + 1))
    if curve == "A03":
        power = power.ffill()
    elif curve != "A01":
        raise EntsoeError(f"unexpected curve type {curve}")
    quarters = pd.date_range(start, end, freq=QUARTER, inclusive="left", unit="ns")
    energy = np.repeat(power.to_numpy(), step // QUARTER) * (QUARTER / grid.HOUR)
    return pd.Series(energy, index=quarters)


def _text(element: ET.Element, path: str, namespace: dict[str, str]) -> str:
    text = element.findtext(path, namespaces=namespace)
    if text is None:
        raise EntsoeError(f"{path.replace('m:', '')} is missing")
    return text


def _energy(parts: list[pd.Series]) -> pd.Series:
    """The quarter-hour energies of `parts` in one series with a UTC index."""
    if not parts:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([], dtype="datetime64[ns, UTC]"))
    energy: pd.Series = pd.concat(parts)
    return energy.sort_index()
