"""Inputs of the anomaly detectors (evaluation plan, axis 2) for a detection run of one country:
its published series, the raw rows they come from, and what each series is compared with.

A run judges the hours [start, end), both hour starts in UTC. Its inputs cover a margin more on
each side, MARGIN unless the run sets another, so that the windowed statistics of its first and
last days are complete where the data exist; the thresholds of 2019-2022 are fixed without one.
Each published series has an id, `<source>.<area>.<quantity>`, and comes in parts, one per dataset
of its rows: a part cites the producer and the dataset of its rows as its provenance, and
says whether they are real-time values, which their producer replaces later. The raw rows of each
dataset, as fetched, go to the completeness detector, up to the last complete hour of a consolidated
dataset (odre.complete_end); the hourly series are converted from the rows that completeness.on_grid
keeps, so that a value off the grid of its dataset is reported instead of stopping the run. Prices
enter only as the classes of regimes.classify_hours.

The physics model covers onshore wind only, so the detectors that compare published wind with it
skip the hours that contain offshore output (evaluation plan, Deviations). `offshore_starts` dates
these hours for each French area.
"""

from dataclasses import dataclass
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
from pandera.typing import DataFrame

from eu_solar_wind_audit import eligibility, grid, physics
from eu_solar_wind_audit.anomalies import completeness
from eu_solar_wind_audit.config import Country
from eu_solar_wind_audit.sources import energy_charts, odre, sdes, smard

# The windowed statistics of a day reach 30 days away from it: the scale of its residual score
# takes the residuals of the 31 days centred on it, each at the level of the 31 days centred on
# its own day (anomalies.residuals).
MARGIN = pd.Timedelta(days=30)

# Mainland France, then the 12 éCO2mix regions, with their names on the SDES panel.
AREAS: dict[str, str | None] = {"FR": None} | {
    f"FR-{code}": name for code, name in odre.REGIONS.items()
}
# Published from odre.WIND_SPLIT_START on.
SPLIT: tuple[odre.RegionalSeries, ...] = ("wind_onshore", "wind_offshore")

# Germany and its four control areas on SMARD. Offshore wind is published where farms connect.
SMARD_AREAS: tuple[smard.Region, ...] = ("DE", "50Hertz", "Amprion", "TenneT", "TransnetBW")

# Saint-Nazaire, the first French offshore farm (odre.FIRST_OFFSHORE_POWER), is in Pays de la Loire.
FIRST_FARM_REGION = "52"
# Elsewhere, offshore output counts from the first quarter in which a region has more than 10 MW
# offshore on the SDES panel, which leaves out the single 10 MW installation that the panel gives
# Pays de la Loire from the end of 2018 on.
OFFSHORE_FROM_MW = 10.0


@dataclass(frozen=True)
class Rows:
    """Rows of one published dataset as fetched, for completeness.completeness, which takes the
    fields of the same names: `values` maps the id of each series of the dataset to its values,
    indexed by UTC stamps, with repeated stamps kept and missing values as NaN. The rows cover the
    hours [start, end) of the run that the dataset publishes."""

    dataset: str
    provenance: str
    provisional: bool
    step: pd.Timedelta
    label: grid.Label
    local_days: str | None
    start: pd.Timestamp
    end: pd.Timestamp
    values: dict[str, pd.Series]


@dataclass(frozen=True)
class Published:
    """The hours of the published series `series` that the rows of one dataset give, with the
    `provenance` of these rows and whether they are real-time values (`provisional`). `hourly` is
    an odre.Hourly frame, with the status of each hour, for éCO2mix, and a grid.Hourly frame
    otherwise."""

    series: str
    provenance: str
    provisional: bool
    hourly: pd.DataFrame


@dataclass(frozen=True)
class SeriesInputs:
    """A published series of `technology`, one of solar, wind, wind_onshore and wind_offshore, and
    what the detectors compare it with, over the hours of its DetectionInputs: `published`, its
    parts by dataset in time order; `modelled`, the uncalibrated physics.Modelled output of its
    area, None without a model; `capacity_mw`, the AC capacity in MW behind the series at the start
    of each hour, None where none is published; `offshore`, whether each hour contains offshore
    output, which the model does not cover, None outside France."""

    technology: str
    published: tuple[Published, ...]
    modelled: DataFrame[physics.Modelled] | None
    capacity_mw: pd.Series | None
    offshore: pd.Series | None


@dataclass(frozen=True)
class DetectionInputs:
    """Inputs of a detection run of `country` over the hours [start, end): the raw rows of each
    dataset, the series by id, and the price classes of the bidding zone of the country, the
    columns of regimes.classify_hours but the mean price. `hours` are the hours of the run with
    its margin on each side: the parts of each series cover them together, and the model, the
    capacity, the offshore hours and the price classes are given on them."""

    country: str
    start: pd.Timestamp
    end: pd.Timestamp
    hours: pd.DatetimeIndex
    rows: tuple[Rows, ...]
    series: dict[str, SeriesInputs]
    price_classes: pd.DataFrame


def france_inputs(
    client: httpx.Client,
    start: pd.Timestamp,
    end: pd.Timestamp,
    parameters: Country,
    *,
    odre_cache_dir: Path = odre.CACHE_DIR,
    smard_cache_dir: Path = smard.CACHE_DIR,
    era5_cache_dir: Path = physics.ERA5_CACHE_DIR,
    margin: pd.Timedelta = MARGIN,
) -> DetectionInputs:
    """Inputs of a detection run over the hours [start, end) of mainland France, both hour starts
    in UTC, from éCO2mix (ODRÉ), the SDES capacity, ERA5 and the French day-ahead prices.

    The series are the solar and wind output of mainland France, `odre.FR.<technology>`, and of
    the 12 éCO2mix regions, `odre.FR-<code>.<technology>`, with onshore and offshore wind too once
    the hours reach odre.WIND_SPLIT_START. A national series has a consolidated and definitive
    part, then a real-time part from the first hour that its consolidated rows do not reach, the
    two parts that odre.national_hourly joins. A regional series has consolidated and definitive
    rows only.

    Solar is compared with the solar model of its area, wind and onshore wind with the onshore
    model, offshore wind with none: one call of physics.model_france_regions per technology models
    the 12 regions, and mainland France is their sum. The capacity is that of sdes.hourly_capacity
    for the perimeter of the series. Wind and offshore wind contain offshore output from the start
    that `offshore_starts` gives their area, onshore wind and solar never.
    """
    grid.check_period(start, end)
    hours = pd.date_range(
        start - margin, end + margin, freq="h", inclusive="left", unit="ns", name="time"
    )
    first, last = hours[0], hours[-1] + grid.HOUR
    technologies: list[odre.RegionalSeries] = ["solar", "wind"]
    if last > odre.WIND_SPLIT_START:
        technologies += SPLIT
    rows, published = _eco2mix(client, technologies, first, last, odre_cache_dir)

    modelled: dict[str, dict[str, DataFrame[physics.Modelled]]] = {}
    for model in ("solar", "wind_onshore"):
        regional = physics.model_france_regions(model, first, last, parameters, era5_cache_dir)
        modelled[model] = {"FR": _mainland(regional)} | {
            f"FR-{code}": frame for code, frame in regional.items()
        }
    monthly = {
        "solar": physics.load_monthly_capacity("solar", last - grid.HOUR),
        "wind": physics.load_monthly_capacity("wind_onshore", last - grid.HOUR),
    }
    workbook = sdes.download_dashboard(client, physics.SDES_MEDIA["wind_onshore"])
    starts = offshore_starts(sdes.read_capacity(workbook))

    series: dict[str, SeriesInputs] = {}
    for area, region in AREAS.items():
        for technology in technologies:
            solar = technology == "solar"
            with_offshore = technology in ("wind", "wind_offshore")
            series[f"odre.{area}.{technology}"] = SeriesInputs(
                technology=technology,
                published=published[f"odre.{area}.{technology}"],
                modelled=None
                if technology == "wind_offshore"
                else modelled["solar" if solar else "wind_onshore"][area],
                capacity_mw=sdes.hourly_capacity(
                    monthly["solar" if solar else "wind"], technology, first, last, region=region
                ),
                offshore=_offshore(hours, starts.get(area) if with_offshore else None),
            )

    classes = eligibility.price_classes(first, last, client=client, cache_dir=smard_cache_dir)
    return DetectionInputs(
        "FR", start, end, hours, tuple(rows), series, classes.drop(columns="mean_price")
    )


def germany_inputs(
    client: httpx.Client,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    smard_cache_dir: Path = smard.CACHE_DIR,
    energy_charts_cache_dir: Path = energy_charts.CACHE_DIR,
    margin: pd.Timedelta = MARGIN,
) -> DetectionInputs:
    """Inputs of a detection run over the hours [start, end) of Germany, both hour starts in UTC:
    the feed-in of Germany and its four control areas from SMARD, `smard.<area>.<technology>`,
    and the public and total power of Germany from Energy-Charts,
    `energy_charts.DE.<endpoint>.<technology>`, with the price classes of the DE-LU bidding zone.

    There is no German model yet, so no series has a modelled output or an offshore mask. The
    series of Germany as a whole get the Energy-Charts installed capacity of their technology; the
    control areas get none, since no capacity is published for them.
    """
    grid.check_period(start, end)
    hours = pd.date_range(
        start - margin, end + margin, freq="h", inclusive="left", unit="ns", name="time"
    )
    first, last = hours[0], hours[-1] + grid.HOUR
    installed = energy_charts.fetch_installed_capacity(
        "de", client=client, cache_dir=energy_charts_cache_dir
    )

    def capacity_mw(technology: str) -> pd.Series:
        return energy_charts.hourly_capacity(installed, technology, first, last)

    rows: list[Rows] = []
    series: dict[str, SeriesInputs] = {}
    for technology, filter_id in smard.FEED_IN.items():
        dataset = f"smard.{filter_id}"
        values: dict[str, pd.Series] = {}
        for area in SMARD_AREAS:
            if technology == "wind_offshore" and area not in smard.OFFSHORE_REGIONS:
                continue
            series_id = f"smard.{area}.{technology}"
            energy = smard.fetch_series(
                filter_id,
                area,
                "quarterhour",
                first,
                last,
                client=client,
                cache_dir=smard_cache_dir,
            )
            values[series_id] = energy
            hourly = grid.hourly_from_period_start_energy(
                completeness.on_grid(energy, smard.FEED_IN_STEP)
            )
            provenance = f"Bundesnetzagentur | SMARD.de, filter {filter_id}, region {area}"
            series[series_id] = SeriesInputs(
                technology=technology,
                published=(
                    Published(series_id, provenance, False, _on_hours(hourly, first, last)),
                ),
                modelled=None,
                capacity_mw=capacity_mw(technology) if area == "DE" else None,
                offshore=None,
            )
        provenance = f"Bundesnetzagentur | SMARD.de, filter {filter_id}"
        rows.append(
            Rows(
                dataset,
                provenance,
                False,
                smard.FEED_IN_STEP,
                smard.FEED_IN_LABEL,
                None,
                first,
                last,
                values,
            )
        )

    local_first = first.tz_convert(energy_charts.TIMEZONE).tz_localize(None)
    local_last = (last - grid.HOUR).tz_convert(energy_charts.TIMEZONE).tz_localize(None)
    months = pd.period_range(local_first, local_last, freq="M")
    for endpoint in energy_charts.PERIMETERS:
        power = pd.concat(
            [
                energy_charts.fetch_power_month(endpoint, month, client, energy_charts_cache_dir)
                for month in months
            ]
        )
        power = power[(power.index >= first) & (power.index < last)]
        provenance = f"Energy-Charts.info, Fraunhofer ISE, {endpoint}"
        values = {}
        for power_series in energy_charts.POWER_SERIES:
            series_id = f"energy_charts.DE.{endpoint}.{power_series}"
            values[series_id] = power[power_series]
            kept = completeness.on_grid(power[power_series], energy_charts.QUARTER_HOUR)
            hourly = grid.hourly_from_period_start_energy(
                kept * (energy_charts.QUARTER_HOUR / grid.HOUR)
            )
            series[series_id] = SeriesInputs(
                technology=power_series,
                published=(
                    Published(series_id, provenance, False, _on_hours(hourly, first, last)),
                ),
                modelled=None,
                capacity_mw=capacity_mw(power_series),
                offshore=None,
            )
        rows.append(
            Rows(
                f"energy_charts.{endpoint}",
                provenance,
                False,
                energy_charts.QUARTER_HOUR,
                energy_charts.POWER_LABEL,
                None,
                first,
                last,
                values,
            )
        )

    classes = eligibility.price_classes(
        first, last, zone="DE-LU", client=client, cache_dir=smard_cache_dir
    )
    return DetectionInputs(
        "DE", start, end, hours, tuple(rows), series, classes.drop(columns="mean_price")
    )


def offshore_starts(panel: pd.DataFrame) -> dict[str, pd.Timestamp]:
    """First hour, in UTC, from which the published wind of each French area holds offshore
    output, for the areas that have some, keyed by area: "FR" for mainland France, "FR-<code>"
    for a region of odre.REGIONS.

    Mainland France and Pays de la Loire start at odre.FIRST_OFFSHORE_POWER. Another region starts
    on the first day, in Europe/Paris, of the first quarter whose offshore capacity exceeds
    OFFSHORE_FROM_MW on `panel`, a quarterly panel of sdes.read_capacity.
    """
    mainland = sdes.metropolitan_without_corsica(panel)
    offshore = mainland[mainland["technology"] == "wind_offshore"]
    by_quarter = offshore.groupby(["region", "quarter_end"])["capacity_mw"].sum().reset_index()
    above = by_quarter[by_quarter["capacity_mw"] > OFFSHORE_FROM_MW]
    first_quarter = above.groupby("region")["quarter_end"].min()
    starts = {"FR": odre.FIRST_OFFSHORE_POWER, f"FR-{FIRST_FARM_REGION}": odre.FIRST_OFFSHORE_POWER}
    for code, name in odre.REGIONS.items():
        if code != FIRST_FARM_REGION and name in first_quarter.index:
            first_day = pd.Timestamp(first_quarter[name]).to_period("Q").start_time
            starts[f"FR-{code}"] = first_day.tz_localize(odre.TIMEZONE).tz_convert("UTC")
    return starts


def _eco2mix(
    client: httpx.Client,
    technologies: list[odre.RegionalSeries],
    first: pd.Timestamp,
    last: pd.Timestamp,
    cache_dir: Path,
) -> tuple[list[Rows], dict[str, tuple[Published, ...]]]:
    """Rows of the éCO2mix datasets over the hours [first, last), and the parts of each French
    series of `technologies`, keyed by series id."""
    days = first.date(), (last - grid.HOUR).date()
    national = odre.fetch_national_consolidated(client, *days, cache_dir=cache_dir)
    regional = odre.fetch_regional_consolidated(client, *days, cache_dir=cache_dir)
    national_kept = completeness.on_grid(national, odre.CONSOLIDATED_STEP)
    regional_kept = completeness.on_grid(regional, odre.CONSOLIDATED_STEP)

    published: dict[str, tuple[Published, ...]] = {}
    # The consolidated and definitive hours of each national series, and the dataset of their rows.
    consolidated: dict[str, tuple[str, pd.DataFrame]] = {}
    for technology in technologies:
        by_region = odre.regional_from_rows(regional_kept, technology, first, last)
        for code, hourly in by_region.items():
            series = f"odre.FR-{code}.{technology}"
            provenance = _provenance(odre.REGIONAL_CONSOLIDATED)
            published[series] = (Published(series, provenance, False, hourly),)
        if technology in SPLIT:
            consolidated[technology] = (
                odre.REGIONAL_CONSOLIDATED,
                odre.sum_of_regions(by_region),
            )
        else:
            power_mw, status = national_kept[f"{technology}_mw"], national_kept["status"]
            consolidated[technology] = (
                odre.NATIONAL_CONSOLIDATED,
                odre.consolidated_hourly(power_mw, status, first, last),
            )

    switches = {}
    for name, (dataset, national_hours) in consolidated.items():
        switches[name] = odre.consolidated_end(national_hours, first)
        before = odre.Hourly.validate(national_hours[national_hours.index < switches[name]])
        series = f"odre.FR.{name}"
        published[series] = (Published(series, _provenance(dataset), False, before),)

    rows = [
        _rows(
            odre.NATIONAL_CONSOLIDATED,
            {
                f"odre.FR.{technology}": national[f"{technology}_mw"]
                for technology in ("solar", "wind")
            },
            first,
            odre.complete_end(consolidated["solar"][1], first),
        ),
        _rows(
            odre.REGIONAL_CONSOLIDATED,
            {
                f"odre.FR-{code}.{technology}": regional.loc[
                    regional["region_code"] == code, f"{technology}_mw"
                ]
                for code in odre.REGIONS
                for technology in _published_from(technologies, first)
            },
            first,
            odre.complete_end(
                odre.sum_of_regions(odre.regional_from_rows(regional_kept, "solar", first, last)),
                first,
            ),
        ),
    ]

    later = {technology: switch for technology, switch in switches.items() if switch < last}
    if later:
        realtime_start = min(later.values())
        realtime = odre.fetch_national_realtime(
            client, realtime_start.date(), days[1], cache_dir=cache_dir
        )
        realtime_kept = completeness.on_grid(realtime, odre.REALTIME_STEP)
        provenance = _provenance(odre.NATIONAL_REALTIME)
        for name, switch in later.items():
            series = f"odre.FR.{name}"
            realtime_hours = odre.realtime_hourly(realtime_kept[f"{name}_mw"], switch, last)
            part = Published(series, provenance, True, odre.Hourly.validate(realtime_hours))
            published[series] += (part,)
        values = {
            f"odre.FR.{technology}": realtime[f"{technology}_mw"]
            for technology in _published_from(technologies, realtime_start)
        }
        rows.append(_rows(odre.NATIONAL_REALTIME, values, realtime_start, last))
    return [dataset for dataset in rows if dataset.end > dataset.start], published


def _rows(
    dataset: str, values: dict[str, pd.Series], start: pd.Timestamp, end: pd.Timestamp
) -> Rows:
    """Rows of the éCO2mix `dataset` over the hours [start, end)."""
    realtime = dataset == odre.NATIONAL_REALTIME
    return Rows(
        dataset=f"odre.{dataset}",
        provenance=_provenance(dataset),
        provisional=realtime,
        step=odre.REALTIME_STEP if realtime else odre.CONSOLIDATED_STEP,
        label=odre.REALTIME_LABEL if realtime else odre.CONSOLIDATED_LABEL,
        local_days=odre.TIMEZONE,
        start=start,
        end=end,
        values=values,
    )


def _provenance(dataset: str) -> str:
    return f"RTE, éCO2mix, ODRÉ dataset {dataset}"


def _published_from(
    technologies: list[odre.RegionalSeries], start: pd.Timestamp
) -> list[odre.RegionalSeries]:
    """The `technologies` whose values the éCO2mix rows hold from `start` on: onshore and offshore
    wind only from odre.WIND_SPLIT_START, before which their fields are empty or zero."""
    return [
        technology
        for technology in technologies
        if technology not in SPLIT or start >= odre.WIND_SPLIT_START
    ]


def _mainland(regional: dict[str, DataFrame[physics.Modelled]]) -> DataFrame[physics.Modelled]:
    """Modelled output of mainland France as the sum of its regions, incomplete in the hours when
    a region is. The regions share the weather and the SDES months, so the ERA5 source and the
    provisional capacity of any of them are those of all."""
    frames = list(regional.values())
    energy = pd.concat([frame["energy_mwh"] for frame in frames], axis="columns", sort=False)
    capacity = pd.concat([frame["capacity_mw"] for frame in frames], axis="columns", sort=False)
    total = energy.sum(axis="columns", skipna=False)
    return physics.Modelled.validate(
        frames[0].assign(
            energy_mwh=total, complete=total.notna(), capacity_mw=capacity.sum(axis="columns")
        )
    )


def _on_hours(hourly: pd.DataFrame, first: pd.Timestamp, last: pd.Timestamp) -> pd.DataFrame:
    return grid.Hourly.validate(grid.on_hours(hourly, first, last))


def _offshore(hours: pd.DatetimeIndex, start: pd.Timestamp | None) -> pd.Series:
    """Whether each of `hours` is at or after `start`, the first hour with offshore output; never
    without a start."""
    flags = np.zeros(len(hours), dtype=bool) if start is None else np.asarray(hours >= start)
    return pd.Series(flags, index=hours, name="offshore")
