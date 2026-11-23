"""Level breaks of published series against their model (evaluation plan, axis 2, and
Deviations).

The detector reads, for each UTC day, the log ratio L = ln(sum y / sum y_hat) of the published
output y to the uncalibrated physics model y_hat of the same perimeter, both summed over the usable
hours of the day: hours complete in both, from consolidated or definitive rows, with the reference
price class and, for wind, without offshore output, which the onshore model does not cover. A day
is used when its usable hours hold USABLE_SHARE_FROM or more of its modelled energy and, for wind,
when its modelled capacity factor reaches WIND_FACTOR_FROM.

Each series has a `SeasonalProfile`, fitted on the used days of 2019-2022: a constant and two
annual harmonics by least squares, and the noise scale sigma of e = L - profile, the median
absolute deviation of the differences of e between consecutive used days, scaled to the standard
deviation of a normal distribution, over sqrt(2). Change points of e / sigma are those of PELT
with a cost on the mean and a penalty per change point, which ruptures.KernelCPD computes with the
linear kernel, on segments of at least MIN_SIZE_DAYS used days. A break is dated at the first day
of its new segment, and its step is the mean of e over the new segment minus its mean over the
segment before. A search with wind never spans WIND_SWITCH, when the French published series go
from wind to onshore wind.
"""

from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from typing import Any, ClassVar, Literal, NamedTuple

import numpy as np
import numpy.typing as npt
import pandas as pd
import pandera.pandas as pa
import ruptures
from pandera.typing import DataFrame, Index, Series
from scipy.stats import median_abs_deviation

from eu_solar_wind_audit import grid, physics
from eu_solar_wind_audit.anomalies import findings
from eu_solar_wind_audit.anomalies.findings import Findings
from eu_solar_wind_audit.sources import odre

FloatArray = npt.NDArray[np.float64]

# Wind covers wind and onshore wind series.
Technology = Literal["solar", "wind"]
TECHNOLOGIES: tuple[Technology, ...] = ("solar", "wind")
DAY = pd.Timedelta(days=1)
HOURS_PER_DAY = 24
FINAL_STATUSES = ("definitive", "consolidated")

# A day is used when its usable hours hold at least 80 % of its modelled energy, and a wind day
# when its modelled capacity factor is at least 0.05 too, below which the ratio is unstable
# (detector design).
USABLE_SHARE_FROM = 0.8
WIND_FACTOR_FROM = 0.05
# Profiles and sigmas rest on 2019-2022, the years that fix the thresholds (evaluation plan,
# axis 2): a constant and two annual harmonics of the day of the year (detector design).
PROFILE_YEARS = range(2019, 2023)
HARMONICS = 2
YEAR_DAYS = 365.25
# Segments of at least 7 used days, set before any measurement (detector design).
MIN_SIZE_DAYS = 7
# The French published wind series are wind up to 2023 and onshore wind from 1 January 2024 in
# Paris time, the first UTC day of which is the first day whole in onshore wind (evaluation plan,
# Deviations).
WIND_SWITCH = odre.WIND_SPLIT_START.ceil("D")
# Breaks within 7 days of a junction are not attributed (evaluation plan, axis 2): the first ERA5T
# hour, the first hour with provisional capacity (anomaly critique), and changes of status.
JUNCTION_DAYS = 7
JUNCTION_KINDS = ["first_era5t_hour", "first_provisional_capacity_hour", "status_change"]
# A break of one series and a joint break at most 3 days apart are one break, the tolerance of the
# attribution and of the dating of steps in the detector design.
MATCH_DAYS = 3
# A joint break is attributed only when the step of each series reaches this many sigmas.
STEP_FROM_SIGMAS = 1.0
REASONS = ["near_junction", "no_joint_break", "single_technology", "opposite_signs"]
# Detection runs search the daily ratios from the start of the period of the plan, and report the
# breaks of [START, END) once the ratios reach 15 days past END (anomaly critique).
FIRST_DAY = pd.Timestamp("2019-01-01", tz="UTC")
MARGIN_DAYS = 15


class DailyRatios(pa.DataFrameModel):
    """Published over modelled output of one series over the UTC day starting at `day`.

    `log_ratio` is ln(sum y / sum y_hat) over the usable hours of the day, null unless both sums
    are positive. `usable_share` is the share of the modelled daily energy in these hours, and
    `modelled_capacity_factor` the modelled daily energy over the energy of the capacity over the
    day; both are null when the model lacks an hour of the day. Only `used` days are searched.
    """

    day: Index[grid.UtcTime] = pa.Field(unique=True, check_name=True)
    log_ratio: Series[float] = pa.Field(nullable=True)
    usable_share: Series[float] = pa.Field(nullable=True)
    modelled_capacity_factor: Series[float] = pa.Field(nullable=True)
    used: Series[bool]

    @pa.check("day")
    @classmethod
    def at_midnight(cls, day: "pd.Series[pd.Timestamp]") -> "pd.Series[bool]":
        return day == day.dt.floor("D")

    @pa.dataframe_check
    @classmethod
    def used_days_have_a_ratio(cls, frame: pd.DataFrame) -> "pd.Series[bool]":
        return ~frame["used"] | frame["log_ratio"].notna()

    class Config:
        strict = True


def daily_log_ratios(
    published: pd.DataFrame,
    modelled: pd.DataFrame,
    price_classes: pd.DataFrame,
    *,
    technology: Technology,
    offshore: pd.Series | None = None,
) -> DataFrame[DailyRatios]:
    """`DailyRatios` of each UTC day with an hour in `published`, a series of grid.Hourly or, with
    the status of each hour, of odre.Hourly.

    `modelled` is the physics.Modelled output of the same perimeter, and `price_classes` gives the
    `regime` of each UTC hour, as regimes.classify_hours does. An hour is usable when it is
    complete in `published` and in `modelled`, its status is definitive or consolidated (a series
    without status has neither real-time nor provisional rows), and its price class is reference.
    `offshore` is the boolean mask of the hours of a wind series that contain offshore output; it
    covers every hour of `published`, and the hours it flags are not usable.
    """
    published, final = _published(published)
    hours = pd.DatetimeIndex(published.index)
    model = physics.Modelled.validate(modelled)
    regime = price_classes["regime"]
    # Classes on other hours would leave every hour out without an error.
    if not isinstance(regime.index, pd.DatetimeIndex) or str(regime.index.tz) != "UTC":
        raise ValueError("expected price classes on a DatetimeIndex in UTC")
    on_hours = model["energy_mwh"].reindex(hours)
    usable = (
        published["complete"] & final & on_hours.notna() & regime.reindex(hours).eq("reference")
    )
    if offshore is not None:
        if technology != "wind":
            raise ValueError("only a wind series has an offshore mask")
        usable &= ~_on_hours(offshore, hours)

    day = hours.floor("D").rename("day")
    published_mwh = published["energy_mwh"].where(usable).groupby(day).sum()
    usable_mwh = on_hours.where(usable).groupby(day).sum()
    by_model_day = model.groupby(pd.DatetimeIndex(model.index).floor("D"))
    days = published_mwh.index
    daily_mwh = by_model_day["energy_mwh"].sum(min_count=HOURS_PER_DAY).reindex(days)
    # The capacity of each hour gives its energy at full load.
    full_load_mwh = by_model_day["capacity_mw"].sum().reindex(days)
    log_ratio = np.log(published_mwh.where(published_mwh > 0) / usable_mwh.where(usable_mwh > 0))
    table = pd.DataFrame(
        {
            "log_ratio": log_ratio,
            "usable_share": usable_mwh / daily_mwh.where(daily_mwh > 0),
            "modelled_capacity_factor": daily_mwh / full_load_mwh.where(full_load_mwh > 0),
        }
    )
    used = (table["usable_share"] >= USABLE_SHARE_FROM) & table["log_ratio"].notna()
    if technology == "wind":
        used &= table["modelled_capacity_factor"] >= WIND_FACTOR_FROM
    table["used"] = used
    table.index = pd.DatetimeIndex(days, name="day").as_unit("ns")
    return DailyRatios.validate(table)


@dataclass(frozen=True)
class SeasonalProfile:
    """Seasonal profile of the daily log ratio of one series, and the noise scale `sigma` of the
    ratio less its profile, as `fit_profile` gives them.

    The profile of a day is c0 + c1 cos(w) + c2 sin(w) + c3 cos(2 w) + c4 sin(2 w), with
    (c0, ..., c4) the `coefficients`, w = 2 pi d / YEAR_DAYS and d the UTC day of the year.
    """

    coefficients: tuple[float, ...]
    sigma: float

    def __post_init__(self) -> None:
        if len(self.coefficients) != 1 + 2 * HARMONICS:
            raise ValueError(f"expected {1 + 2 * HARMONICS} coefficients")
        if not self.sigma > 0:
            raise ValueError(f"expected a positive sigma, not {self.sigma}")


class Ratios(NamedTuple):
    """Daily log ratios of one series and the profile that deseasonalizes them."""

    daily: DataFrame[DailyRatios]
    profile: SeasonalProfile


def fit_profile(daily: pd.DataFrame) -> SeasonalProfile:
    """`SeasonalProfile` of the used days of `daily`, a DailyRatios frame of days of PROFILE_YEARS
    only: the least-squares coefficients of the profile, then sigma from the differences of the
    ratio less its profile between consecutive used days.

    Raises ValueError if a day lies outside PROFILE_YEARS, if there are no more used days than
    coefficients, or if the ratio less its profile does not vary.
    """
    table = DailyRatios.validate(daily).sort_index()
    _check_profile_years(pd.DatetimeIndex(table.index))
    ratio = table.loc[table["used"], "log_ratio"]
    terms = _harmonics(pd.DatetimeIndex(ratio.index))
    if len(ratio) <= terms.shape[1]:
        raise ValueError(f"{len(ratio)} used days are too few to fit the profile")
    coefficients, *_ = np.linalg.lstsq(terms, ratio.to_numpy(), rcond=None)
    differences = np.diff(ratio.to_numpy() - terms @ coefficients)
    sigma = median_abs_deviation(differences, scale="normal") / np.sqrt(2)
    return SeasonalProfile(tuple(float(value) for value in coefficients), float(sigma))


class Breaks(pa.DataFrameModel):
    """Breaks of one search, one row per break and searched series, which `technology` names.

    `day` is the first day of the new segment. `step` is the mean of the deseasonalized log ratio
    of the series over the new segment minus its mean over the segment before, and `step_sigmas`
    the step over the sigma of the series. The segment before starts on `before_from`, the new
    segment ends before `after_until`, and they hold `days_before` and `days_after` used days.
    """

    day: Series[grid.UtcTime]
    technology: Series[str] = pa.Field(isin=list(TECHNOLOGIES))
    step: Series[float]
    step_sigmas: Series[float]
    before_from: Series[grid.UtcTime]
    after_until: Series[grid.UtcTime]
    days_before: Series[int] = pa.Field(ge=MIN_SIZE_DAYS)
    days_after: Series[int] = pa.Field(ge=MIN_SIZE_DAYS)

    class Config:
        strict = True
        unique: ClassVar[list[str]] = ["day", "technology"]


def change_points(ratios: Mapping[Technology, Ratios], *, penalty: float) -> DataFrame[Breaks]:
    """Breaks of the search of one series, or of the solar and the wind series of an area
    together when `ratios` holds both, with `penalty`.

    The search reads e / sigma of each series on the days used in all of them, in time order. A
    search with wind runs apart on the days before WIND_SWITCH and on the days from it.
    """
    if not ratios or not set(ratios) <= set(TECHNOLOGIES):
        raise ValueError(f"expected the ratios of some of {TECHNOLOGIES}")
    residuals = pd.concat(
        {technology: _residuals(series) for technology, series in ratios.items()},
        axis="columns",
        sort=True,
    ).dropna()
    sigmas = {technology: series.profile.sigma for technology, series in ratios.items()}
    parts = [residuals]
    if "wind" in ratios:
        days = pd.DatetimeIndex(residuals.index)
        parts = [residuals[days < WIND_SWITCH], residuals[days >= WIND_SWITCH]]
    rows = [row for part in parts for row in _part_breaks(part, sigmas, penalty)]
    if not rows:
        return Breaks.empty()
    return Breaks.validate(pd.DataFrame(rows))


def _part_breaks(
    residuals: pd.DataFrame, sigmas: Mapping[Technology, float], penalty: float
) -> list[dict[str, object]]:
    """Rows of `Breaks` for the change points of `residuals`, the deseasonalized log ratio of each
    series of `sigmas`, one column each, on consecutive used days."""
    # ruptures refuses a signal too short for two segments.
    if len(residuals) < 2 * MIN_SIZE_DAYS:
        return []
    scaled = np.column_stack(
        [residuals[technology] / sigma for technology, sigma in sigmas.items()]
    )
    search = ruptures.KernelCPD(kernel="linear", min_size=MIN_SIZE_DAYS).fit(scaled)
    # The ends of the segments, the last one included.
    bounds = [0, *(int(end) for end in search.predict(pen=penalty))]
    days = pd.DatetimeIndex(residuals.index)
    rows: list[dict[str, object]] = []
    for first, cut, last in zip(bounds, bounds[1:], bounds[2:], strict=False):
        for technology, sigma in sigmas.items():
            values = residuals[technology].to_numpy()
            step = float(values[cut:last].mean() - values[first:cut].mean())
            rows.append(
                {
                    "day": days[cut],
                    "technology": technology,
                    "step": step,
                    "step_sigmas": step / sigma,
                    "before_from": days[first],
                    "after_until": days[last - 1] + DAY,
                    "days_before": cut - first,
                    "days_after": last - cut,
                }
            )
    return rows


class Junctions(pa.DataFrameModel):
    """Hours from which the reference or the published rows of a series change, which can move
    the level of solar and wind at once: the first hour of ERA5T weather, the first hour with
    provisional capacity, and each change of the publication status."""

    time: Series[grid.UtcTime]
    kind: Series[str] = pa.Field(isin=JUNCTION_KINDS)

    class Config:
        strict = True


def junctions(published: pd.DataFrame, modelled: pd.DataFrame) -> DataFrame[Junctions]:
    """`Junctions` of `published`, a series of grid.Hourly or, with the status of each hour, of
    odre.Hourly, whose model is `modelled`, a physics.Modelled frame. A status change is an hour
    whose status differs from that of the last hour before it with a status."""
    model = physics.Modelled.validate(modelled).sort_index()
    rows: list[tuple[pd.Timestamp, str]] = []
    for kind, flagged in [
        ("first_era5t_hour", model["era5_source"].eq("ERA5T")),
        ("first_provisional_capacity_hour", model["capacity_provisional"]),
    ]:
        hours = pd.DatetimeIndex(model.index[flagged.to_numpy()])
        if not hours.empty:
            rows.append((hours[0], kind))
    if "status" in published.columns:
        status = odre.Hourly.validate(published).sort_index()["status"].dropna()
        changed = status.ne(status.shift()).to_numpy()
        rows += [(hour, "status_change") for hour in status.index[1:][changed[1:]]]
    if not rows:
        return Junctions.empty()
    frame = pd.DataFrame(rows, columns=["time", "kind"])
    return Junctions.validate(frame.astype({"time": "datetime64[ns, UTC]"}))


class Unattributed(Findings):
    """Level breaks that are not attributed to the published data, with the first `reason` that
    applies, as `level_breaks` gives them."""

    reason: Series[str] = pa.Field(isin=REASONS)


class Penalties(NamedTuple):
    """Penalties of the search of a solar series, of a wind series, and of the solar and the wind
    series of an area together."""

    solar: float
    wind: float
    joint: float


class LevelBreaks(NamedTuple):
    """Level breaks of the series of an area: the findings, attributed to the published data, and
    the breaks left unattributed."""

    findings: DataFrame[Findings]
    unattributed: DataFrame[Unattributed]


def level_breaks(
    ratios: Mapping[Technology, Ratios],
    junctions: pd.DataFrame,
    *,
    series: Mapping[Technology, str],
    provenance: Mapping[Technology, str],
    penalties: Penalties,
    start: pd.Timestamp,
    end: pd.Timestamp,
    step_from_sigmas: float = STEP_FROM_SIGMAS,
) -> LevelBreaks:
    """Level breaks dated in [start, end), both hour starts in UTC, of the solar and the wind
    series of an area: their `ratios`, their ids `series`, the `provenance` of their rows, and the
    `junctions` of both.

    Each series is searched alone with the penalty of its technology, and both together with the
    joint penalty, as change_points does. A joint break is attributed to the published data when
    the step of each series reaches `step_from_sigmas` sigmas, both steps have the same sign, and
    no junction lies within JUNCTION_DAYS of it: it gives a `level_break` finding on each series.
    Such a joint break near a junction is unattributed on both series. A break of one series is
    unattributed unless such a joint break lies within MATCH_DAYS of it, with the first reason that
    applies: a junction within JUNCTION_DAYS (near_junction), no joint break within MATCH_DAYS
    (no_joint_break), or, for the nearest joint break, a step below `step_from_sigmas` sigmas in a
    series (single_technology) or steps of opposite signs (opposite_signs).

    The daily ratios start by FIRST_DAY and reach MARGIN_DAYS past `end`, except that a wind
    series needs no day from WIND_SWITCH when `end` is not after it, since no search spans that
    day. Raises ValueError otherwise: the breaks of the period are then deferred to a run with more
    days. The evidence gives the step as a log ratio, as a relative change and in sigmas, the
    adjacent segments, the search, and the junction or the joint break behind the reason.
    """
    grid.check_period(start, end)
    if set(ratios) != set(TECHNOLOGIES):
        raise ValueError("expected the ratios of a solar and a wind series")
    for technology, searched in ratios.items():
        _check_window(searched.daily, series[technology], technology, end)
    nearby = Junctions.validate(junctions)
    junction_days = nearby.assign(day=nearby["time"].dt.floor("D"))

    joint = change_points(ratios, penalty=penalties.joint)
    joint_steps: dict[pd.Timestamp, dict[str, float]] = {}
    for row in joint.to_dict("records"):
        joint_steps.setdefault(row["day"], {})[row["technology"]] = row["step_sigmas"]
    joint_days = pd.DatetimeIndex(sorted(joint_steps))

    found: list[dict[str, object]] = []
    unattributed: list[tuple[dict[str, object], str]] = []
    for row in joint.to_dict("records"):
        steps = joint_steps[row["day"]]
        if _joint_reason(steps, step_from_sigmas) is not None or not start <= row["day"] < end:
            continue
        this: Technology = row["technology"]
        other: Technology = "wind" if this == "solar" else "solar"
        evidence = {
            **_evidence(row),
            "search": "solar_and_wind",
            "other_series": series[other],
            "other_step_sigmas": steps[other],
        }
        junction = _nearest_junction(junction_days, row["day"])
        if junction is not None:
            evidence["junction"] = junction
        record = _record(row, series[this], provenance[this], evidence)
        if junction is None:
            found.append(record)
        else:
            unattributed.append((record, "near_junction"))

    for technology in TECHNOLOGIES:
        penalty = penalties.solar if technology == "solar" else penalties.wind
        alone = change_points({technology: ratios[technology]}, penalty=penalty)
        for row in alone.to_dict("records"):
            joint_day = _nearest_day(joint_days, row["day"])
            joint_reason = None
            if joint_day is not None:
                joint_reason = _joint_reason(joint_steps[joint_day], step_from_sigmas)
                # The joint break is that of the finding, or of a break kept near a junction.
                if joint_reason is None:
                    continue
            if not start <= row["day"] < end:
                continue
            evidence = {**_evidence(row), "search": technology}
            if joint_day is not None:
                evidence["joint_break"] = {
                    "day": _date(joint_day),
                    **{
                        f"{name}_step_sigmas": step for name, step in joint_steps[joint_day].items()
                    },
                }
            junction = _nearest_junction(junction_days, row["day"])
            if junction is not None:
                evidence["junction"] = junction
            reason = "near_junction" if junction is not None else joint_reason or "no_joint_break"
            record = _record(row, series[technology], provenance[technology], evidence)
            unattributed.append((record, reason))

    return LevelBreaks(_findings(found), _unattributed(unattributed))


def _check_window(
    daily: pd.DataFrame, series: str, technology: Technology, end: pd.Timestamp
) -> None:
    """Raise ValueError unless the daily ratios `daily` of `series` cover the days that
    `level_breaks` needs for a period that ends at `end`."""
    days = pd.DatetimeIndex(DailyRatios.validate(daily).index).sort_values()
    if days.empty or days[0] > FIRST_DAY:
        raise ValueError(f"the daily ratios of {series} start after {FIRST_DAY.date()}")
    needed = end + MARGIN_DAYS * DAY
    if technology == "wind" and end <= WIND_SWITCH:
        needed = min(needed, WIND_SWITCH)
    if days[-1] + DAY < needed:
        raise ValueError(
            f"the daily ratios of {series} end on {days[-1].date()}, before {needed.date()}: "
            f"the breaks up to {end} are deferred"
        )


def _joint_reason(steps: Mapping[str, float], step_from_sigmas: float) -> str | None:
    """Why a joint break, with the step of each series in sigmas, is not a break of both series:
    single_technology or opposite_signs; None when it is."""
    if min(abs(step) for step in steps.values()) < step_from_sigmas:
        return "single_technology"
    if steps["solar"] * steps["wind"] < 0:
        return "opposite_signs"
    return None


def _nearest_day(days: pd.DatetimeIndex, day: pd.Timestamp) -> pd.Timestamp | None:
    """The day of `days` nearest to `day`, if it lies within MATCH_DAYS of it."""
    if days.empty:
        return None
    offsets = np.abs(((days - day) / DAY).to_numpy())
    position = int(offsets.argmin())
    return days[position] if offsets[position] <= MATCH_DAYS else None


def _nearest_junction(junctions: pd.DataFrame, day: pd.Timestamp) -> dict[str, str] | None:
    """Kind and hour of the junction of `junctions`, a Junctions frame with the UTC `day` of each,
    nearest to `day`, if it lies within JUNCTION_DAYS of it."""
    if junctions.empty:
        return None
    offsets = np.abs(((junctions["day"] - day) / DAY).to_numpy())
    position = int(offsets.argmin())
    if offsets[position] > JUNCTION_DAYS:
        return None
    time = pd.DatetimeIndex(junctions["time"])[position]
    return {"kind": str(junctions["kind"].iloc[position]), "time": time.isoformat()}


def _evidence(row: Mapping[Hashable, Any]) -> dict[str, object]:
    """Evidence of the break of a `Breaks` row."""
    return {
        "step": row["step"],
        "relative_step": float(np.expm1(row["step"])),
        "step_sigmas": row["step_sigmas"],
        "before": {
            "start": _date(row["before_from"]),
            "end": _date(row["day"]),
            "used_days": row["days_before"],
        },
        "after": {
            "start": _date(row["day"]),
            "end": _date(row["after_until"]),
            "used_days": row["days_after"],
        },
    }


def _record(
    row: Mapping[Hashable, Any], series: str, provenance: str, evidence: dict[str, object]
) -> dict[str, object]:
    """Finding of the series `series` for the break of a `Breaks` row, over its first day."""
    return {
        "series": series,
        "provenance": provenance,
        "provisional": False,
        "start": row["day"],
        "end": row["day"] + DAY,
        "kind": "level_break",
        "score": row["step"],
        "evidence": evidence,
    }


def _findings(records: list[dict[str, object]]) -> DataFrame[Findings]:
    found = findings.from_records(records)
    return Findings.validate(found.sort_values(["start", "series"], ignore_index=True))


def _unattributed(records: list[tuple[dict[str, object], str]]) -> DataFrame[Unattributed]:
    if not records:
        return Unattributed.empty()
    found = findings.from_records([record for record, _ in records])
    found["reason"] = [reason for _, reason in records]
    return Unattributed.validate(found.sort_values(["start", "series"], ignore_index=True))


def _date(day: pd.Timestamp) -> str:
    return day.date().isoformat()


def _published(published: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """`published` validated, and whether each of its hours comes from consolidated or definitive
    rows."""
    if "status" in published.columns:
        with_status = odre.Hourly.validate(published)
        return with_status, with_status["status"].isin(FINAL_STATUSES)
    hours = grid.Hourly.validate(published)
    return hours, pd.Series(True, index=hours.index)


def _on_hours(offshore: pd.Series, hours: pd.DatetimeIndex) -> pd.Series:
    """The boolean mask `offshore` on `hours`, which it must cover."""
    if offshore.dtype != bool:
        raise ValueError("the offshore mask must be boolean")
    mask = offshore.reindex(hours)
    if mask.isna().any():
        raise ValueError("the offshore mask does not cover every published hour")
    return mask.astype(bool)


def _check_profile_years(days: pd.DatetimeIndex) -> None:
    outside = days[~days.year.isin(PROFILE_YEARS)]
    if not outside.empty:
        first, last = PROFILE_YEARS[0], PROFILE_YEARS[-1]
        raise ValueError(f"{outside[0].date()} lies outside the profile years {first}-{last}")


def _harmonics(days: pd.DatetimeIndex) -> FloatArray:
    """Terms of the profile at `days`: 1, then cos(k w) and sin(k w) for k from 1 to HARMONICS."""
    angle = 2 * np.pi * days.day_of_year.to_numpy() / YEAR_DAYS
    terms = [np.ones(angle.size)]
    for k in range(1, HARMONICS + 1):
        terms += [np.cos(k * angle), np.sin(k * angle)]
    return np.column_stack(terms)


def _residuals(ratios: Ratios) -> pd.Series:
    """The log ratio less its profile on the used days of `ratios`, in time order."""
    table = DailyRatios.validate(ratios.daily).sort_index()
    ratio: pd.Series = table.loc[table["used"], "log_ratio"]
    profile = _harmonics(pd.DatetimeIndex(ratio.index)) @ np.array(ratios.profile.coefficients)
    return ratio.sub(profile)
