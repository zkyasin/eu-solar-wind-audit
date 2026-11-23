"""Thresholds of the statistical anomaly detectors of mainland France, fixed on 2019-2022
(evaluation plan, axis 2, and Deviations) and kept in THRESHOLDS_DIR/<country>.json with the
reference model they rest on.

`fix` reads the detection inputs of FIX_YEARS, without margins, and gives:
- the centroid bias of the model: the median residual of the valid days of national solar;
- the solar series eligible for shift claims: those with at most BUDGET claim in every year;
- the residual threshold of each technology, among RESIDUAL_THRESHOLDS, and the level-break
  penalties of solar, wind and joint searches, among breaks.PENALTIES: for each, the smallest value
  with at most BUDGET finding in every series and year, as the largest choice of the
  leave-one-year-out folds (breaks.penalty_folds);
- the seasonal profile and noise of each series, for the level breaks;
- the findings of every statistical detector per series and year with these thresholds, reported
  against the budget. The level breaks counted are those of the search of the series alone,
  attributed or not: an upper bound of its level-break findings.

detect-anomalies refuses a file whose reference model differs from the current one on
REFERENCE_KEYS; the git description is recorded only.
"""

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from eu_solar_wind_audit import physics
from eu_solar_wind_audit.anomalies import breaks, residuals, timestamp
from eu_solar_wind_audit.anomalies.inputs import DetectionInputs, SeriesInputs
from eu_solar_wind_audit.config import Country
from eu_solar_wind_audit.sources import era5, natural_earth

THRESHOLDS_DIR = Path("config/anomaly-thresholds")
FIX_YEARS = range(2019, 2023)
FIX_START = pd.Timestamp(FIX_YEARS[0], 1, 1, tz="UTC")
FIX_END = pd.Timestamp(FIX_YEARS[-1] + 1, 1, 1, tz="UTC")
# Grid of the detector design, and the false-alarm budget per series and year (evaluation plan,
# Deviations).
RESIDUAL_THRESHOLDS = (6.0, 8.0, 10.0, 12.0, 15.0, 20.0)
BUDGET = 1
NATIONAL_SOLAR = "odre.FR.solar"
REFERENCE_KEYS = ("parameters_sha256", "sdes_media", "era5_store", "regions")


@dataclass(frozen=True)
class Thresholds:
    """Thresholds of one country, as the module docstring lists them; `false_alarms` has one row
    per series and year of FIX_YEARS, with the findings of each statistical detector and their
    total."""

    country: str
    reference_model: dict[str, Any]
    centroid_bias_minutes: float
    shift_series: tuple[str, ...]
    residual_thresholds: dict[str, float]
    penalties: breaks.Penalties
    profiles: dict[str, breaks.SeasonalProfile]
    false_alarms: pd.DataFrame


def reference_model(parameters: Country, git_describe: str | None) -> dict[str, Any]:
    """The reference model of thresholds fixed with the country `parameters`: the SHA-256 of the
    parameters, the SDES media of the capacity, the ERA5 stores and the regions, and the git
    description of the code."""
    settings = json.dumps(dataclasses.asdict(parameters), sort_keys=True, default=str)
    return {
        "parameters_sha256": hashlib.sha256(settings.encode()).hexdigest(),
        "sdes_media": dict(physics.SDES_MEDIA),
        "era5_store": {
            "final": f"s3://{era5.ICECHUNK_BUCKET}/{era5.ICECHUNK_PREFIX}, "
            f"group {era5.ICECHUNK_GROUP}",
            "recent": era5.ARCO_URL,
        },
        "regions": f"Natural Earth admin-1 {natural_earth.VERSION}",
        "git_describe": git_describe,
    }


def check_reference(thresholds: Thresholds, reference: dict[str, Any]) -> None:
    """Raise ValueError if `thresholds` rest on another reference model than `reference`."""
    changed = [
        key for key in REFERENCE_KEYS if thresholds.reference_model.get(key) != reference.get(key)
    ]
    if changed:
        raise ValueError(f"the thresholds rest on another reference model: {', '.join(changed)}")


def fix(found: DetectionInputs, reference: dict[str, Any]) -> Thresholds:
    """Thresholds from `found`, the detection inputs of mainland France over FIX_YEARS, which rest
    on `reference`.

    Raises ValueError unless every hour of `found` lies in FIX_YEARS.
    """
    if found.hours[0] < FIX_START or found.hours[-1] >= FIX_END:
        raise ValueError(f"thresholds rest on {FIX_YEARS[0]}-{FIX_YEARS[-1]} only")
    classes = found.price_classes
    by_technology = {
        technology: {
            series: item for series, item in found.series.items() if item.technology == technology
        }
        for technology in breaks.TECHNOLOGIES
    }

    national = found.series[NATIONAL_SOLAR]
    unbiased = timestamp.daily_centroids(
        _published(national), _modelled(national), classes, bias_minutes=0.0
    )
    bias = float(unbiased.loc[unbiased["valid"], "residual_minutes"].median())
    claims = {
        series: per_year(
            timestamp.shifts(
                timestamp.daily_centroids(
                    _published(item), _modelled(item), classes, bias_minutes=bias
                ),
                series=series,
                provenance=item.published[0].provenance,
            )
        )
        for series, item in by_technology["solar"].items()
    }
    shift_series = tuple(series for series, counts in claims.items() if counts.max() <= BUDGET)

    residual_counts = []
    for technology, items in by_technology.items():
        for series, item in items.items():
            scores = residuals.residual_scores(
                _published(item),
                _modelled(item),
                classes,
                technology=technology,
                offshore=item.offshore if technology == "wind" else None,
            )
            for threshold in RESIDUAL_THRESHOLDS:
                found_outliers = residuals.residual_outliers(
                    scores,
                    series=series,
                    provenance=item.published[0].provenance,
                    threshold=threshold,
                )
                counts = per_year(found_outliers)
                residual_counts += [
                    (technology, series, threshold, year, int(count))
                    for year, count in counts.items()
                ]
    residual_table = pd.DataFrame(
        residual_counts, columns=["technology", "series", "threshold", "year", "findings"]
    )
    residual_thresholds: dict[str, float] = {
        technology: _largest_fold_choice(residual_table[residual_table["technology"] == technology])
        for technology in breaks.TECHNOLOGIES
    }

    ratios = {}
    for technology, items in by_technology.items():
        for series, item in items.items():
            daily = breaks.daily_log_ratios(
                _published(item),
                _modelled(item),
                classes,
                technology=technology,
                offshore=item.offshore if technology == "wind" else None,
            )
            ratios[series] = breaks.Ratios(daily, breaks.fit_profile(daily))
    searches: dict[str, dict[str, dict[breaks.Technology, breaks.Ratios]]] = {
        "solar": {series: {"solar": ratios[series]} for series in by_technology["solar"]},
        "wind": {series: {"wind": ratios[series]} for series in by_technology["wind"]},
        # The solar and the wind series of each area together, named after the area.
        "joint": {
            series.removesuffix(".solar"): {
                "solar": ratios[series],
                "wind": ratios[series.removesuffix(".solar") + ".wind"],
            }
            for series in by_technology["solar"]
        },
    }
    counts_of = {name: breaks.break_counts(found) for name, found in searches.items()}
    chosen = {
        name: float(breaks.penalty_folds(counts)["penalty"].max())
        for name, counts in counts_of.items()
    }
    penalties = breaks.Penalties(chosen["solar"], chosen["wind"], chosen["joint"])

    rows = []
    for technology, items in by_technology.items():
        break_table = counts_of[technology]
        level_breaks_of = break_table[break_table["penalty"] == chosen[technology]].set_index(
            ["search", "year"]
        )["breaks"]
        chosen_residual = (residual_table["technology"] == technology) & (
            residual_table["threshold"] == residual_thresholds[technology]
        )
        outliers_of = residual_table[chosen_residual].set_index(["series", "year"])["findings"]
        for series in items:
            for year in FIX_YEARS:
                shift_claims = int(claims[series][year]) if series in shift_series else 0
                outliers = int(outliers_of.get((series, year), 0))
                level_breaks = int(level_breaks_of.get((series, year), 0))
                rows.append(
                    {
                        "series": series,
                        "technology": technology,
                        "year": year,
                        "timestamp_shift": shift_claims,
                        "residual_outlier": outliers,
                        "level_break": level_breaks,
                        "total": shift_claims + outliers + level_breaks,
                    }
                )
    return Thresholds(
        country=found.country,
        reference_model=reference,
        centroid_bias_minutes=bias,
        shift_series=shift_series,
        residual_thresholds=residual_thresholds,
        penalties=penalties,
        profiles={series: searched.profile for series, searched in ratios.items()},
        false_alarms=pd.DataFrame(rows),
    )


def per_year(found: pd.DataFrame) -> "pd.Series[int]":
    """Findings of `found`, a Findings frame, per UTC year of FIX_YEARS, by the year they start
    in."""
    years = pd.DatetimeIndex(found["start"]).year
    return pd.Series(years, dtype="int64").value_counts().reindex(FIX_YEARS, fill_value=0)


def write(path: Path, thresholds: Thresholds) -> None:
    """Write `thresholds` to `path` as JSON."""
    content = {
        "country": thresholds.country,
        "fixed_on": [FIX_YEARS[0], FIX_YEARS[-1]],
        "reference_model": thresholds.reference_model,
        "timestamp": {
            "centroid_bias_minutes": thresholds.centroid_bias_minutes,
            "shift_series": list(thresholds.shift_series),
        },
        "residual_thresholds": thresholds.residual_thresholds,
        "level_break_penalties": thresholds.penalties._asdict(),
        "profiles": {
            series: {"coefficients": list(profile.coefficients), "sigma": profile.sigma}
            for series, profile in sorted(thresholds.profiles.items())
        },
        "false_alarms": thresholds.false_alarms.to_dict("records"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=1, allow_nan=False) + "\n", encoding="utf-8")


def read(path: Path) -> Thresholds:
    """The thresholds that `write` wrote to `path`."""
    content = json.loads(path.read_text(encoding="utf-8"))
    return Thresholds(
        country=content["country"],
        reference_model=content["reference_model"],
        centroid_bias_minutes=content["timestamp"]["centroid_bias_minutes"],
        shift_series=tuple(content["timestamp"]["shift_series"]),
        residual_thresholds=content["residual_thresholds"],
        penalties=breaks.Penalties(**content["level_break_penalties"]),
        profiles={
            series: breaks.SeasonalProfile(tuple(profile["coefficients"]), profile["sigma"])
            for series, profile in content["profiles"].items()
        },
        false_alarms=pd.DataFrame(content["false_alarms"]),
    )


def _largest_fold_choice(counts: pd.DataFrame) -> float:
    """The largest leave-one-year-out choice of a threshold, from its findings per series and
    year: the rule of breaks.penalty_folds, with the series as searches."""
    table = counts.rename(
        columns={"series": "search", "threshold": "penalty", "findings": "breaks"}
    )
    folds = breaks.penalty_folds(table[["search", "year", "penalty", "breaks"]], budget=BUDGET)
    return float(folds["penalty"].max())


def _published(item: SeriesInputs) -> pd.DataFrame:
    """The hours of all the parts of a series, in time order."""
    return pd.concat([part.hourly for part in item.published]).sort_index()


def _modelled(item: SeriesInputs) -> pd.DataFrame:
    if item.modelled is None:
        raise ValueError("statistical detectors need the modelled output of each series")
    return item.modelled
