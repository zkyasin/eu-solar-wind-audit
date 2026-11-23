import dataclasses
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import pytest
from pandera.typing import DataFrame
from scipy.optimize import OptimizeResult, least_squares
from scipy.stats import norm

from eu_solar_wind_audit import calibration, precompute
from eu_solar_wind_audit.config import WindGeneration
from eu_solar_wind_audit.precompute import FloatArray
from eu_solar_wind_audit.pv import PVParameters, System

CAPACITY_MW = 100.0
PV = PVParameters(fleet=(System(35.0, 180.0, 0.25), System(15.0, 110.0, 0.75)), dc_ac_ratio=1.2)
SOLAR_TRUTH = (0.85, 0.55, 0.04)
GENERATIONS = (
    WindGeneration("VestasV82_1.65MW_82", 80.0, 2012),
    WindGeneration("IEA_Reference_3.4MW_130", 100.0),
)
WIND_TRUTH = 0.88

SolarCase = tuple[list[precompute.SolarArrays], pd.Series, pd.Series]
WindCase = tuple[list[precompute.WindArrays], pd.Series, pd.Series]


def covariates(year: int) -> DataFrame[precompute.Covariates]:
    """Every fifth hour of `year`, so that the hours go through every hour of the day and every
    season, with 100 MW."""
    hours = pd.date_range(
        f"{year}-01-01", f"{year + 1}-01-01", freq="5h", inclusive="left", tz="UTC", unit="ns"
    )
    frame = pd.DataFrame(
        {
            "complete": True,
            "capacity_mw": CAPACITY_MW,
            "capacity_provisional": False,
            "era5_source": "ERA5",
            "sun_elevation_deg": 30.0,
            "ghi_w_m2": 500.0,
            "clear_sky_ghi_w_m2": 700.0,
        },
        index=hours.rename("time"),
    )
    return precompute.Covariates.validate(frame)


def solar_year(year: int, rng: np.random.Generator) -> precompute.SolarArrays:
    hours = covariates(year)
    return precompute.SolarArrays(
        fleet_mwh=rng.uniform(0.0, 80.0, (len(hours), 2, precompute.DC_AC_GRID.size)),
        horizontal_mwh=rng.uniform(0.0, 80.0, (len(hours), precompute.DC_AC_GRID.size)),
        shares=np.array([system.share for system in PV.fleet]),
        covariates=hours,
    )


def wind_year(year: int, rng: np.random.Generator) -> precompute.WindArrays:
    """Each generation holds 50 MW, spread over 20 cells whose hub-height speeds scatter around
    a mean speed of the hour."""
    hours = covariates(year)
    cells = 20
    mean_speed = rng.uniform(2.0, 14.0, len(hours))
    capacity_mw = np.zeros((len(GENERATIONS), len(hours), precompute.SPEED_CENTRES.size))
    for generation in capacity_mw:
        speeds = mean_speed[:, np.newaxis] * rng.lognormal(0.0, 0.25, (len(hours), cells))
        bins = np.digitize(np.minimum(speeds, 39.0), precompute.SPEED_EDGES) - 1
        rows = np.repeat(np.arange(len(hours)), cells)
        np.add.at(generation, (rows, bins.ravel()), CAPACITY_MW / len(GENERATIONS) / cells)
    return precompute.WindArrays(capacity_mw=capacity_mw, generations=GENERATIONS, covariates=hours)


def with_errors(potential: pd.Series, rng: np.random.Generator) -> pd.Series:
    """`potential` with a noise of 0.01 in capacity factor, and ten times its value, a decimal
    shift, on one hour in a hundred."""
    values = potential.to_numpy() + rng.normal(0.0, 0.01 * CAPACITY_MW, potential.size)
    values[rng.choice(potential.size, potential.size // 100, replace=False)] *= 10
    return pd.Series(values, index=potential.index)


def solar_case(
    truth: tuple[float, float, float] = SOLAR_TRUTH, dc_ac_ratio: float = 1.2
) -> SolarCase:
    rng = np.random.default_rng(2019)
    arrays = [solar_year(year, rng) for year in (2019, 2020)]
    potential = pd.concat([precompute.solar_energy(part, dc_ac_ratio, *truth) for part in arrays])
    return arrays, with_errors(potential, rng), pd.Series(True, index=potential.index)


def wind_case(truth: float = WIND_TRUTH) -> WindCase:
    rng = np.random.default_rng(2020)
    arrays = [wind_year(year, rng) for year in (2019, 2020)]
    potential = pd.concat([precompute.wind_energy(part, truth) for part in arrays])
    return arrays, with_errors(potential, rng), pd.Series(True, index=potential.index)


def test_solar_parameters_are_recovered_despite_gross_errors() -> None:
    arrays, published_mwh, eligible = solar_case()

    fitted = calibration.fit_solar(arrays, published_mwh, eligible, PV)

    estimates = (fitted.level, fitted.horizontal_share, fitted.seasonal_amplitude)
    assert estimates == pytest.approx(SOLAR_TRUTH, abs=0.005)
    assert not fitted.level_at_bound
    assert not fitted.horizontal_share_at_bound
    assert not fitted.seasonal_amplitude_at_bound
    assert fitted.dc_ac_ratio == 1.2
    assert fitted.effective_gain == pytest.approx(fitted.level * 1.2 * (1 - PV.dc_losses))
    # The scale of the noise, which the errors on one hour in a hundred hardly move.
    assert fitted.fit.f_scale == pytest.approx(1.345 * 0.01, rel=0.05)
    assert fitted.fit.status > 0
    assert fitted.fit.eligible_hours == len(published_mwh)
    assert fitted.fit.training_years == (2019, 2020)


def test_the_solar_potential_takes_the_dc_ac_ratio_of_the_parameters() -> None:
    arrays, published_mwh, eligible = solar_case(dc_ac_ratio=1.35)

    fitted = calibration.fit_solar(
        arrays, published_mwh, eligible, dataclasses.replace(PV, dc_ac_ratio=1.35)
    )

    estimates = (fitted.level, fitted.horizontal_share, fitted.seasonal_amplitude)
    assert estimates == pytest.approx(SOLAR_TRUTH, abs=0.005)
    assert fitted.dc_ac_ratio == 1.35
    assert fitted.effective_gain == pytest.approx(fitted.level * 1.35 * (1 - PV.dc_losses))


def test_wind_speed_scale_is_recovered_despite_gross_errors() -> None:
    arrays, published_mwh, eligible = wind_case()

    fitted = calibration.fit_wind(arrays, published_mwh, eligible)

    assert fitted.speed_scale == pytest.approx(WIND_TRUTH, abs=0.005)
    assert not fitted.speed_scale_at_bound
    assert fitted.fit.f_scale == pytest.approx(1.345 * 0.01, rel=0.05)
    assert fitted.fit.status > 0
    assert fitted.fit.training_years == (2019, 2020)


def test_gross_errors_bias_plain_least_squares_but_not_the_huber_fit() -> None:
    arrays, published_mwh, eligible = solar_case()

    def residuals(x: FloatArray) -> FloatArray:
        potential = pd.concat([precompute.solar_energy(part, 1.2, *x) for part in arrays])
        return (potential - published_mwh).to_numpy() / CAPACITY_MW

    plain = least_squares(
        residuals,
        calibration.SOLAR_START,
        bounds=(calibration.SOLAR_LOWER, calibration.SOLAR_UPPER),
    )
    fitted = calibration.fit_solar(arrays, published_mwh, eligible, PV)

    # Ten times the value on 1 % of the hours raises the least-squares level by about 9 %.
    assert plain.x[0] - SOLAR_TRUTH[0] > 0.05
    assert fitted.level == pytest.approx(SOLAR_TRUTH[0], abs=0.005)


def normal_mad(values: FloatArray) -> float:
    """Median absolute deviation divided by the third quartile of the standard normal law."""
    return float(np.median(np.abs(values - np.median(values))) / norm.ppf(0.75))


def test_each_pass_takes_its_scale_from_the_residuals_at_its_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arrays, published_mwh, eligible = solar_case()
    passes: list[tuple[FloatArray, float, OptimizeResult]] = []

    def recorded(
        fun: Callable[[FloatArray], FloatArray], x0: FloatArray, **options: Any
    ) -> OptimizeResult:
        result = least_squares(fun, x0, **options)
        passes.append((np.array(x0), options["f_scale"], result))
        return result

    monkeypatch.setattr(calibration, "least_squares", recorded)
    fitted = calibration.fit_solar(arrays, published_mwh, eligible, PV)

    nominal = pd.concat([precompute.solar_energy(part, 1.2) for part in arrays])
    at_start = (nominal - published_mwh).to_numpy() / CAPACITY_MW
    (start, first_scale, first), (restart, final_scale, final) = passes
    assert start.tolist() == list(calibration.SOLAR_START)
    assert first_scale == pytest.approx(1.345 * normal_mad(at_start), rel=1e-12)
    assert restart.tolist() == first.x.tolist()
    assert final_scale == pytest.approx(1.345 * normal_mad(first.fun), rel=1e-12)
    assert fitted.fit.f_scale == final_scale
    assert [fitted.level, fitted.horizontal_share, fitted.seasonal_amplitude] == final.x.tolist()
    assert (fitted.fit.cost, fitted.fit.nfev, fitted.fit.status, fitted.fit.message) == (
        final.cost,
        final.nfev,
        final.status,
        final.message,
    )


def test_a_pass_that_does_not_converge_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def limited(
        fun: Callable[[FloatArray], FloatArray], x0: FloatArray, **options: Any
    ) -> OptimizeResult:
        return least_squares(fun, x0, **{**options, "max_nfev": 1})

    monkeypatch.setattr(calibration, "least_squares", limited)

    with pytest.raises(RuntimeError, match="did not converge"):
        calibration.fit_solar(*solar_case(), PV)


def with_hourly_capacity(case: SolarCase) -> SolarCase:
    """`case` with an AC capacity that changes from hour to hour, the energies unchanged."""
    arrays, published_mwh, eligible = case
    rng = np.random.default_rng(1)
    changed = [
        dataclasses.replace(
            part,
            covariates=precompute.Covariates.validate(
                part.covariates.assign(capacity_mw=rng.uniform(50.0, 150.0, len(part.covariates)))
            ),
        )
        for part in arrays
    ]
    return changed, published_mwh, eligible


def test_the_cost_is_the_huber_cost_of_the_capacity_factor_errors() -> None:
    arrays, published_mwh, eligible = with_hourly_capacity(solar_case())

    fitted = calibration.fit_solar(arrays, published_mwh, eligible, PV)

    values = (fitted.level, fitted.horizontal_share, fitted.seasonal_amplitude)
    potential = pd.concat([precompute.solar_energy(part, 1.2, *values) for part in arrays])
    capacity_mw = pd.concat([part.covariates["capacity_mw"] for part in arrays])
    errors = ((potential - published_mwh) / capacity_mw).abs().to_numpy()
    scale = fitted.fit.f_scale
    huber = np.where(errors <= scale, errors**2, 2 * scale * errors - scale**2)
    assert fitted.fit.cost == pytest.approx(huber.sum() / 2, rel=1e-12)


def test_a_level_beyond_its_bounds_is_flagged() -> None:
    arrays, published_mwh, eligible = solar_case(truth=(0.25, 0.55, 0.04))

    fitted = calibration.fit_solar(arrays, published_mwh, eligible, PV)

    assert fitted.level == pytest.approx(calibration.SOLAR_LOWER[0], abs=calibration.AT_BOUND)
    assert fitted.level_at_bound
    assert not fitted.horizontal_share_at_bound
    assert not fitted.seasonal_amplitude_at_bound


def test_a_seasonal_amplitude_beyond_its_bounds_is_flagged() -> None:
    arrays, published_mwh, eligible = solar_case(truth=(0.85, 0.55, 0.4))

    fitted = calibration.fit_solar(arrays, published_mwh, eligible, PV)

    upper = calibration.SOLAR_UPPER[2]
    assert fitted.seasonal_amplitude == pytest.approx(upper, abs=calibration.AT_BOUND)
    assert fitted.seasonal_amplitude_at_bound
    assert not fitted.level_at_bound
    assert not fitted.horizontal_share_at_bound


def test_a_speed_scale_beyond_its_bounds_is_flagged() -> None:
    arrays, published_mwh, eligible = wind_case(truth=0.45)

    fitted = calibration.fit_wind(arrays, published_mwh, eligible)

    assert fitted.speed_scale == pytest.approx(calibration.WIND_LOWER[0], abs=calibration.AT_BOUND)
    assert fitted.speed_scale_at_bound


def test_hours_of_2026_are_never_fitted() -> None:
    rng = np.random.default_rng(0)
    solar = [solar_year(2025, rng), solar_year(2026, rng)]
    wind = [wind_year(2025, rng), wind_year(2026, rng)]
    hours = pd.DatetimeIndex(pd.concat([part.covariates for part in solar]).index)
    published_mwh = pd.Series(10.0, index=hours)
    # Refused even when none of them is eligible.
    eligible = pd.Series(hours.year == 2025, index=hours)
    error = r"2026-01-01 00:00:00\+00:00 lies outside the training years 2019-2025"

    with pytest.raises(ValueError, match=error):
        calibration.fit_solar(solar, published_mwh, eligible, PV)
    with pytest.raises(ValueError, match=error):
        calibration.fit_wind(wind, published_mwh, eligible)


def test_fits_are_deterministic() -> None:
    solar = solar_case()
    wind = wind_case()

    assert calibration.fit_solar(*solar, PV) == calibration.fit_solar(*solar, PV)
    assert calibration.fit_wind(*wind) == calibration.fit_wind(*wind)


def only_2020_eligible(case: SolarCase) -> SolarCase:
    arrays, published_mwh, eligible = case
    return arrays, published_mwh, eligible & (pd.DatetimeIndex(eligible.index).year == 2020)


def test_only_eligible_hours_are_fitted() -> None:
    arrays, published_mwh, eligible = only_2020_eligible(solar_case())
    # Values of the other hours do not count.
    published_mwh[~eligible] = 0.0

    fitted = calibration.fit_solar(arrays, published_mwh, eligible, PV)

    assert fitted.level == pytest.approx(SOLAR_TRUTH[0], abs=0.01)
    assert fitted.fit.training_years == (2020,)
    assert fitted.fit.eligible_hours == eligible.sum()


def published_off_the_hours(case: SolarCase) -> SolarCase:
    arrays, published_mwh, eligible = case
    return arrays, published_mwh.iloc[1:], eligible


def eligibility_as_numbers(case: SolarCase) -> SolarCase:
    arrays, published_mwh, eligible = case
    return arrays, published_mwh, eligible.astype(float)


def nothing_eligible(case: SolarCase) -> SolarCase:
    arrays, published_mwh, eligible = case
    return arrays, published_mwh, ~eligible


def eligible_without_published_energy(case: SolarCase) -> SolarCase:
    arrays, published_mwh, eligible = case
    published_mwh.iloc[3] = np.nan
    return arrays, published_mwh, eligible


def eligible_without_complete_potential(case: SolarCase) -> SolarCase:
    arrays, published_mwh, eligible = case
    hours = arrays[0].covariates
    incomplete = precompute.Covariates.validate(
        hours.assign(complete=hours.index != hours.index[3])
    )
    return (
        [dataclasses.replace(arrays[0], covariates=incomplete), arrays[1]],
        published_mwh,
        eligible,
    )


def same_year_twice(case: SolarCase) -> SolarCase:
    arrays, published_mwh, eligible = case
    return [arrays[0], *arrays], published_mwh, eligible


@pytest.mark.parametrize(
    ("change", "error"),
    [
        (published_off_the_hours, "on the hours of the arrays"),
        (eligibility_as_numbers, "must be boolean"),
        (nothing_eligible, "no eligible hours"),
        (eligible_without_published_energy, "has no published energy, potential or capacity"),
        (eligible_without_complete_potential, "has no published energy, potential or capacity"),
        (same_year_twice, "more than once"),
    ],
)
def test_inputs_are_checked(change: Callable[[SolarCase], SolarCase], error: str) -> None:
    arrays, published_mwh, eligible = change(solar_case())

    with pytest.raises(ValueError, match=error):
        calibration.fit_solar(arrays, published_mwh, eligible, PV)
