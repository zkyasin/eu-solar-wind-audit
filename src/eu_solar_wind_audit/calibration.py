"""Calibrated potential of the evaluation plan ("Two model objects", item 2): the parameters of the
solar and onshore wind potential of `precompute`, fitted to the published output of the eligible
hours with a Huber loss.

For parameters x, the residual of an eligible hour t is

    r_t(x) = (p_t(x) - y_t) / C_t

with p_t(x) the potential in MWh, y_t the published energy in MWh and C_t the AC capacity in MW
at the start of the hour: an error in capacity factor, with the same weight for every hour. For a
scale c > 0, the cost is

    F(x; c) = 1/2 * sum over t of h_c(r_t(x)),  h_c(r) = r^2 if |r| <= c, else 2 c |r| - c^2,

minimised with scipy.optimize.least_squares over the box between the LOWER and UPPER constants
of the technology, such as SOLAR_LOWER and SOLAR_UPPER: method "trf", loss "huber" with f_scale c,
Jacobian by forward differences ("2-point"), the X_SCALE constant of the technology, and FTOL,
XTOL and GTOL. The scale comes from the residuals, in two passes:

1. c0 = 1.345 s(r(x0)), with x0 the START constant of the technology and s the median absolute
   deviation scaled to the standard deviation of a normal distribution,
   median(|r - median(r)|) / 0.6744897501960817 (scipy.stats.median_abs_deviation with
   scale="normal");
2. x1 minimises F(x; c0), starting from x0;
3. c1 = 1.345 s(r(x1));
4. x2 minimises F(x; c1), starting from x1.

The calibration is x2, with the scale c1 and the cost F(x2; c1). The constant 1.345 gives the
Huber estimator 95 % efficiency when the errors are normal (Huber 1981, Robust Statistics).

Solar: x = (level, horizontal share, seasonal amplitude) of precompute.solar_energy, at the
configured DC/AC ratio. Onshore wind: x = (speed scale) of precompute.wind_energy, without speed
offset and with the default smoothing. Fits refuse any hour outside evaluation.TRAINING_YEARS: 2026
is a prospective test.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import median_abs_deviation

from eu_solar_wind_audit import precompute
from eu_solar_wind_audit.evaluation import TRAINING_YEARS
from eu_solar_wind_audit.precompute import BoolArray, FloatArray
from eu_solar_wind_audit.pv import PVParameters

HUBER_TUNING = 1.345
# Termination tolerances of least_squares, set here so that the estimator does not change with
# the defaults of scipy. On 2019-2022, tolerances from 1e-8 to 1e-14 and x_scale values from 0.05
# to 1.2 move the fitted parameters by less than 1e-7.
FTOL = 1e-10
XTOL = 1e-10
GTOL = 1e-10
# A parameter closer than this to one of its bounds is reported as at that bound.
AT_BOUND = 1e-6

# Level, horizontal share and seasonal amplitude; the start is the nominal model.
SOLAR_START = (1.0, 0.0, 0.0)
SOLAR_LOWER = (0.3, 0.0, -0.3)
SOLAR_UPPER = (1.5, 1.0, 0.3)
SOLAR_X_SCALE = (1.0, 1.0, 1.0)
# Speed scale, from the nominal model.
WIND_START = (1.0,)
WIND_LOWER = (0.5,)
WIND_UPPER = (1.5,)
WIND_X_SCALE = (1.0,)


@dataclass(frozen=True)
class HuberFit:
    """Final pass of a fit: its scale c1, in capacity factor, its cost F(x2; c1), the nfev, status
    and message that least_squares gives for it, and the number of eligible hours and their UTC
    years."""

    f_scale: float
    cost: float
    nfev: int
    status: int
    message: str
    eligible_hours: int
    training_years: tuple[int, ...]


@dataclass(frozen=True)
class SolarCalibration:
    """Parameters of precompute.solar_energy at `dc_ac_ratio`, each flagged when it lies within
    AT_BOUND of a bound. `effective_gain` is level * dc_ac_ratio * (1 - DC losses): outside the
    hours when inverters clip, the data identify this product, not its factors."""

    level: float
    horizontal_share: float
    seasonal_amplitude: float
    level_at_bound: bool
    horizontal_share_at_bound: bool
    seasonal_amplitude_at_bound: bool
    dc_ac_ratio: float
    effective_gain: float
    fit: HuberFit


@dataclass(frozen=True)
class WindCalibration:
    """Speed scale of precompute.wind_energy, flagged when it lies within AT_BOUND of a bound."""

    speed_scale: float
    speed_scale_at_bound: bool
    fit: HuberFit


def fit_solar(
    arrays: Sequence[precompute.SolarArrays],
    published_mwh: pd.Series,
    eligible: pd.Series,
    parameters: PVParameters,
) -> SolarCalibration:
    """Solar calibration on the eligible hours of `arrays`, which cover consecutive periods such
    as the UTC years that precompute.solar gives. `published_mwh` and the boolean `eligible` are
    indexed on the hours of the arrays, in the same order. `parameters` are the PV parameters the
    arrays were computed with: the potential takes their DC/AC ratio, and the effective gain their
    DC losses.

    Raises ValueError if an hour lies outside evaluation.TRAINING_YEARS, if the inputs are not on
    the same hours, or if an eligible hour has no published energy, potential or capacity.
    """

    def potential(x: FloatArray) -> FloatArray:
        level, horizontal_share, seasonal_amplitude = x
        # Each period is evaluated apart, so that the arrays are never copied.
        return np.concatenate(
            [
                precompute.solar_energy(
                    part,
                    parameters.dc_ac_ratio,
                    level=level,
                    horizontal_share=horizontal_share,
                    seasonal_amplitude=seasonal_amplitude,
                ).to_numpy()
                for part in arrays
            ]
        )

    values, at_bound, fit = _huber_fit(
        potential,
        [part.covariates for part in arrays],
        published_mwh,
        eligible,
        SOLAR_START,
        SOLAR_LOWER,
        SOLAR_UPPER,
        SOLAR_X_SCALE,
    )
    level, horizontal_share, seasonal_amplitude = values
    return SolarCalibration(
        level=level,
        horizontal_share=horizontal_share,
        seasonal_amplitude=seasonal_amplitude,
        level_at_bound=at_bound[0],
        horizontal_share_at_bound=at_bound[1],
        seasonal_amplitude_at_bound=at_bound[2],
        dc_ac_ratio=parameters.dc_ac_ratio,
        effective_gain=level * parameters.dc_ac_ratio * (1 - parameters.dc_losses),
        fit=fit,
    )


def fit_wind(
    arrays: Sequence[precompute.WindArrays], published_mwh: pd.Series, eligible: pd.Series
) -> WindCalibration:
    """Onshore wind calibration on the eligible hours of `arrays`, which cover consecutive periods
    such as the UTC years that precompute.wind_onshore gives. `published_mwh` and the boolean
    `eligible` are indexed on the hours of the arrays, in the same order.

    Raises ValueError as `fit_solar` does.
    """

    def potential(x: FloatArray) -> FloatArray:
        (speed_scale,) = x
        return np.concatenate(
            [precompute.wind_energy(part, speed_scale).to_numpy() for part in arrays]
        )

    (speed_scale,), (at_bound,), fit = _huber_fit(
        potential,
        [part.covariates for part in arrays],
        published_mwh,
        eligible,
        WIND_START,
        WIND_LOWER,
        WIND_UPPER,
        WIND_X_SCALE,
    )
    return WindCalibration(speed_scale=speed_scale, speed_scale_at_bound=at_bound, fit=fit)


def _huber_fit(
    potential: Callable[[FloatArray], FloatArray],
    covariates: Sequence[pd.DataFrame],
    published_mwh: pd.Series,
    eligible: pd.Series,
    start: tuple[float, ...],
    lower: tuple[float, ...],
    upper: tuple[float, ...],
    x_scale: tuple[float, ...],
) -> tuple[list[float], list[bool], HuberFit]:
    """The estimator of the module docstring for `potential`, the energy in MWh over every hour
    of `covariates`. Gives the parameters, whether each lies at a bound, and the final pass."""
    hours = pd.concat(covariates)
    training = _training_hours(hours, published_mwh, eligible)
    published: FloatArray = published_mwh.to_numpy(dtype=float)[training]
    capacity_mw: FloatArray = hours["capacity_mw"].to_numpy(dtype=float)[training]

    def residuals(x: FloatArray) -> FloatArray:
        return (potential(x)[training] - published) / capacity_mw

    x = np.array(start)
    # The two passes of the module docstring: each takes its scale from the residuals at its
    # start.
    for _ in range(2):
        f_scale = HUBER_TUNING * median_abs_deviation(residuals(x), scale="normal")
        result = least_squares(
            residuals,
            x,
            jac="2-point",
            bounds=(lower, upper),
            method="trf",
            ftol=FTOL,
            xtol=XTOL,
            gtol=GTOL,
            x_scale=np.array(x_scale),
            loss="huber",
            f_scale=f_scale,
        )
        x = result.x

    at_bound = (x - np.array(lower) <= AT_BOUND) | (np.array(upper) - x <= AT_BOUND)
    years = pd.DatetimeIndex(hours.index)[training].year
    fit = HuberFit(
        f_scale=float(f_scale),
        cost=float(result.cost),
        nfev=int(result.nfev),
        status=int(result.status),
        message=str(result.message),
        eligible_hours=int(training.sum()),
        training_years=tuple(int(year) for year in np.unique(years)),
    )
    return x.tolist(), at_bound.tolist(), fit


def _training_hours(
    hours: pd.DataFrame, published_mwh: pd.Series, eligible: pd.Series
) -> BoolArray:
    """Mask of the hours to fit among `hours`, the joined covariates of the arrays. Raises
    ValueError on the inputs that fit_solar refuses."""
    time = pd.DatetimeIndex(hours.index)
    if not time.is_unique:
        raise ValueError("the arrays cover some hours more than once")
    outside = time[~time.year.isin(TRAINING_YEARS)]
    if not outside.empty:
        first, last = TRAINING_YEARS[0], TRAINING_YEARS[-1]
        raise ValueError(f"{outside[0]} lies outside the training years {first}-{last}")
    if not (published_mwh.index.equals(time) and eligible.index.equals(time)):
        raise ValueError("published energy and eligibility must be on the hours of the arrays")
    if eligible.dtype != bool:
        raise ValueError("eligibility must be boolean")
    training: BoolArray = eligible.to_numpy()
    if not training.any():
        raise ValueError("no eligible hours")
    usable = (
        hours["complete"].to_numpy()
        & published_mwh.notna().to_numpy()
        & (hours["capacity_mw"].to_numpy() > 0)
    )
    unusable = time[training & ~usable]
    if not unusable.empty:
        raise ValueError(
            f"eligible hour {unusable[0]} has no published energy, potential or capacity"
        )
    return training
