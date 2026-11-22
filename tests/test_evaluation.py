import numpy as np
import pandas as pd
import pytest
from pandera.errors import SchemaError, SchemaErrors

from eu_solar_wind_audit.evaluation import SEASON_OF_MONTH, Predictions


def predictions(
    start: str,
    published_mwh: list[float],
    capacity_mw: float | list[float] = 100.0,
    **models: list[float],
) -> pd.DataFrame:
    """Eligible clear-sky hours from `start` in UTC, with one column per model."""
    time = pd.date_range(
        start, periods=len(published_mwh), freq="h", tz="UTC", unit="ns", name="time"
    )
    return pd.DataFrame(
        {
            "held_out_year": time.year.astype("int64"),
            "published_mwh": published_mwh,
            "capacity_mw": capacity_mw,
            **{f"{model}_mwh": values for model, values in models.items()},
            "eligible": True,
            "season": time.month.map(SEASON_OF_MONTH),
            "sky_class": "clear",
        },
        index=time,
    )


def valid_predictions() -> pd.DataFrame:
    # Two hours of the 2019 fold, one of the 2020 fold.
    return predictions(
        "2019-12-31 22:00", [10.0, 20.0, 30.0], raw=[12.0, 22.0, 33.0], fitted=[9.0, 21.0, 30.0]
    )


def without(columns: list[str], hour: int) -> pd.DataFrame:
    """`valid_predictions` with no value in `columns` at its `hour`-th hour."""
    frame = valid_predictions()
    frame.loc[frame.index[hour], columns] = np.nan
    return frame


def test_predictions_accept_a_valid_frame() -> None:
    Predictions.validate(valid_predictions())


def test_hours_that_are_not_eligible_may_lack_values() -> None:
    frame = without(["published_mwh", "raw_mwh", "sky_class"], 0)
    frame["eligible"] = [False, True, True]

    Predictions.validate(frame)


@pytest.mark.parametrize(
    ("frame", "error"),
    [
        pytest.param(
            valid_predictions().drop(columns=["raw_mwh", "fitted_mwh"]),
            "did not match any columns",
            id="no-model",
        ),
        pytest.param(valid_predictions().assign(raw_mwh="12"), "float64", id="model-not-a-number"),
        pytest.param(
            valid_predictions().assign(held_out_year=2019),
            "held_out_year_is_the_utc_year",
            id="year-of-another-fold",
        ),
        pytest.param(
            valid_predictions().assign(season="MAM"),
            "season_of_the_utc_month",
            id="season-of-another-month",
        ),
        pytest.param(valid_predictions().assign(sky_class="sunny"), "isin", id="unknown-sky"),
        pytest.param(
            without(["published_mwh"], 1),
            "eligible_hours_have_every_value",
            id="eligible-without-published",
        ),
        pytest.param(
            without(["fitted_mwh"], 2),
            "eligible_hours_have_every_value",
            id="eligible-without-a-model",
        ),
        pytest.param(
            without(["sky_class"], 1),
            "eligible_hours_have_every_value",
            id="eligible-without-sky-class",
        ),
        pytest.param(
            valid_predictions().assign(capacity_mw=[100.0, 0.0, 100.0]),
            "greater_than",
            id="no-capacity",
        ),
        pytest.param(
            valid_predictions().set_axis(
                pd.date_range(
                    "2019-12-31 22:30", periods=3, freq="h", tz="UTC", unit="ns", name="time"
                )
            ),
            "on_the_hour",
            id="off-hour",
        ),
        pytest.param(valid_predictions().tz_localize(None), r"datetime64\[ns\]", id="naive"),
    ],
)
def test_predictions_reject(frame: pd.DataFrame, error: str) -> None:
    with pytest.raises(SchemaError, match=error):
        Predictions.validate(frame)


def test_predictions_reject_an_undeclared_column() -> None:
    # pandera reports an undeclared column as SchemaErrors, even without lazy validation.
    with pytest.raises(SchemaErrors, match="price"):
        Predictions.validate(valid_predictions().assign(price=50.0))
