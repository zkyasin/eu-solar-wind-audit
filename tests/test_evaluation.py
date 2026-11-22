import numpy as np
import pandas as pd
import pytest
from pandera.errors import SchemaError, SchemaErrors

from eu_solar_wind_audit.evaluation import SEASON_OF_MONTH, Predictions, climatology


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


def observed(
    stamps: list[str], published_mwh: list[float], capacity_mw: list[float], eligible: list[bool]
) -> pd.DataFrame:
    time = pd.DatetimeIndex(stamps, tz="UTC", name="time").as_unit("ns")
    return pd.DataFrame(
        {"published_mwh": published_mwh, "capacity_mw": capacity_mw, "eligible": eligible},
        index=time,
    )


def test_climatology_is_the_mean_capacity_factor_of_the_other_years() -> None:
    hours = observed(
        [
            "2019-01-01 10:00",
            "2019-01-02 10:00",
            "2020-01-01 10:00",
            "2020-01-01 11:00",
            "2021-01-05 10:00",
            "2021-01-05 11:00",
        ],
        published_mwh=[20.0, 40.0, 180.0, 30.0, 1.0, 1.0],
        capacity_mw=[100.0, 100.0, 300.0, 300.0, 50.0, 50.0],
        eligible=[True] * 6,
    )

    expected = climatology(hours, 2021)

    # January at 10:00: factors 0.2, 0.4 and 0.6 (a ratio of sums would give 240 / 500 = 0.48).
    assert expected.tolist() == pytest.approx([0.4 * 50, 0.1 * 50])
    assert expected.index.equals(hours.index[4:])
    assert expected.name == "climatology_mwh"


def test_climatology_learns_from_eligible_hours_of_other_years_only() -> None:
    hours = observed(
        ["2019-07-01 12:00", "2019-07-02 12:00", "2020-07-01 12:00", "2020-07-02 12:00"],
        published_mwh=[50.0, 90.0, 70.0, 10.0],
        capacity_mw=[100.0] * 4,
        eligible=[True, False, True, True],
    )

    # The second hour of 2019 is not eligible and 2020 is held out.
    assert climatology(hours, 2020).tolist() == pytest.approx([50.0, 50.0])


def test_climatology_is_missing_without_another_year_in_the_same_month_and_hour() -> None:
    hours = observed(
        ["2019-01-01 10:00", "2020-01-01 11:00", "2020-02-01 10:00"],
        published_mwh=[20.0, 1.0, 1.0],
        capacity_mw=[100.0] * 3,
        eligible=[True] * 3,
    )

    assert climatology(hours, 2020).isna().all()


def test_climatology_of_each_fold_is_a_model_of_the_predictions() -> None:
    frame = pd.concat(
        [
            predictions("2019-06-01 10:00", [10.0, 20.0, 30.0], raw=[1.0, 1.0, 1.0]),
            predictions("2020-06-01 10:00", [40.0, 50.0, 60.0], 200.0, raw=[1.0, 1.0, 1.0]),
        ]
    )

    frame["climatology_mwh"] = pd.concat([climatology(frame, year) for year in (2019, 2020)])

    Predictions.validate(frame)
    # 2019 takes the capacity factors of 2020 and the other way round.
    assert frame["climatology_mwh"].tolist() == pytest.approx([20.0, 25.0, 30.0, 20.0, 40.0, 60.0])


@pytest.mark.parametrize("tz", [None, "Europe/Paris"])
def test_climatology_needs_hours_in_utc(tz: str | None) -> None:
    hours = observed(["2019-01-01 10:00"], [1.0], [10.0], [True]).tz_convert(tz)

    with pytest.raises(ValueError, match="DatetimeIndex in UTC"):
        climatology(hours, 2020)
