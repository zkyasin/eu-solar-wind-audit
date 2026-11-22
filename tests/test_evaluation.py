import numpy as np
import pandas as pd
import pytest
from pandera.errors import SchemaError, SchemaErrors

from eu_solar_wind_audit.evaluation import (
    SEASON_OF_MONTH,
    Metrics,
    Predictions,
    climatology,
    mean_absolute_by_period,
    metrics,
)


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
        pytest.param(
            pd.concat([valid_predictions().iloc[[0]], valid_predictions()]),
            "duplicate",
            id="duplicate-hour",
        ),
        pytest.param(valid_predictions().tz_localize(None), r"datetime64\[ns\]", id="naive"),
        pytest.param(valid_predictions().rename_axis(index=None), "name 'time'", id="unnamed"),
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
        published_mwh=[20.0, 40.0, 270.0, 30.0, 1.0, 1.0],
        capacity_mw=[100.0, 100.0, 300.0, 300.0, 50.0, 80.0],
        eligible=[True] * 6,
    )

    held_out = climatology(hours, 2021)

    # January at 10:00: factors 0.2, 0.4 and 0.9, whose mean is 0.5 (their median is 0.4, and a
    # ratio of sums would give 330 / 500 = 0.66). Each hour takes its own capacity.
    assert held_out.tolist() == pytest.approx([0.5 * 50, 0.1 * 80])
    assert held_out.index.equals(hours.index[4:])
    assert held_out.name == "climatology_mwh"


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


# Hourly capacity factors 0.1 to 0.6 on 100 MW.
PUBLISHED = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]


def value(
    table: pd.DataFrame, model: str, metric: str, group: str = "all", year: int = 2019
) -> float:
    """The one value of `table` for this model, metric, group and held-out year."""
    row = table[
        (table["held_out_year"] == year)
        & (table["model"] == model)
        & (table["metric"] == metric)
        & (table["group"] == group)
    ]
    return float(row["value"].item())


def test_a_model_equal_to_the_published_series_has_no_error() -> None:
    table = metrics(predictions("2019-06-01 06:00", PUBLISHED, same=PUBLISHED), "solar")

    assert value(table, "same", "correlation") == pytest.approx(1.0)
    assert value(table, "same", "rmse_cf") == 0.0
    assert value(table, "same", "bias") == 0.0


def test_a_constant_multiple_keeps_the_correlation_and_has_that_bias() -> None:
    scaled = [1.25 * mwh for mwh in PUBLISHED]

    table = metrics(predictions("2019-06-01 06:00", PUBLISHED, scaled=scaled), "solar")

    assert value(table, "scaled", "correlation") == pytest.approx(1.0)
    assert value(table, "scaled", "bias") == pytest.approx(0.25)
    # The error is a quarter of each capacity factor.
    expected_rmse = 0.25 * np.sqrt(np.mean(np.square([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])))
    assert value(table, "scaled", "rmse_cf") == pytest.approx(expected_rmse)


def test_a_series_shifted_by_one_hour() -> None:
    # Capacity factors 0, 0.5, 1, 0.5 and 0, against the same curve one hour later.
    published = [0.0, 50.0, 100.0, 50.0, 0.0]
    later = [0.0, 0.0, 50.0, 100.0, 50.0]

    table = metrics(predictions("2019-06-01 08:00", published, later=later), "solar")

    # Deviations from the mean 0.4: -0.4, 0.1, 0.6, 0.1, -0.4 and -0.4, -0.4, 0.1, 0.6, 0.1.
    assert value(table, "later", "correlation") == pytest.approx(0.2 / 0.7)
    # Errors 0, -0.5, -0.5, 0.5 and 0.5.
    assert value(table, "later", "rmse_cf") == pytest.approx(np.sqrt(0.2))
    assert value(table, "later", "bias") == 0.0


def test_capacity_factors_use_the_capacity_of_each_hour() -> None:
    frame = predictions(
        "2019-06-01 10:00", [50.0, 25.0, 10.0], [100.0, 50.0, 100.0], model=[60.0, 35.0, 10.0]
    )

    table = metrics(frame, "solar")

    # Capacity factor errors 0.1, 0.2 and 0.
    assert value(table, "model", "rmse_cf") == pytest.approx(np.sqrt(0.05 / 3))


def test_correlation_is_between_capacity_factors() -> None:
    # Modelled capacity factors are the published ones, 0.2, 0.4 and 0.6, plus 0.1. The energies
    # are not linearly related, since the capacity changes from hour to hour.
    frame = predictions(
        "2019-06-01 10:00", [20.0, 80.0, 60.0], [100.0, 200.0, 100.0], model=[30.0, 100.0, 70.0]
    )

    assert value(metrics(frame, "solar"), "model", "correlation") == pytest.approx(1.0)


def test_only_eligible_hours_are_evaluated() -> None:
    frame = predictions("2019-06-01 06:00", [*PUBLISHED, 70.0], model=[*PUBLISHED, 700.0])
    frame.loc[frame.index[-1], "eligible"] = False

    table = metrics(frame, "wind_onshore")

    assert value(table, "model", "rmse_cf") == 0.0
    assert value(table, "model", "bias") == 0.0
    assert set(table["hours"]) == {6}


def test_each_held_out_year_is_evaluated_apart() -> None:
    frame = pd.concat(
        [
            predictions("2019-06-01 06:00", PUBLISHED, model=[1.1 * mwh for mwh in PUBLISHED]),
            predictions("2020-06-01 06:00", PUBLISHED, model=[0.8 * mwh for mwh in PUBLISHED]),
        ]
    )

    table = metrics(frame, "solar")

    assert value(table, "model", "bias", year=2019) == pytest.approx(0.1)
    assert value(table, "model", "bias", year=2020) == pytest.approx(-0.2)
    assert table["technology"].eq("solar").all()
    assert list(table.columns) == [
        "technology",
        "held_out_year",
        "model",
        "metric",
        "group",
        "value",
        "hours",
    ]


def test_metrics_need_eligible_hours() -> None:
    frame = predictions("2019-06-01 06:00", PUBLISHED, model=PUBLISHED).assign(eligible=False)

    with pytest.raises(ValueError, match="no eligible hours"):
        metrics(frame, "solar")


def test_metrics_validate_the_predictions() -> None:
    with pytest.raises(SchemaError, match="eligible_hours_have_every_value"):
        metrics(without(["fitted_mwh"], 2), "solar")


def test_metric_table_holds_one_value_per_metric() -> None:
    table = metrics(predictions("2019-06-01 06:00", PUBLISHED, model=PUBLISHED), "solar")

    with pytest.raises(SchemaError, match="not unique"):
        Metrics.validate(pd.concat([table, table.iloc[[0]]], ignore_index=True))


def two_seasons() -> pd.DataFrame:
    """Two hours in January and two in July 2019, the first one clear; the model is 20 % high in
    January and 10 % low in July."""
    frame = pd.concat(
        [
            predictions("2019-01-15 11:00", [100.0, 100.0], 200.0, model=[120.0, 120.0]),
            predictions("2019-07-15 11:00", [300.0, 300.0], 400.0, model=[270.0, 270.0]),
        ]
    )
    frame["sky_class"] = ["clear", "overcast", "overcast", "overcast"]
    return frame


def test_bias_by_season_and_sky_class() -> None:
    table = metrics(two_seasons(), "solar")

    assert value(table, "model", "bias", "DJF") == pytest.approx(0.2)
    assert value(table, "model", "bias", "JJA") == pytest.approx(-0.1)
    assert value(table, "model", "bias", "clear") == pytest.approx(0.2)
    assert value(table, "model", "bias", "overcast") == pytest.approx(660 / 700 - 1)
    assert value(table, "model", "bias") == pytest.approx(780 / 800 - 1)
    # Groups without eligible hours, such as spring here, get no row.
    hours = table[table["metric"] == "bias"].set_index("group")["hours"]
    assert hours.to_dict() == {"all": 4, "DJF": 2, "JJA": 2, "clear": 1, "overcast": 3}


def test_group_biases_relative_to_the_bias_of_the_year() -> None:
    frame = two_seasons()
    frame["scaled_mwh"] = 1.25 * frame["published_mwh"]

    table = metrics(frame, "solar")

    # The model is 2.5 % low over the year: 780 MWh against 800.
    assert value(table, "model", "bias_relative_to_year", "DJF") == pytest.approx(1.2 / 0.975 - 1)
    assert value(table, "model", "bias_relative_to_year", "JJA") == pytest.approx(0.9 / 0.975 - 1)
    overcast = (660 / 700) / 0.975 - 1
    assert value(table, "model", "bias_relative_to_year", "overcast") == pytest.approx(overcast)
    # A constant multiple has the bias of its year in every group.
    relative = table[(table["model"] == "scaled") & (table["metric"] == "bias_relative_to_year")]
    assert relative["value"].to_numpy() == pytest.approx(0.0)
    assert sorted(relative["group"]) == ["DJF", "JJA", "clear", "overcast"]


def annual_biases(by_year: dict[int, float]) -> pd.DataFrame:
    """`Metrics` table of the annual biases of a solar model, each over 100 hours."""
    return pd.DataFrame(
        {
            "technology": "solar",
            "held_out_year": list(by_year),
            "model": "fitted",
            "metric": "bias",
            "group": "all",
            "value": list(by_year.values()),
            "hours": 100,
        }
    )


def test_each_period_is_summarised_by_the_mean_absolute_value_of_its_folds() -> None:
    table = annual_biases({2019: 0.1, 2020: -0.3, 2021: 0.02, 2023: -0.04, 2024: 0.06, 2026: 0.5})

    summary = mean_absolute_by_period(table)

    # The prospective 2026 fold belongs to no period.
    assert summary["period"].tolist() == ["2019-2022", "2023-2025"]
    assert summary["mean_absolute_value"].tolist() == pytest.approx([0.14, 0.05])
    assert summary["folds"].tolist() == [3, 2]
    assert summary["hours"].tolist() == [300, 200]


def test_undefined_values_are_left_out_of_the_summary() -> None:
    summary = mean_absolute_by_period(annual_biases({2019: np.nan, 2020: -0.2}))

    assert summary["mean_absolute_value"].tolist() == pytest.approx([0.2])
    assert summary["folds"].tolist() == [1]


def test_models_metrics_and_groups_are_summarised_apart() -> None:
    table = pd.concat(
        [
            annual_biases({2019: 0.1}),
            annual_biases({2019: 0.3}).assign(model="raw"),
            annual_biases({2019: 0.2}).assign(group="DJF"),
            annual_biases({2019: 0.4}).assign(metric="rmse_cf"),
        ],
        ignore_index=True,
    )

    summary = mean_absolute_by_period(table)

    assert summary["mean_absolute_value"].tolist() == pytest.approx([0.1, 0.3, 0.2, 0.4])
    assert set(summary["folds"]) == {1}
