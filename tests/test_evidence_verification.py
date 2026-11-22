"""Corrupt saved synthetic outputs to check that evidence verification detects it."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from broadway.modelling import metric_values, reduction, save_predictions, subgroup_tables

spec = importlib.util.spec_from_file_location("artifact_verifier", Path(__file__).parents[1] / "scripts/verify_artifacts.py")
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


@pytest.fixture
def synthetic_evidence(tmp_path):
    frame = pd.DataFrame({"show": ["Synthetic"] * 3, "theatre": ["Synthetic venue"] * 3,
        "week_ending": pd.to_datetime(["2018-01-07", "2018-01-14", "2018-01-21"]),
        "inferred_run": [1] * 3, "run_weeks_known": [4, 5, 6],
        "weekly_gross": [100., 120., 90.], "weekly_gross_lag1": [110., 100., 120.]})
    predictions = pd.concat([save_predictions(frame, "naive", frame.weekly_gross_lag1),
                             save_predictions(frame, "synthetic_ml", [105., 115., 95.])], ignore_index=True)
    predictions.to_csv(tmp_path / "final_predictions.csv", index=False)
    baseline_mae = np.mean(np.abs(frame.weekly_gross_lag1 - frame.weekly_gross))
    metrics = []
    for model, group in predictions.groupby("model"):
        values = metric_values(group.weekly_gross, group.prediction_usd)
        improvement = reduction(baseline_mae, values["mae_usd"])
        metrics.append({"model": model, "rows": 3, "weeks": 3, **values,
                        "mae_reduction_vs_naive_pct": improvement, "meets_example_kpi": improvement >= 5})
    pd.DataFrame(metrics).to_csv(tmp_path / "final_metrics.csv", index=False)
    subgroup_tables(predictions, tmp_path, "final")
    return tmp_path, frame


def test_saved_synthetic_final_outputs_verify_without_model_scoring(synthetic_evidence):
    folder, frame = synthetic_evidence
    assert verifier.check_final_tables(folder, frame, {"naive", "synthetic_ml"}) == (3, 3)


def test_corrupt_final_metric_is_detected(synthetic_evidence):
    folder, frame = synthetic_evidence
    metrics = pd.read_csv(folder / "final_metrics.csv")
    metrics.loc[metrics.model == "synthetic_ml", "mae_usd"] += 1
    metrics.to_csv(folder / "final_metrics.csv", index=False)
    with pytest.raises(ValueError, match="differs from recomputed"):
        verifier.check_final_tables(folder, frame, {"naive", "synthetic_ml"})


def test_different_model_keys_are_detected(synthetic_evidence):
    folder, frame = synthetic_evidence
    predictions = pd.read_csv(folder / "final_predictions.csv")
    predictions.loc[predictions.model == "synthetic_ml", "show"] = "Wrong show"
    predictions.to_csv(folder / "final_predictions.csv", index=False)
    with pytest.raises(ValueError, match="Forecast keys differ"):
        verifier.check_final_tables(folder, frame, {"naive", "synthetic_ml"})


def test_corrupt_subgroup_count_is_detected(synthetic_evidence):
    folder, frame = synthetic_evidence
    subgroups = pd.read_csv(folder / "final_subgroups_theatre.csv")
    subgroups.loc[0, "rows"] += 1
    subgroups.to_csv(folder / "final_subgroups_theatre.csv", index=False)
    with pytest.raises(ValueError, match="differs from recomputed"):
        verifier.check_final_tables(folder, frame, {"naive", "synthetic_ml"})


@pytest.mark.parametrize("damage", [None, "score", "grid", "selection", "decision"])
def test_validation_summary_integrity(damage):
    # Hand-authored three-fold fixture: Ridge 10 beats naive by 10%, all others lose.
    records = []
    for model, params_list in {"naive": [{}], **verifier.GRID}.items():
        for params in params_list:
            mae = 100. if model == "naive" else (90. if model == "ridge" and params["alpha"] == 10. else 110.)
            for _, year in verifier.FOLDS:
                records.append({"model": model, "parameters": verifier.json.dumps(params, sort_keys=True),
                    "validation_year": year, "mae_usd": mae, "rmse_usd": mae + 10., "mean_signed_error_usd": -5.})
    tuning = pd.DataFrame(records)
    summary = tuning.drop_duplicates(["model", "parameters"]).drop(columns="validation_year").rename(columns={
        "mae_usd": "mean_validation_mae_usd", "rmse_usd": "mean_validation_rmse_usd",
        "mean_signed_error_usd": "mean_validation_signed_error_usd"})
    summary["folds"] = 3
    summary["mae_reduction_vs_naive_pct"] = 100 - summary.mean_validation_mae_usd
    frozen = {"chosen_parameters": {"ridge": {"alpha": 10.},
        "hist_gradient_boosting": {"learning_rate": 0.05, "max_leaf_nodes": 15}}, "validation_selected_method": "ridge"}
    if damage == "score":
        summary.iloc[0, summary.columns.get_loc("mean_validation_rmse_usd")] += 1
    elif damage == "grid":
        tuning.iloc[0, tuning.columns.get_loc("parameters")] = '{"unexpected": 1}'
    elif damage == "selection":
        frozen["chosen_parameters"]["ridge"] = {"alpha": 1.}
    elif damage == "decision":
        frozen["validation_selected_method"] = "naive"
    if damage is None:
        verifier.check_validation_summary(tuning, summary, frozen)
    else:
        with pytest.raises(ValueError):
            verifier.check_validation_summary(tuning, summary, frozen)


def test_consistent_predictions_with_wrong_source_actuals_are_detected(synthetic_evidence):
    _, source = synthetic_evidence
    baseline = source.copy()
    baseline["weekly_gross"] += 1
    with pytest.raises(ValueError, match="weekly_gross differs from source"):
        verifier.check_source_rows(baseline, source, "Validation")
