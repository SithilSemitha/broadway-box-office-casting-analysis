"""Expanding chronological validation; explicit, frozen, once-only final evaluation."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

from .config import (CATEGORICAL, FEATURES, FINAL_END, FINAL_START, FOLDS, GRID, KEYS,
                     KPI_MAE_REDUCTION_PCT, MODELS, NUMERIC, PREDICTORS, ROOT, TABLES, TARGET)
from .eda import read_features
from .io import parquet_fingerprint, read_json, sha256, utc_now, versions, write_json


def build_pipeline(model_name, params):
    numeric_steps = [("impute", SimpleImputer(strategy="median", keep_empty_features=True))]
    if model_name == "ridge":
        numeric_steps.append(("scale", StandardScaler()))
    transform = ColumnTransformer([
        ("numeric", Pipeline(numeric_steps), NUMERIC),
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
    ], remainder="drop", sparse_threshold=0)
    if model_name == "ridge":
        estimator = Ridge(**params)
    elif model_name == "hist_gradient_boosting":
        estimator = HistGradientBoostingRegressor(**params, max_iter=150, loss="absolute_error",
                                                  early_stopping=False, random_state=42)
    else:
        raise ValueError(model_name)
    return Pipeline([("transform", transform), ("model", estimator)])


def metric_values(actual, prediction):
    return {"mae_usd": float(mean_absolute_error(actual, prediction)),
            "rmse_usd": float(np.sqrt(mean_squared_error(actual, prediction))),
            "mean_signed_error_usd": float(np.mean(np.asarray(prediction) - np.asarray(actual)))}


def reduction(baseline_mae, model_mae):
    return 100 * (baseline_mae - model_mae) / baseline_mae if baseline_mae != 0 else None


def chronological_fold(frame, train_end, validation_year):
    train = frame.loc[frame.week_ending < pd.Timestamp(f"{train_end + 1}-01-01")]
    valid = frame.loc[(frame.week_ending >= pd.Timestamp(f"{validation_year}-01-01"))
                      & (frame.week_ending < pd.Timestamp(f"{validation_year + 1}-01-01"))]
    if train.empty or valid.empty:
        raise ValueError(f"Empty chronological fold {train_end}/{validation_year}")
    if train.week_ending.max() >= valid.week_ending.min():
        raise ValueError("Chronological train/validation overlap")
    assert set(train.week_ending).isdisjoint(set(valid.week_ending))
    return train, valid


def protocol():
    return {"predictors": PREDICTORS, "target": TARGET, "folds": FOLDS, "grid": GRID,
        "primary_metric": "mean of the three fold MAEs, equal fold weights",
        "kpi_reduction_pct": KPI_MAE_REDUCTION_PCT,
        "selection_rule": "Choose minimum mean validation MAE per model; exact ties use sorted parameter JSON. Choose best ML method only if validation reduction >= KPI; otherwise retain naive benchmark.",
        "fixed_hgb": {"max_iter": 150, "loss": "absolute_error", "early_stopping": False, "random_state": 42},
        "final_start": FINAL_START, "final_end": FINAL_END,
        "availability": "One-week rolling origin; prior reported outcomes available, including prior holdout weeks, without refitting.",
        "eligibility": "Observed target with four consecutive earlier weeks in same inferred show-theatre run",
        "prediction_postprocessing": "None; negative predictions remain in evaluation if produced."}


def code_fingerprints():
    files = list((ROOT / "src/broadway").glob("*.py")) + [ROOT / "requirements.txt"]
    return {str(path.relative_to(ROOT)): sha256(path) for path in sorted(files)}


def save_predictions(rows, model, prediction, fold=None):
    result = rows[KEYS + ["inferred_run", "run_weeks_known", TARGET, "weekly_gross_lag1"]].copy()
    result["model"] = model
    result["prediction_usd"] = prediction
    result["error_usd"] = result.prediction_usd - result[TARGET]
    result["absolute_error_usd"] = result.error_usd.abs()
    if fold is not None:
        result["validation_year"] = fold
    return result


def subgroup_tables(predictions, folder, prefix):
    predictions = predictions.copy()
    predictions["known_run_age_band"] = pd.cut(predictions.run_weeks_known, [3, 12, 52, 260, np.inf],
                                               labels=["4–12 earlier weeks", "13–52", "53–260", "261+"])
    for subgroup in ["known_run_age_band", "theatre"]:
        rows = []
        for (group, model), data in predictions.groupby([subgroup, "model"], observed=True):
            baseline = data[TARGET] - data.weekly_gross_lag1
            values = metric_values(data[TARGET], data.prediction_usd)
            rows.append({"subgroup": group, "model": model, "rows": len(data),
                "weeks": data.week_ending.nunique(), "small_subgroup_under_30": len(data) < 30,
                **values, "mae_reduction_vs_naive_pct": reduction(float(baseline.abs().mean()), values["mae_usd"])})
        pd.DataFrame(rows).to_csv(folder / f"{prefix}_subgroups_{subgroup}.csv", index=False)


def run_validation(path=FEATURES, tables=TABLES, models=MODELS):
    tables, models = Path(tables), Path(models)
    tables.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)
    if (tables / "final_evaluation_receipt.json").exists():
        raise RuntimeError("Real final evaluation already recorded; do not retune after seeing the holdout.")
    (models / "frozen_protocol.json").unlink(missing_ok=True)
    starting_code = code_fingerprints()
    starting_data = parquet_fingerprint(path)
    write_json(models / "protocol_before_validation.json", {"recorded_at_utc": utc_now(), **protocol()})
    # No final holdout outcomes, summaries or subgroup errors are evaluated here.
    dev = read_features(path)
    dev = dev.loc[dev.eligible & (dev.week_ending < pd.Timestamp(FINAL_START))].copy()
    if dev[PREDICTORS].isna().all().any():
        raise ValueError("An entire development predictor is missing; inspect input before training.")
    tuning, boundaries, selected_predictions = [], [], []
    print("Validation only; final holdout locked.", flush=True)
    with threadpool_limits(limits=2):
        for train_end, valid_year in FOLDS:
            train, valid = chronological_fold(dev, train_end, valid_year)
            boundaries.append({"train_through_year": train_end, "validation_year": valid_year,
                "train_first": train.week_ending.min(), "train_last": train.week_ending.max(),
                "validation_first": valid.week_ending.min(), "validation_last": valid.week_ending.max(),
                "train_rows": len(train), "validation_rows": len(valid),
                "train_weeks": train.week_ending.nunique(), "validation_weeks": valid.week_ending.nunique()})
            base = metric_values(valid[TARGET], valid.weekly_gross_lag1)
            tuning.append({"model": "naive", "parameters": "{}", "validation_year": valid_year,
                           "rows": len(valid), **base, "mae_reduction_vs_naive_pct": reduction(base["mae_usd"], base["mae_usd"])})
            selected_predictions.append(save_predictions(valid, "naive", valid.weekly_gross_lag1, valid_year))
            for model, parameters in GRID.items():
                for params in parameters:
                    estimator = build_pipeline(model, params)
                    estimator.fit(train[PREDICTORS], train[TARGET])
                    values = metric_values(valid[TARGET], estimator.predict(valid[PREDICTORS]))
                    tuning.append({"model": model, "parameters": json.dumps(params, sort_keys=True),
                        "validation_year": valid_year, "rows": len(valid), **values,
                        "mae_reduction_vs_naive_pct": reduction(base["mae_usd"], values["mae_usd"])})
            print(f"Completed validation year {valid_year}: {len(train):,} train / {len(valid):,} validation rows", flush=True)
        tuning = pd.DataFrame(tuning)
        tuning.to_csv(tables / "validation_tuning.csv", index=False)
        pd.DataFrame(boundaries).to_csv(tables / "validation_fold_boundaries.csv", index=False)
        summary = tuning.groupby(["model", "parameters"], as_index=False).agg(
            mean_validation_mae_usd=("mae_usd", "mean"), mean_validation_rmse_usd=("rmse_usd", "mean"),
            mean_validation_signed_error_usd=("mean_signed_error_usd", "mean"), folds=("validation_year", "size"))
        baseline_mae = float(summary.loc[summary.model == "naive", "mean_validation_mae_usd"].iloc[0])
        summary["mae_reduction_vs_naive_pct"] = summary.mean_validation_mae_usd.map(lambda v: reduction(baseline_mae, v))
        summary.sort_values(["mean_validation_mae_usd", "parameters"]).to_csv(tables / "validation_summary.csv", index=False)
        chosen = {}
        model_files = {}
        for model in GRID:
            row = summary.loc[summary.model == model].sort_values(["mean_validation_mae_usd", "parameters"]).iloc[0]
            params = json.loads(row.parameters)
            chosen[model] = params
            # Refit chosen settings on each historical fold for auditable OOF predictions.
            for train_end, valid_year in FOLDS:
                train, valid = chronological_fold(dev, train_end, valid_year)
                estimator = build_pipeline(model, params).fit(train[PREDICTORS], train[TARGET])
                selected_predictions.append(save_predictions(valid, model, estimator.predict(valid[PREDICTORS]), valid_year))
            # Frozen model sees all eligible pre-2018 rows, no final-period training.
            fitted = build_pipeline(model, params).fit(dev[PREDICTORS], dev[TARGET])
            file = models / f"{model}_pre2018.joblib"
            joblib.dump(fitted, file)
            model_files[file.name] = sha256(file)
        predictions = pd.concat(selected_predictions, ignore_index=True)
        predictions.to_csv(tables / "validation_predictions.csv", index=False)
        subgroup_tables(predictions, tables, "validation")
    best = summary.loc[summary.model != "naive"].sort_values(["mean_validation_mae_usd", "model", "parameters"]).iloc[0]
    decision = best.model if (best.mae_reduction_vs_naive_pct is not None and best.mae_reduction_vs_naive_pct >= KPI_MAE_REDUCTION_PCT) else "naive"
    if starting_code != code_fingerprints() or starting_data != parquet_fingerprint(path):
        raise RuntimeError("Source or Parquet changed during validation. Keep inputs fixed and rerun validation.")
    frozen = {"frozen_at_utc": utc_now(), "protocol": protocol(), "chosen_parameters": chosen,
        "validation_selected_method": decision, "code_sha256": starting_code,
        "parquet_sha256": starting_data, "model_sha256": model_files,
        "software": versions(), "pre2018_fit_rows": len(dev), "final_evaluation_run": False}
    write_json(models / "frozen_protocol.json", frozen)
    print(summary.to_string(index=False), flush=True)
    return frozen


def evaluate_frozen(frame, fitted_models):
    """Pure scoring helper; used with synthetic data in tests, real data only after unlock."""
    predictions = [save_predictions(frame, "naive", frame.weekly_gross_lag1)]
    for name, estimator in fitted_models.items():
        predictions.append(save_predictions(frame, name, estimator.predict(frame[PREDICTORS])))
    predictions = pd.concat(predictions, ignore_index=True)
    base_mae = metric_values(frame[TARGET], frame.weekly_gross_lag1)["mae_usd"]
    metrics = []
    for name, group in predictions.groupby("model"):
        values = metric_values(group[TARGET], group.prediction_usd)
        improvement = reduction(base_mae, values["mae_usd"])
        metrics.append({"model": name, "rows": len(group), "weeks": group.week_ending.nunique(), **values,
                        "mae_reduction_vs_naive_pct": improvement,
                        "meets_example_kpi": improvement >= KPI_MAE_REDUCTION_PCT if improvement is not None else None})
    return pd.DataFrame(metrics), predictions


def run_final(path=FEATURES, tables=TABLES, models=MODELS, *, unlocked=False):
    if not unlocked:
        raise PermissionError("Final holdout locked. Explicit --final-test is required after reviewing frozen_protocol.json.")
    tables, models = Path(tables), Path(models)
    receipt = tables / "final_evaluation_receipt.json"
    if receipt.exists():
        raise RuntimeError("Final evaluation already recorded. Reuse saved scores; do not retune/repeatedly score.")
    frozen = read_json(models / "frozen_protocol.json")
    # JSON-normalise tuples before equality comparison.
    if frozen["protocol"] != json.loads(json.dumps(protocol())) or frozen["code_sha256"] != code_fingerprints():
        raise RuntimeError("Code/protocol changed since freeze; rerun development validation before opening holdout.")
    if frozen["software"] != versions():
        raise RuntimeError("Dependency/runtime versions changed since freeze. Restore the recorded environment.")
    if frozen["parquet_sha256"] != parquet_fingerprint(path):
        raise RuntimeError("Parquet changed since freeze. Verify provenance and rerun development validation.")
    for name, digest in frozen["model_sha256"].items():
        if sha256(models / name) != digest:
            raise RuntimeError(f"Frozen fitted model modified: {name}")
    frame = read_features(path)
    final = frame.loc[frame.eligible & frame.week_ending.between(pd.Timestamp(FINAL_START), pd.Timestamp(FINAL_END))].copy()
    if final.empty:
        raise ValueError("Final period empty")
    fitted = {model: joblib.load(models / f"{model}_pre2018.joblib") for model in GRID}
    # Create once-only marker BEFORE scoring. A failed attempt remains visible for inspection.
    with receipt.open("x", encoding="utf-8") as stream:
        json.dump({"status": "started", "at_utc": utc_now(), "frozen_protocol_sha256": sha256(models / "frozen_protocol.json")}, stream, indent=2)
    with threadpool_limits(limits=2):
        metrics, predictions = evaluate_frozen(final, fitted)
    metrics.to_csv(tables / "final_metrics.csv", index=False)
    predictions.to_csv(tables / "final_predictions.csv", index=False)
    subgroup_tables(predictions, tables, "final")
    write_json(receipt, {"status": "completed", "at_utc": utc_now(), "rows_per_model": len(final),
        "frozen_protocol_sha256": sha256(models / "frozen_protocol.json"),
        "kpi_reduction_pct_recorded_before_evaluation": KPI_MAE_REDUCTION_PCT,
        "validation_selected_method": frozen["validation_selected_method"]})
    return metrics
