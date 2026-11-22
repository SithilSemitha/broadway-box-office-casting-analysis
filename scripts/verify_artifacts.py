"""Verify saved development evidence, or saved final evidence with --final.

This script never trains a model or generates new forecasts.
"""
import argparse
import json

import numpy as np
import pandas as pd
from PIL import Image

from broadway.config import (AUDIT, FEATURES, FIGURES, FINAL_END, FINAL_START, FOLDS,
                             GRID, KEYS, KPI_MAE_REDUCTION_PCT, MODELS, RAW, TABLES, TARGET)
from broadway.io import parquet_fingerprint, read_json, sha256, utc_now, versions, write_json
from broadway.modelling import code_fingerprints, protocol


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, context):
    if expected is None or pd.isna(expected):
        require(pd.isna(actual), f"{context}: expected an undefined value")
    else:
        require(np.isclose(float(actual), float(expected), rtol=1e-9, atol=1e-7),
                f"{context}: saved {actual} differs from recomputed {expected}")


def check_predictions(frame, model_names, first_date, last_date):
    required = KEYS + [TARGET, "weekly_gross_lag1", "inferred_run", "run_weeks_known",
                       "model", "prediction_usd", "error_usd", "absolute_error_usd"]
    require(set(required) <= set(frame), "Prediction file is missing required audit columns")
    require(not frame.empty and not frame[required].isna().any().any(), "Predictions contain missing audit values")
    require(set(frame.model) == set(model_names), "Prediction model names do not match the frozen comparison")
    require(not frame.duplicated(KEYS + ["model"]).any(), "Duplicate model/forecast keys in predictions")
    require(frame.week_ending.between(pd.Timestamp(first_date), pd.Timestamp(last_date)).all(),
            "Prediction dates fall outside the declared evaluation period")
    numeric = frame[[TARGET, "weekly_gross_lag1", "prediction_usd", "error_usd", "absolute_error_usd"]]
    require(np.isfinite(numeric.to_numpy()).all(), "Nonfinite prediction/target/error values")
    error = frame.prediction_usd - frame[TARGET]
    require(np.allclose(frame.error_usd, error), "Saved signed errors do not match predictions minus actuals")
    require(np.allclose(frame.absolute_error_usd, np.abs(error)), "Saved absolute errors do not match")
    baseline = frame.loc[frame.model == "naive"].sort_values(KEYS).reset_index(drop=True)
    require(np.allclose(baseline.prediction_usd, baseline.weekly_gross_lag1),
            "Naive predictions do not equal the previous reported week's gross")
    baseline_mae = float(np.abs(baseline.prediction_usd - baseline[TARGET]).mean())
    scores = {}
    for model, group in frame.groupby("model"):
        group = group.sort_values(KEYS).reset_index(drop=True)
        require(group[KEYS].equals(baseline[KEYS]), f"Forecast keys differ from the benchmark for {model}")
        for field in [TARGET, "weekly_gross_lag1", "inferred_run", "run_weeks_known"]:
            require(np.allclose(group[field], baseline[field]), f"{field} differs across models for {model}")
        residual = group.prediction_usd - group[TARGET]
        mae = float(np.abs(residual).mean())
        scores[model] = {"rows": len(group), "weeks": group.week_ending.nunique(),
            "mae_usd": mae, "rmse_usd": float(np.sqrt(np.mean(residual ** 2))),
            "mean_signed_error_usd": float(residual.mean()),
            "mae_reduction_vs_naive_pct": 100 * (baseline_mae - mae) / baseline_mae if baseline_mae else None}
    return scores, baseline


def check_metric_table(table, expected_scores, context):
    require(set(table.model) == set(expected_scores) and not table.model.duplicated().any(),
            f"{context}: expected one metrics row per comparison method")
    for model, expected in expected_scores.items():
        row = table.loc[table.model == model].iloc[0]
        for field, value in expected.items():
            close(row[field], value, f"{context}, {model}, {field}")


def check_source_rows(baseline, source, context):
    """Detect mutually consistent prediction files that disagree with source data."""
    expected = source.sort_values(KEYS).reset_index(drop=True)
    actual = baseline.sort_values(KEYS).reset_index(drop=True)
    require(actual[KEYS].equals(expected[KEYS]), f"{context}: keys differ from eligible source rows")
    for column in [TARGET, "weekly_gross_lag1", "inferred_run", "run_weeks_known"]:
        require(np.allclose(actual[column], expected[column]), f"{context}: {column} differs from source")


def check_validation_summary(tuning, summary, frozen):
    """Reconcile all candidate summaries and the frozen selection with fold scores."""
    normalise = lambda value: json.dumps(json.loads(value), sort_keys=True)
    tuning, summary = tuning.copy(), summary.copy()
    tuning["parameters"] = tuning.parameters.map(normalise)
    summary["parameters"] = summary.parameters.map(normalise)
    candidates = {("naive", "{}")} | {(model, json.dumps(params, sort_keys=True))
        for model, choices in GRID.items() for params in choices}
    expected_grid = {(model, params, year) for model, params in candidates for _, year in FOLDS}
    require(not tuning.duplicated(["model", "parameters", "validation_year"]).any(), "Duplicate tuning rows")
    require(set(tuning[["model", "parameters", "validation_year"]].itertuples(index=False, name=None)) == expected_grid,
            "Tuning candidate/fold keys differ from the configured grid")
    require(not summary.duplicated(["model", "parameters"]).any(), "Duplicate summary rows")
    require(set(summary[["model", "parameters"]].itertuples(index=False, name=None)) == candidates,
            "Summary candidate keys differ from the configured grid")
    fields = {"mae_usd": "mean_validation_mae_usd", "rmse_usd": "mean_validation_rmse_usd",
              "mean_signed_error_usd": "mean_validation_signed_error_usd"}
    require(np.isfinite(tuning[list(fields)].to_numpy()).all(), "Nonfinite tuning metrics")
    baseline_mae = tuning.loc[tuning.model == "naive", "mae_usd"].mean()
    for row in summary.itertuples():
        group = tuning.loc[(tuning.model == row.model) & (tuning.parameters == row.parameters)]
        close(row.folds, len(FOLDS), "Summary fold count")
        for field, output in fields.items():
            close(getattr(row, output), group[field].mean(), f"Summary/{row.model}/{output}")
        reduction = 100 * (baseline_mae - group.mae_usd.mean()) / baseline_mae if baseline_mae else None
        close(row.mae_reduction_vs_naive_pct, reduction, "Summary baseline reduction")
    for model in GRID:
        best = summary.loc[summary.model == model].sort_values(["mean_validation_mae_usd", "parameters"]).iloc[0]
        require(json.loads(best.parameters) == frozen["chosen_parameters"][model], "Frozen parameters differ from minimum validation MAE")
    best = summary.loc[summary.model != "naive"].sort_values(["mean_validation_mae_usd", "model", "parameters"]).iloc[0]
    choice = best.model if best.mae_reduction_vs_naive_pct >= KPI_MAE_REDUCTION_PCT else "naive"
    require(frozen["validation_selected_method"] == choice, "Frozen decision differs from KPI rule")


def check_subgroups(predictions, folder, prefix):
    predictions = predictions.copy()
    predictions["known_run_age_band"] = pd.cut(predictions.run_weeks_known,
        [3, 12, 52, 260, np.inf], labels=["4–12 earlier weeks", "13–52", "53–260", "261+"])
    for field in ["known_run_age_band", "theatre"]:
        saved = pd.read_csv(folder / f"{prefix}_subgroups_{field}.csv")
        groups = list(predictions.groupby([field, "model"], observed=True))
        require(len(saved) == len(groups), f"{prefix}: missing or extra {field} subgroup rows")
        require(not saved.duplicated(["subgroup", "model"]).any(), f"{prefix}: duplicate subgroup keys")
        for (name, model), group in groups:
            match = saved.loc[(saved.subgroup.astype(str) == str(name)) & (saved.model == model)]
            require(len(match) == 1, f"{prefix}: missing {field}/{name}/{model}")
            row = match.iloc[0]
            error = group.prediction_usd - group[TARGET]
            baseline_mae = float(np.abs(group.weekly_gross_lag1 - group[TARGET]).mean())
            mae = float(np.abs(error).mean())
            expected = {"rows": len(group), "weeks": group.week_ending.nunique(), "mae_usd": mae,
                "rmse_usd": float(np.sqrt(np.mean(error ** 2))), "mean_signed_error_usd": float(error.mean()),
                "mae_reduction_vs_naive_pct": 100 * (baseline_mae - mae) / baseline_mae if baseline_mae else None}
            for metric, value in expected.items():
                close(row[metric], value, f"{prefix}/{field}/{name}/{model}/{metric}")
            require(bool(row.small_subgroup_under_30) == (len(group) < 30), "Small-subgroup flag differs")


def check_final_tables(folder, expected_frame, model_names, kpi=KPI_MAE_REDUCTION_PCT):
    """Pure saved-evidence checks, also exercised with synthetic fixtures."""
    predictions = pd.read_csv(folder / "final_predictions.csv", parse_dates=["week_ending"])
    metrics = pd.read_csv(folder / "final_metrics.csv")
    scores, baseline = check_predictions(predictions, model_names, FINAL_START, FINAL_END)
    expected = expected_frame.sort_values(KEYS).reset_index(drop=True)
    require(baseline[KEYS].equals(expected[KEYS]), "Final predictions do not cover exactly the eligible source keys")
    for field in [TARGET, "weekly_gross_lag1", "inferred_run", "run_weeks_known"]:
        require(np.allclose(baseline[field], expected[field]), f"Final {field} differs from the engineered source")
    check_metric_table(metrics, scores, "Final metrics")
    for model, values in scores.items():
        improvement = values["mae_reduction_vs_naive_pct"]
        flag = metrics.loc[metrics.model == model, "meets_example_kpi"].iloc[0]
        if improvement is None:
            require(pd.isna(flag), "KPI must be undefined when baseline MAE is zero")
        else:
            require(bool(flag) == (improvement >= kpi), "Saved KPI flag does not match the predeclared threshold")
    check_subgroups(predictions, folder, "final")
    return len(baseline), baseline.week_ending.nunique()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final", action="store_true", help="Verify already saved, explicitly unlocked final evidence")
    args = parser.parse_args(argv)
    receipt_path = TABLES / "final_evaluation_receipt.json"
    if args.final:
        require(receipt_path.exists(), "No completed final evaluation exists. This verifier cannot unlock or run scoring.")
        receipt = read_json(receipt_path)
        require(receipt.get("status") == "completed", "Final evaluation did not complete; inspect its receipt and error")
    else:
        require(not list(TABLES.glob("final_*")),
                "Final artifacts exist. Use --final to verify them without running scoring again.")

    manifest = read_json(AUDIT / "input_manifest.json")
    require(sha256(RAW) == manifest["sha256"] and RAW.stat().st_size == manifest["bytes"],
            "Raw data bytes differ from the acquisition manifest")
    crosschecks = pd.read_csv(AUDIT / "snapshot_crosschecks.csv")
    require(crosschecks.matches.all(), "Snapshot differs from brief; inspect and explain before proceeding")
    require(read_json(AUDIT / "parquet_roundtrip.json")["row_multisets_match"], "Parquet roundtrip failed")
    scope = read_json(TABLES / "eda_scope.json")
    require(scope["max_target_date"] <= "2014-12-31" and scope["figures"] == 6, "EDA scope/figure count differs")
    index = pd.read_csv(TABLES / "figure_index.csv")
    require(len(index) == 6 and index.file.nunique() == 6, "Expected six distinct figure artifacts")
    for filename in index.file:
        with Image.open(FIGURES / filename) as image:
            image.verify()
    folds = pd.read_csv(TABLES / "validation_fold_boundaries.csv")
    require(len(folds) == len(FOLDS), "Fold count differs from protocol")
    require((pd.to_datetime(folds.train_last) < pd.to_datetime(folds.validation_first)).all(), "Overlapping fold dates")
    require(list(zip(folds.train_through_year, folds.validation_year)) == FOLDS, "Fold years differ from protocol")
    tuning = pd.read_csv(TABLES / "validation_tuning.csv")
    require(len(tuning) == len(FOLDS) * (1 + sum(map(len, GRID.values()))), "Tuning grid is incomplete")
    require(set(tuning.validation_year) == {year for _, year in FOLDS}, "Tuning validation years differ")
    predictions = pd.read_csv(TABLES / "validation_predictions.csv", parse_dates=["week_ending"])
    models = {"naive", *GRID}
    _, baseline = check_predictions(predictions, models, "2015-01-01", "2017-12-31")
    require((predictions.validation_year == predictions.week_ending.dt.year).all(), "Validation year/date mismatch")
    frozen = read_json(MODELS / "frozen_protocol.json")
    require(frozen["protocol"] == json.loads(json.dumps(protocol())), "Frozen protocol differs from configuration")
    check_validation_summary(tuning, pd.read_csv(TABLES / "validation_summary.csv"), frozen)
    # Read only the development partition for actuals/keys; do not summarise the holdout.
    development = pd.read_parquet(FEATURES, engine="pyarrow", filters=[("year", "<", 2018)])
    development["week_ending"] = pd.to_datetime(development.week_ending)
    development = development.loc[development.eligible]
    check_source_rows(baseline, development.loc[development.year.isin([year for _, year in FOLDS])], "Validation")
    for row in folds.itertuples():
        train = development.loc[development.year <= row.train_through_year]
        valid = development.loc[development.year == row.validation_year]
        for field, actual in [("train_rows", len(train)), ("validation_rows", len(valid)),
                              ("train_weeks", train.week_ending.nunique()), ("validation_weeks", valid.week_ending.nunique())]:
            close(getattr(row, field), actual, f"Fold {row.validation_year}/{field}")
        for prefix, frame in [("train", train), ("validation", valid)]:
            require(pd.Timestamp(getattr(row, prefix + "_first")) == frame.week_ending.min()
                    and pd.Timestamp(getattr(row, prefix + "_last")) == frame.week_ending.max(), "Fold dates differ from source")
    close(frozen["pre2018_fit_rows"], len(development), "Frozen refit row count")
    for (model, year), group in predictions.groupby(["model", "validation_year"]):
        parameters = json.dumps(frozen["chosen_parameters"][model], sort_keys=True) if model != "naive" else "{}"
        match = tuning.loc[(tuning.model == model) & (tuning.validation_year == year) & (tuning.parameters == parameters)]
        require(len(match) == 1, "Selected validation settings do not match exactly one tuning row")
        row = match.iloc[0]
        errors = group.prediction_usd - group[TARGET]
        for field, value in {"mae_usd": np.abs(errors).mean(), "rmse_usd": np.sqrt(np.mean(errors ** 2)),
                             "mean_signed_error_usd": errors.mean(), "rows": len(group)}.items():
            close(row[field], value, f"Validation/{model}/{year}/{field}")
    check_subgroups(predictions, TABLES, "validation")
    require(frozen["code_sha256"] == code_fingerprints(), "Frozen code hashes differ")
    require(frozen["software"] == versions(), "Frozen runtime/package versions differ")
    require(frozen["parquet_sha256"] == parquet_fingerprint(FEATURES), "Frozen Parquet fingerprint differs")
    for filename, checksum in frozen["model_sha256"].items():
        require(sha256(MODELS / filename) == checksum, f"Frozen fitted model differs: {filename}")
    result = {"verified_at_utc": utc_now(), "snapshot_checks_passed": len(crosschecks),
        "raw_rows": manifest["raw_rows"], "eligible_rows": manifest["eligible_rows"],
        "figures_verified": len(index), "tuning_rows": len(tuning), "chronological_folds": len(folds),
        "validation_predictions_per_model": len(baseline), "selected_predictions_match_tuning_scores": True,
        "subgroup_counts_and_scores_verified": True, "frozen_hashes_match": True,
        "raw_bytes_match_manifest": True, "validation_keys_actuals_match_source": True,
        "summary_grid_selection_and_fold_counts_verified": True,
        "real_final_holdout_locked": not args.final, "real_final_holdout_evaluated": args.final}
    if args.final:
        require(receipt["frozen_protocol_sha256"] == sha256(MODELS / "frozen_protocol.json"), "Receipt/protocol hash mismatch")
        source = pd.read_parquet(FEATURES, engine="pyarrow")
        source["week_ending"] = pd.to_datetime(source.week_ending)
        eligible_final = source.loc[source.eligible & source.week_ending.between(pd.Timestamp(FINAL_START), pd.Timestamp(FINAL_END))]
        rows, weeks = check_final_tables(TABLES, eligible_final, models, frozen["protocol"]["kpi_reduction_pct"])
        require(rows == receipt["rows_per_model"], "Final receipt row count differs")
        require(receipt["validation_selected_method"] == frozen["validation_selected_method"], "Final decision rule differs")
        close(receipt["kpi_reduction_pct_recorded_before_evaluation"], frozen["protocol"]["kpi_reduction_pct"], "Receipt KPI")
        result.update(final_predictions_per_model=rows, final_weeks=weeks,
                      final_keys_actuals_metrics_and_subgroups_match=True)
    destination = AUDIT / ("final_verification.json" if args.final else "development_verification.json")
    write_json(destination, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
