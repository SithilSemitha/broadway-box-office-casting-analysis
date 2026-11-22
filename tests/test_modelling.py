import json

import numpy as np
import pandas as pd
import pytest
import joblib

from broadway.config import GRID, PREDICTORS
from broadway.modelling import (build_pipeline, chronological_fold, evaluate_frozen,
                                metric_values, reduction, run_final)
from broadway.io import sha256, versions, write_json, read_json
from broadway.modelling import protocol, code_fingerprints


def synthetic():
    weeks = pd.date_range("2013-01-06", "2020-03-01", freq="W-SUN")
    n = len(weeks)
    frame = pd.DataFrame({"week_ending": weeks, "show": "Synthetic", "theatre": "Known venue",
        "inferred_run": 1, "run_weeks_known": np.arange(n) + 4,
        "weekly_gross": 1000 + np.arange(n) * 5, "weekly_gross_lag1": 995 + np.arange(n) * 5,
        "seats_sold_lag1": 100, "pct_capacity_lag1": 0.9, "performances_lag1": 8,
        "previews_lag1": 0, "gross_prior_4_mean": 987.5 + np.arange(n) * 5,
        "month": weeks.month.astype(str)})
    return frame


def test_chronological_folds_keep_every_week_together():
    frame = pd.concat([synthetic(), synthetic().assign(show="Other")], ignore_index=True)
    for train_end, validation_year in [(2014, 2015), (2015, 2016), (2016, 2017)]:
        train, valid = chronological_fold(frame, train_end, validation_year)
        assert train.week_ending.max() < valid.week_ending.min()
        assert valid.week_ending.dt.year.eq(validation_year).all()
        assert train.week_ending.dt.year.le(train_end).all()
        assert valid.groupby("week_ending").size().eq(2).all()


@pytest.mark.parametrize("name,params", [("ridge", {"alpha": 1}),
    ("hist_gradient_boosting", {"learning_rate": .1, "max_leaf_nodes": 15})])
def test_fold_local_transform_and_unseen_categories(name, params):
    frame = synthetic()
    train, valid = chronological_fold(frame, 2014, 2015)
    estimator = build_pipeline(name, params).fit(train[PREDICTORS], train.weekly_gross)
    transformer = estimator.named_steps["transform"]
    median = transformer.named_transformers_["numeric"].named_steps["impute"].statistics_[0]
    assert median == np.median(train.weekly_gross_lag1)
    unseen = valid.copy()
    unseen["theatre"] = "Unseen venue"
    unseen["month"] = "unseen month"
    unseen["seats_sold_lag1"] = np.nan
    assert np.isfinite(estimator.predict(unseen[PREDICTORS])).all()
    if name == "hist_gradient_boosting":
        assert estimator.named_steps["model"].early_stopping is False


def test_predictor_allowlist_excludes_realised_target_week_fields():
    forbidden = {"weekly_gross", "weekly_gross_overall", "seats_sold", "pct_capacity", "avg_ticket_price",
                 "potential_gross", "top_ticket_price", "performances", "previews", "eligible", "inferred_run", "year", "show"}
    assert not forbidden.intersection(PREDICTORS)
    assert len(PREDICTORS) == 9


def test_metrics_sign_reduction_and_zero_baseline():
    assert metric_values([10, 20], [12, 24]) == {"mae_usd": 3.0, "rmse_usd": np.sqrt(10), "mean_signed_error_usd": 3.0}
    assert reduction(100, 90) == 10
    assert reduction(0, 0) is None


def test_final_locked_without_explicit_unlock(tmp_path):
    with pytest.raises(PermissionError, match="locked"):
        run_final(path=tmp_path / "absent", tables=tmp_path, models=tmp_path)


def test_synthetic_final_scorer_same_rows_and_no_real_holdout():
    frame = synthetic()
    train = frame.loc[frame.week_ending < "2018-01-01"]
    final = frame.loc[frame.week_ending >= "2018-01-01"]
    fitted = {name: build_pipeline(name, GRID[name][0]).fit(train[PREDICTORS], train.weekly_gross) for name in GRID}
    metrics, predictions = evaluate_frozen(final, fitted)
    assert set(metrics.model) == {"naive", *GRID}
    assert metrics.rows.eq(len(final)).all()
    assert predictions.groupby("model").size().eq(len(final)).all()
    assert metrics.loc[metrics.model == "naive", "mae_usd"].iloc[0] == 5
    assert not predictions.week_ending.lt(pd.Timestamp("2018-01-01")).any()


def test_complete_final_artifact_path_with_synthetic_data_only(tmp_path, monkeypatch):
    import broadway.modelling as module
    frame = synthetic().assign(eligible=True)
    train = frame.loc[frame.week_ending < "2018-01-01"]
    models = tmp_path / "models"
    tables = tmp_path / "tables"
    models.mkdir()
    tables.mkdir()
    hashes = {}
    for name in GRID:
        estimator = build_pipeline(name, GRID[name][0]).fit(train[PREDICTORS], train.weekly_gross)
        file = models / f"{name}_pre2018.joblib"
        joblib.dump(estimator, file)
        hashes[file.name] = sha256(file)
    frozen = {"protocol": protocol(), "code_sha256": code_fingerprints(), "software": versions(),
              "parquet_sha256": "synthetic-fixture-only", "model_sha256": hashes,
              "validation_selected_method": "naive"}
    write_json(models / "frozen_protocol.json", frozen)
    monkeypatch.setattr(module, "read_features", lambda path: frame)
    monkeypatch.setattr(module, "parquet_fingerprint", lambda path: "synthetic-fixture-only")
    metrics = run_final(path=tmp_path / "synthetic", tables=tables, models=models, unlocked=True)
    assert len(metrics) == 3
    assert (tables / "final_predictions.csv").exists()
    assert (tables / "final_subgroups_theatre.csv").exists()
    assert read_json(tables / "final_evaluation_receipt.json")["status"] == "completed"
    with pytest.raises(RuntimeError, match="already recorded"):
        run_final(path=tmp_path / "synthetic", tables=tables, models=models, unlocked=True)
