import csv
from datetime import date, timedelta

import pytest
from pyspark.sql import functions as F

from broadway.config import PREDICTORS
from broadway.spark_pipeline import (SCHEMA, audit_and_clean, engineer_features,
                                     read_raw, verify_csv_structure)


def fixture_csv(tmp_path, rows, name="fixture.csv"):
    path = tmp_path / name
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(SCHEMA.fieldNames())
        writer.writerows(rows)
    return path


def observation(day, gross, show="Show A", theatre="Theatre A"):
    return [day.isoformat(), 1, 100000.0, show, theatre, gross, "NA", 50.0,
            "NA", 2000, 1000, 1.02, 8, 0]


def frame_from_rows(spark, tmp_path, rows, name="fixture.csv"):
    return read_raw(spark, fixture_csv(tmp_path, rows, name), tmp_path / (name + "_audit"))


def test_gap_resets_lags_and_full_four_week_history(spark, tmp_path):
    start = date(2014, 1, 5)
    offsets = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]
    raw = frame_from_rows(spark, tmp_path, [observation(start + timedelta(weeks=i), 100.0 + i) for i in offsets])
    output = engineer_features(raw).orderBy("week_ending").collect()
    assert [r.eligible for r in output] == [False] * 4 + [True] + [False] * 4 + [True]
    assert output[5].weekly_gross_lag1 is None
    assert output[5].run_weeks_known == 0
    assert output[5].inferred_run != output[4].inferred_run
    assert output[4].gross_prior_4_mean == 101.5
    assert output[9].gross_prior_4_mean == 107.5
    raw.unpersist()


def test_current_and_future_target_changes_do_not_change_own_predictors(spark, tmp_path):
    start = date(2014, 1, 5)
    raw = frame_from_rows(spark, tmp_path, [observation(start + timedelta(weeks=i), 100.0 + i) for i in range(7)])
    target = start + timedelta(weeks=4)
    original = engineer_features(raw).filter(F.col("week_ending") == F.lit(target)).select(PREDICTORS).first()
    changed = raw.withColumn("weekly_gross", F.when(F.col("week_ending") >= F.lit(target), 9999999.0).otherwise(F.col("weekly_gross")))
    recomputed = engineer_features(changed).filter(F.col("week_ending") == F.lit(target)).select(PREDICTORS).first()
    assert original == recomputed
    assert original.weekly_gross_lag1 == 103.0 and original.gross_prior_4_mean == 101.5
    raw.unpersist()


def test_windows_never_mix_shows_or_venues(spark, tmp_path):
    start = date(2014, 1, 5)
    rows = [observation(start + timedelta(weeks=i), base + i, show, venue)
            for show, venue, base in [("A", "V1", 10), ("A", "V2", 100), ("B", "V1", 1000)] for i in range(5)]
    raw = frame_from_rows(spark, tmp_path, rows)
    result = engineer_features(raw).filter("eligible").orderBy("show", "theatre").collect()
    assert [r.weekly_gross_lag1 for r in result] == [13, 103, 1003]
    assert [r.gross_prior_4_mean for r in result] == [11.5, 101.5, 1001.5]
    raw.unpersist()


def test_exact_duplicates_removed_conflicting_keys_stop(spark, tmp_path):
    row = observation(date(2014, 1, 5), 100)
    raw = frame_from_rows(spark, tmp_path, [row, row])
    clean, stages = audit_and_clean(raw, tmp_path / "quality")
    assert clean.count() == 1 and stages[1]["rows"] == 1
    clean.unpersist()
    raw.unpersist()
    different = row.copy()
    different[5] = 200
    conflicting = frame_from_rows(spark, tmp_path, [row, different], "conflict.csv")
    with pytest.raises(ValueError, match="Conflicting business keys"):
        audit_and_clean(conflicting, tmp_path / "conflicts")
    conflicting.unpersist()


@pytest.mark.parametrize("column,value", [(5, "bad-number"), (0, "2014-02-30"), (1, "1.5")])
def test_malformed_typed_values_stop(spark, tmp_path, column, value):
    row = observation(date(2014, 1, 5), 100)
    row[column] = value
    with pytest.raises(ValueError, match="parsing"):
        frame_from_rows(spark, tmp_path, [row])


def test_header_order_and_width_checked(tmp_path):
    row = observation(date(2014, 1, 5), 100)
    path = fixture_csv(tmp_path, [row[:-1]])
    count, bad = verify_csv_structure(path)
    assert count == 1 and len(bad) == 1
    path.write_text("wrong,header\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Header/order"):
        verify_csv_structure(path)


def test_zero_gross_overcapacity_retained_invalid_target_excluded(spark, tmp_path):
    start = date(2014, 1, 5)
    rows = [observation(start + timedelta(weeks=i), gross) for i, gross in enumerate([0, -1, "NA", 200])]
    raw = frame_from_rows(spark, tmp_path, rows)
    clean, stages = audit_and_clean(raw, tmp_path / "quality")
    assert clean.count() == 2 and clean.filter("weekly_gross = 0").count() == 1
    assert clean.filter("pct_capacity > 1").count() == 2
    clean.unpersist()
    raw.unpersist()


def test_parquet_preserves_engineered_rows_and_types(spark, tmp_path):
    start = date(2014, 1, 5)
    raw = frame_from_rows(spark, tmp_path, [observation(start + timedelta(weeks=i), 100 + i) for i in range(5)])
    output = engineer_features(raw)
    destination = (tmp_path / "features.parquet").as_posix()
    output.write.parquet(destination)
    restored = spark.read.parquet(destination)
    assert [(f.name, f.dataType) for f in output.schema] == [(f.name, f.dataType) for f in restored.schema]
    assert output.exceptAll(restored).count() == restored.exceptAll(output).count() == 0
    raw.unpersist()
