"""Native Spark CSV -> quality audit -> point-in-time windows -> Parquet.

No pandas ingestion, Python UDFs, or alternate feature-engineering path.
"""
import csv
import os
import re
import shutil
import sys
import urllib.request
from datetime import date, timedelta
from pathlib import Path

import pyspark
from pyspark.sql import SparkSession, Window, functions as F, types as T

from .config import AUDIT, DOC_URL, FEATURES, KEYS, RAW, ROOT, SOURCE_URL
from .io import read_json, sha256, utc_now, versions, write_csv, write_json

SCHEMA = T.StructType([
    T.StructField(name, kind(), True) for name, kind in [
        ("week_ending", T.DateType), ("week_number", T.IntegerType),
        ("weekly_gross_overall", T.DoubleType), ("show", T.StringType),
        ("theatre", T.StringType), ("weekly_gross", T.DoubleType),
        ("potential_gross", T.DoubleType), ("avg_ticket_price", T.DoubleType),
        ("top_ticket_price", T.DoubleType), ("seats_sold", T.IntegerType),
        ("seats_in_theatre", T.IntegerType), ("pct_capacity", T.DoubleType),
        ("performances", T.IntegerType), ("previews", T.IntegerType),
    ]
])
DOUBLE_COLS = [f.name for f in SCHEMA if isinstance(f.dataType, T.DoubleType)]


def start_spark():
    os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
    os.environ.setdefault("SPARK_HOME", str(Path(pyspark.__file__).parent))
    if os.name == "nt":
        # Spark's .cmd discovery script does not quote PYSPARK_PYTHON. Resolve
        # python.exe through this environment's Scripts directory instead.
        os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]
        if os.environ.get("PYSPARK_PYTHON", sys.executable) == sys.executable:
            os.environ["PYSPARK_PYTHON"] = "python.exe"
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    if os.name == "nt" and "JAVA_HOME" not in os.environ:
        candidate = Path("C:/Program Files/Java/jdk-17")
        if (candidate / "bin/java.exe").exists():
            os.environ["JAVA_HOME"] = str(candidate)
    native_home = ROOT / ".tools/hadoop"
    if os.name == "nt" and (native_home / "bin/hadoop.dll").exists() and "HADOOP_HOME" not in os.environ:
        os.environ["HADOOP_HOME"] = str(native_home)
        os.environ["PATH"] = str(native_home / "bin") + os.pathsep + os.environ["PATH"]
    spark = (SparkSession.builder.master("local[2]").appName("BroadwayCW")
             .config("spark.ui.enabled", "false")
             .config("spark.sql.shuffle.partitions", "4")
             .config("spark.sql.session.timeZone", "UTC")
             .config("spark.sql.legacy.timeParserPolicy", "CORRECTED")
             .config("spark.sql.csv.parser.columnPruning.enabled", "false")
             .config("spark.sql.adaptive.enabled", "false")
             .getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def smoke(spark, output):
    """Exercise JVM, SQL and actual filesystem Parquet write/read (not just startup)."""
    output = Path(output).resolve()
    frame = spark.range(5).withColumn("value", F.col("id") * 2)
    frame.write.mode("overwrite").parquet(output.as_posix())
    check = spark.read.parquet(output.as_posix())
    assert check.count() == 5 and check.agg(F.sum("value")).first()[0] == 20
    return {"spark": spark.version, "java": spark.sparkContext._jvm.java.lang.System.getProperty("java.version"),
            "hadoop": spark.sparkContext._jvm.org.apache.hadoop.util.VersionInfo.getVersion(),
            "hadoop_home": os.environ.get("HADOOP_HOME"),
            "parquet_rows": 5, "value_sum": 20, "verified_at_utc": utc_now(), **versions()}


def acquire(path=RAW, local_csv=None):
    """Preserve existing bytes; never overwrite a different existing raw file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path = path.with_suffix(".source.json")
    if local_csv:
        source = Path(local_csv).resolve()
        if path.exists() and sha256(path) != sha256(source):
            raise FileExistsError(f"Raw file differs: preserve/move {path} before importing another snapshot.")
        if source != path.resolve():
            shutil.copyfile(source, path)
        method = "local CSV supplied by user; upstream access not independently established"
    elif path.exists():
        return path
    else:
        temporary = path.with_suffix(".download")
        try:
            with urllib.request.urlopen(SOURCE_URL, timeout=90) as response, temporary.open("wb") as dest:
                shutil.copyfileobj(response, dest)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        method = "downloaded from source URL"
    write_json(receipt_path, {"source_url": SOURCE_URL, "method": method,
                             "acquired_at_utc": utc_now(), "sha256": sha256(path)})
    return path


def verify_csv_structure(path):
    """Only lexical CSV checks in stdlib; typed ingestion remains Spark's job."""
    bad = []
    n = 0
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.reader(stream, strict=True)
        header = next(reader)
        if header != SCHEMA.fieldNames():
            raise ValueError(f"Header/order differs. Expected {SCHEMA.fieldNames()}, received {header}")
        for n, row in enumerate(reader, 1):
            if len(row) != len(header) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}|NA|", row[0]):
                bad.append({"record": n, "field_count": len(row), "date_token": row[0] if row else ""})
    return n, bad


def read_raw(spark, path, audit_dir=AUDIT):
    audit_dir = Path(audit_dir)
    count, bad = verify_csv_structure(path)
    write_csv(audit_dir / "malformed_csv_structure.csv", bad, ["record", "field_count", "date_token"])
    if bad:
        raise ValueError("Malformed CSV structure/date tokens; inspect malformed_csv_structure.csv")
    schema = T.StructType(SCHEMA.fields + [T.StructField("_corrupt_record", T.StringType(), True)])
    raw = (spark.read.schema(schema).option("header", True).option("dateFormat", "yyyy-MM-dd")
           .option("enableDateTimeParsingFallback", "false")
           .option("nullValue", "NA").option("mode", "PERMISSIVE")
           .option("columnNameOfCorruptRecord", "_corrupt_record")
           .csv(Path(path).resolve().as_posix()).cache())
    actual = raw.count()
    corrupt = raw.filter(F.col("_corrupt_record").isNotNull())
    corrupt_count = corrupt.count()
    examples = [r.asDict() for r in corrupt.limit(20).collect()]
    write_json(audit_dir / "parse_audit.json", {"csv_records": count, "spark_rows": actual, "malformed_rows": corrupt_count,
                                               "malformed_examples": examples})
    if actual != count or corrupt_count or not actual:
        raw.unpersist()
        raise ValueError("Spark parsing failed or row count differs. Inspect parse_audit.json")
    # Cache the projected frame before freeing the parser cache.
    typed = raw.drop("_corrupt_record").cache()
    typed.count()
    raw.unpersist()
    return typed


def missing_table(frame):
    expressions = []
    for field in SCHEMA:
        col = F.col(field.name)
        expressions.append(F.sum(col.isNull().cast("long")).alias(field.name + "__null"))
        if field.name in DOUBLE_COLS:
            expressions.extend([
                F.sum(F.isnan(col).cast("long")).alias(field.name + "__nan"),
                F.sum((F.abs(col) == float("inf")).cast("long")).alias(field.name + "__inf"),
            ])
    values = frame.agg(*expressions).first().asDict()
    total = frame.count()
    return [{"field": f.name, "null_count": values[f.name + "__null"],
             "nan_count": values.get(f.name + "__nan", 0),
             "infinite_count": values.get(f.name + "__inf", 0),
             "missing_count": values[f.name + "__null"] + values.get(f.name + "__nan", 0),
             "missing_rate": (values[f.name + "__null"] + values.get(f.name + "__nan", 0)) / total}
            for f in SCHEMA]


def audit_and_clean(raw, audit_dir=AUDIT):
    audit_dir = Path(audit_dir)
    raw_count = raw.count()
    write_csv(audit_dir / "missingness.csv", missing_table(raw))
    unique = raw.dropDuplicates(SCHEMA.fieldNames()).cache()
    unique_count = unique.count()
    conflicts = unique.groupBy(KEYS).count().filter(F.col("count") > 1)
    conflict_count = conflicts.count()
    write_json(audit_dir / "duplicates.json", {"raw_rows": raw_count,
               "exact_duplicates_removed": raw_count - unique_count,
               "conflicting_keys": conflict_count,
               "conflicting_key_examples": [r.asDict() for r in conflicts.limit(20).collect()]})
    if conflict_count:
        unique.unpersist()
        raise ValueError("Conflicting business keys: inspect duplicates.json before continuing.")
    invalid_key = (F.col("week_ending").isNull() | F.col("show").isNull() | F.col("theatre").isNull()
                   | (F.length(F.trim("show")) == 0) | (F.length(F.trim("theatre")) == 0))
    invalid_target = (F.col("weekly_gross").isNull() | F.isnan("weekly_gross")
                      | (F.abs(F.col("weekly_gross")) == float("inf")) | (F.col("weekly_gross") < 0))
    reason = (F.when(invalid_key, "missing_key")
              .when(F.dayofweek("week_ending") != 1, "non_sunday")
              .when(invalid_target, "invalid_target"))
    tagged = unique.withColumn("exclusion_reason", reason)
    exclusions = tagged.filter(F.col("exclusion_reason").isNotNull())
    write_csv(audit_dir / "excluded_rows.csv", [r.asDict() for r in exclusions.collect()],
              SCHEMA.fieldNames() + ["exclusion_reason"])
    # Keep zeros and all >100% occupancy. Optional nonfinite measurements become null, never zero.
    clean = tagged.filter(F.col("exclusion_reason").isNull()).drop("exclusion_reason")
    for col in DOUBLE_COLS:
        if col != "weekly_gross":
            clean = clean.withColumn(col, F.when(F.isnan(col) | (F.abs(F.col(col)) == float("inf")),
                                               F.lit(None).cast("double")).otherwise(F.col(col)))
    clean = clean.cache()
    clean_count = clean.count()
    # Review threshold is derived ONLY from the development EDA period.
    early = clean.filter(F.year("week_ending") <= 2014)
    quantiles = early.approxQuantile("weekly_gross", [0.25, 0.75], 0.001)
    if not quantiles:
        raise ValueError("No development-period data for the anomaly review threshold.")
    q1, q3 = quantiles
    threshold = q3 + 3 * (q3 - q1)
    flags = {
        "zero_reported_gross": F.col("weekly_gross") == 0,
        "zero_theatre_capacity": F.col("seats_in_theatre") == 0,
        "occupancy_above_100_pct": F.col("pct_capacity") > 1,
        "high_gross_development_3iqr": F.col("weekly_gross") > threshold,
        "negative_reported_non_target": sum(
            [(F.col(c) < 0).cast("int") for c in ["seats_sold", "seats_in_theatre", "pct_capacity",
                                                  "performances", "previews", "avg_ticket_price"]]) > 0,
    }
    summaries, examples = [], []
    for name, expression in flags.items():
        subset = clean.filter(expression)
        summaries.append({"flag": name, "count": subset.count(), "action": "retained for inspection"})
        # Examples are restricted to the exploratory period to avoid inspecting final outcomes.
        for row in subset.filter(F.year("week_ending") <= 2014).orderBy(*KEYS).limit(12).collect():
            examples.append({"flag": name, **row.asDict()})
    write_csv(audit_dir / "anomaly_counts.csv", summaries)
    write_csv(audit_dir / "anomaly_examples_through_2014.csv", examples, ["flag"] + SCHEMA.fieldNames())
    write_json(audit_dir / "anomaly_policy.json", {"gross_review_threshold_nominal_usd": threshold,
        "threshold_source": "Q3 + 3 IQR on cleaned target weeks through 2014; approximate quantiles error=0.001",
        "zero_and_overcapacity_policy": "Retain; >1 occupancy can represent standing-room sales.",
        "optional_nonfinite_policy": "NaN/infinity converted to null after raw missingness audit.",
        "examples_scope": "through 2014 only; aggregate quality counts include all dates"})
    stages = [{"stage": "raw", "rows": raw_count}, {"stage": "exact_unique", "rows": unique_count},
              {"stage": "excluded_invalid_key_date_or_target", "rows": unique_count - clean_count},
              {"stage": "cleaned", "rows": clean_count}]
    unique.unpersist()
    return clean, stages


def engineer_features(clean):
    pair = Window.partitionBy("show", "theatre").orderBy("week_ending")
    previous = F.lag("week_ending").over(pair)
    frame = clean.withColumn("_run_start", F.when(previous.isNull() | (F.datediff("week_ending", previous) != 7), 1).otherwise(0))
    frame = frame.withColumn("inferred_run", F.sum("_run_start").over(pair.rowsBetween(Window.unboundedPreceding, 0)))
    run = Window.partitionBy("show", "theatre", "inferred_run").orderBy("week_ending")
    for col in ["weekly_gross", "seats_sold", "pct_capacity", "performances", "previews"]:
        frame = frame.withColumn(col + "_lag1", F.lag(col).over(run))
    history = run.rowsBetween(-4, -1)
    frame = (frame.withColumn("gross_prior_4_mean", F.avg("weekly_gross").over(history))
             .withColumn("prior_gross_count", F.count("weekly_gross").over(history))
             .withColumn("run_weeks_known", F.row_number().over(run) - 1)
             .withColumn("month", F.month("week_ending").cast("string"))
             .withColumn("year", F.year("week_ending")))
    return frame.withColumn("eligible", (F.col("run_weeks_known") >= 4) & (F.col("prior_gross_count") == 4)).drop("_run_start")


def leakage_example(frame, audit_dir):
    candidate = (frame.filter(F.col("eligible") & (F.col("year") <= 2014))
                 .orderBy("week_ending", "show", "theatre").first())
    if candidate is None:
        raise ValueError("No eligible development example for leakage check.")
    r = candidate.asDict()
    history = (frame.filter((F.col("show") == r["show"]) & (F.col("theatre") == r["theatre"])
                  & (F.col("inferred_run") == r["inferred_run"])
                  & (F.col("week_ending") >= F.lit(r["week_ending"] - timedelta(days=28)))
                  & (F.col("week_ending") < F.lit(r["week_ending"])))
               .orderBy("week_ending").select("week_ending", "weekly_gross").collect())
    assert len(history) == 4
    mean = sum(h.weekly_gross for h in history) / 4
    assert abs(mean - r["gross_prior_4_mean"]) < 1e-7
    write_json(Path(audit_dir) / "leakage_example.json", {
        "show": r["show"], "theatre": r["theatre"], "target_date": r["week_ending"],
        "inferred_run": r["inferred_run"], "prior_dates_and_grosses": [h.asDict() for h in history],
        "computed_four_week_mean": mean, "model_feature_gross_prior_4_mean": r["gross_prior_4_mean"],
        "forecast_origin": r["week_ending"] - timedelta(days=7),
        "availability_assumption": "Previous reported week available before prediction; reporting timestamps absent.",
    })


def preprocess(spark, raw_path=RAW, output=FEATURES, audit_dir=AUDIT):
    audit_dir = Path(audit_dir)
    audit_dir.mkdir(parents=True, exist_ok=True)
    raw = read_raw(spark, raw_path, audit_dir)
    clean, stages = audit_and_clean(raw, audit_dir)
    frame = engineer_features(clean).cache()
    n = frame.count()
    eligible = frame.filter("eligible").count()
    final = frame.filter("eligible AND week_ending >= DATE '2018-01-01' AND week_ending <= DATE '2020-03-01'")
    stages += [{"stage": "engineered_all", "rows": n}, {"stage": "eligible", "rows": eligible},
               {"stage": "ineligible_retained", "rows": n - eligible}]
    assert stages[3]["rows"] == n == eligible + (n - eligible)
    write_csv(audit_dir / "row_counts.csv", stages)
    leakage_example(frame, audit_dir)
    output = Path(output).resolve()
    frame.orderBy(*KEYS).coalesce(1).write.mode("overwrite").parquet(output.as_posix())
    restored = spark.read.parquet(output.as_posix())
    schema_before = [(f.name, f.dataType.simpleString()) for f in frame.schema]
    schema_after = [(f.name, f.dataType.simpleString()) for f in restored.schema]
    assert schema_before == schema_after and restored.count() == n
    # Compare complete row multisets, not only counts.
    assert frame.exceptAll(restored).limit(1).count() == 0
    assert restored.exceptAll(frame).limit(1).count() == 0
    write_json(audit_dir / "parquet_roundtrip.json", {"rows": n, "schema": schema_after,
                "count_matches": True, "types_match": True, "row_multisets_match": True,
                "nullable_note": "Parquet may mark non-null computed fields nullable on read."})
    bounds = raw.agg(F.min("week_ending"), F.max("week_ending")).first()
    receipt_path = Path(raw_path).with_suffix(".source.json")
    receipt = read_json(receipt_path) if receipt_path.exists() else {"method": "existing local file; upstream acquisition date unknown"}
    with (audit_dir / "missingness.csv").open(newline="", encoding="utf-8") as stream:
        missing = {r["field"]: int(r["missing_count"]) for r in csv.DictReader(stream)}
    duplicates = read_json(audit_dir / "duplicates.json")
    manifest = {"source_url": SOURCE_URL, "documentation_url": DOC_URL,
        "attribution": "Alex Cookson; Playbill reporting of Broadway League grosses; TidyTuesday 2020-04-28",
        "verified_at_utc": utc_now(), "acquisition": receipt,
        "bytes": Path(raw_path).stat().st_size, "sha256": sha256(raw_path),
        "raw_rows": stages[0]["rows"], "raw_columns": len(SCHEMA), "cleaned_rows": n,
        "first_week": bounds[0], "last_week": bounds[1], "eligible_rows": eligible,
        "missing_potential_gross": missing["potential_gross"], "missing_top_ticket_price": missing["top_ticket_price"],
        "exact_duplicate_rows": duplicates["exact_duplicates_removed"], "conflicting_keys": duplicates["conflicting_keys"],
        "final_eligible_rows": final.count(), "final_distinct_weeks": final.select("week_ending").distinct().count(),
        "raw_schema": SCHEMA.jsonValue(), "software": versions(), "spark_version": spark.version,
        "java_version": spark.sparkContext._jvm.java.lang.System.getProperty("java.version"),
        "forecast_contract": "Nominal USD, observed continuing show-theatre target weeks; four consecutive earlier reported weeks.",
        "run_caveat": "Inferred from seven-day continuity, not official production IDs. Gaps and venue moves split runs.",
        "target_and_keys_never_imputed": True}
    write_json(audit_dir / "input_manifest.json", manifest)
    # Independent reference comparisons are diagnostics; never processing constants.
    references = {"bytes": 4904731, "sha256": "1e383807947d11121737abe21c3b134011b878c5bd8a8cc4c6bc4e7cc7b04e4d",
                  "raw_rows": 47524, "raw_columns": 14, "first_week": "1985-06-09", "last_week": "2020-03-01",
                  "eligible_rows": 42039, "final_eligible_rows": 3285, "final_distinct_weeks": 113,
                  "missing_potential_gross": 12613, "missing_top_ticket_price": 11357,
                  "exact_duplicate_rows": 0, "conflicting_keys": 0}
    write_csv(audit_dir / "snapshot_crosschecks.csv", [{"check": k, "reference": v,
              "observed": manifest[k], "matches": str(manifest[k]) == str(v)} for k, v in references.items()])
    frame.unpersist()
    clean.unpersist()
    raw.unpersist()
    return manifest
