import argparse
import json

from .config import AUDIT, FEATURES, RAW


def main():
    parser = argparse.ArgumentParser(description="Broadway evidence pipeline; real final holdout locked by default")
    parser.add_argument("stage", choices=["download", "smoke", "preprocess", "eda", "validate", "final"])
    parser.add_argument("--local-csv", help="Import an already downloaded source CSV without network access")
    parser.add_argument("--final-test", action="store_true", help="Explicitly unlock frozen real final evaluation")
    args = parser.parse_args()
    if args.stage in {"download", "smoke", "preprocess"}:
        from .spark_pipeline import acquire, preprocess, smoke, start_spark
        if args.stage == "download":
            print(acquire(local_csv=args.local_csv))
            return
        spark = start_spark()
        try:
            if args.stage == "smoke":
                from .io import write_json
                result = smoke(spark, AUDIT / "smoke.parquet")
                write_json(AUDIT / "spark_smoke.json", result)
            else:
                acquire(local_csv=args.local_csv)
                result = preprocess(spark)
            print(json.dumps(result, indent=2, default=str))
        finally:
            spark.stop()
    elif args.stage == "eda":
        from .eda import generate_eda
        print(json.dumps(generate_eda(), indent=2))
    elif args.stage == "validate":
        from .modelling import run_validation
        run_validation()
    else:
        from .modelling import run_final
        print(run_final(unlocked=args.final_test).to_string(index=False))


if __name__ == "__main__":
    main()
