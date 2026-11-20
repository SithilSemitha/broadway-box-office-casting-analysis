import csv
import hashlib
import importlib.metadata
import json
import platform
from datetime import datetime, timezone
from pathlib import Path


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(path, rows, fields=None):
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]) if rows else fields or [])
        writer.writeheader()
        writer.writerows(rows)


def versions():
    packages = ["pyspark", "pandas", "pyarrow", "numpy", "scipy", "scikit-learn",
                "matplotlib", "pytest", "nbformat", "nbclient", "ipykernel", "joblib", "py4j"]
    result = {"python": platform.python_version(), "platform": platform.platform()}
    for name in packages:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "not installed"
    return result


def parquet_fingerprint(path):
    files = sorted(Path(path).rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No Parquet parts found in {path}. Run preprocess first.")
    payload = "\n".join(f"{f.relative_to(path).as_posix()}:{sha256(f)}" for f in files)
    return hashlib.sha256(payload.encode()).hexdigest()
