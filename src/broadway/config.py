from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE_URL = "https://raw.githubusercontent.com/rfordatascience/tidytuesday/main/data/2020/2020-04-28/grosses.csv"
DOC_URL = "https://github.com/rfordatascience/tidytuesday/tree/main/data/2020/2020-04-28"
NUMERIC = ["weekly_gross_lag1", "seats_sold_lag1", "pct_capacity_lag1",
           "performances_lag1", "previews_lag1", "gross_prior_4_mean", "run_weeks_known"]
CATEGORICAL = ["theatre", "month"]
PREDICTORS = NUMERIC + CATEGORICAL
KEYS = ["show", "theatre", "week_ending"]
TARGET = "weekly_gross"
RAW = ROOT / "data/raw/grosses.csv"
FEATURES = ROOT / "data/processed/broadway_features.parquet"
AUDIT = ROOT / "outputs/audit"
MODELS = ROOT / "outputs/models"
FIGURES = ROOT / "outputs/figures"
TABLES = ROOT / "outputs/tables"
# Decision threshold is recorded before any validation or final evaluation.
KPI_MAE_REDUCTION_PCT = 5.0
FOLDS = [(2014, 2015), (2015, 2016), (2016, 2017)]
FINAL_START = "2018-01-01"
FINAL_END = "2020-03-01"
GRID = {
    "ridge": [{"alpha": a} for a in [1.0, 10.0, 100.0]],
    "hist_gradient_boosting": [
        {"learning_rate": rate, "max_leaf_nodes": leaves}
        for rate in [0.05, 0.10] for leaves in [15, 31]
    ],
}
