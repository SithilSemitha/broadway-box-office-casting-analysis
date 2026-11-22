# Broadway Box-Office Forecasting

A time-series machine learning project that forecasts the next reported weekly box-office gross of an ongoing Broadway production at a specific theatre, using historical performance data.

## Project Overview

This project explores whether historical Broadway performance data can be used to predict a production's upcoming weekly gross in nominal USD.

Using Alex Cookson's Broadway Weekly Grosses dataset, sourced from Playbill's reporting of Broadway League figures, the project implements a complete data engineering and machine learning pipeline.

The main objective is to compare traditional forecasting against machine learning approaches while maintaining a strict time-based evaluation process to prevent data leakage.

**Dataset:** [Broadway Weekly Grosses – TidyTuesday (28 April 2020)](https://github.com/rfordatascience/tidytuesday/tree/main/data/2020/2020-04-28)

## Technology Stack

- **Python** – Data processing and machine learning
- **Apache Spark (PySpark)** – Data ingestion, cleaning and preprocessing
- **Pandas & PyArrow** – Data analysis and Parquet handling
- **Scikit-learn** – Model development and evaluation
- **Jupyter Notebook** – Exploratory analysis and experimentation
- **Java 17** – Spark runtime

## Project Workflow

The project follows an end-to-end forecasting pipeline:

1. **Data Acquisition:** Retrieve and preserve the original Broadway weekly grosses dataset.
2. **Data Preprocessing:** Use Spark to validate the schema, handle missing values, detect duplicates and identify data quality issues.
3. **Feature Engineering:** Generate historical gross values, four-week rolling averages, production run history and calendar features.
4. **Exploratory Data Analysis:** Investigate revenue trends, theatre performance, occupancy and historical relationships.
5. **Model Development:** Implement and compare three forecasting approaches:
   - Naive baseline
   - Ridge Regression
   - Histogram Gradient Boosting
6. **Model Evaluation:** Use expanding-window time-series validation and a separate, once-only final evaluation period.

## Forecasting Results

**Final evaluation: 3,285 predictions across 113 reporting weeks.**

| Model | MAE (USD) | RMSE (USD) | MAE Improvement |
|---|---:|---:|---:|
| Naive | 109,805.43 | 191,796.28 | Baseline |
| Ridge Regression | 103,087.47 | 171,631.36 | 6.12% |
| Histogram Gradient Boosting | 100,291.81 | 168,938.88 | 8.66% |

Although Histogram Gradient Boosting achieved the lowest final MAE, the development-stage selection protocol retained the **Naive baseline** as the selected forecasting method.

This distinction is intentional: the final evaluation is used for independent comparison and does not change the previously frozen model selection.

## Key Technical Considerations

- Four consecutive historical reporting weeks are required for forecasting eligibility.
- Historical features are calculated within continuous show–theatre reporting periods.
- Missing reporting weeks and theatre changes break historical continuity.
- Only information available before the target reporting week is used.
- All forecasting methods are evaluated on identical observations.
- The final evaluation period is kept separate from model development and tuning.
- Predictions and evaluation metrics are reported in nominal USD.

## Project Structure

```text
broadway-box-office-forecasting/
│
├── data/
│   ├── raw/                 # Original source dataset
│   └── processed/            # Cleaned Parquet dataset
│
├── notebooks/
│   ├── 01_spark_preprocessing.ipynb
│   └── 02_eda_and_modelling.ipynb
│
├── src/broadway/             # Core processing and modelling code
├── scripts/                  # Verification and utility scripts
├── config.py                 # Project configuration
├── docs/                     # Technical documentation
├── outputs/
│   ├── audit/                # Data quality and verification evidence
│   ├── figures/              # Exploratory analysis figures
│   ├── models/               # Frozen trained models
│   ├── tables/               # Evaluation and analysis results
│   
│
├── tests/                    # Automated tests
├── requirements.txt
└── README.md
```

## Environment Setup

Requirements:
- Python 3.11+
- Java 17
- Windows, Linux or macOS

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip install --no-deps -e .
```

Linux/macOS:

```bash
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install --no-deps -e .
```

## Running the Project

For a fresh development reproduction, execute the following commands from the project root:

```bash
python -m broadway smoke
python -m pytest -q
python -m broadway download
python -m broadway preprocess
python -m broadway eda
python -m broadway validate
```

The Jupyter notebooks provide an alternative way to explore preprocessing, analysis and modelling.

## Final Evaluation and Verification

**Evaluation status: Completed – 20 November 2026.**

The authorised final evaluation has already been performed once. All 28 automated tests passed, and both notebooks executed successfully in evidence-review mode.

To independently verify the saved final results:

```bash
python scripts/verify_artifacts.py --final
```

This command verifies the saved evidence, model and source fingerprints, predictions, evaluation metrics and final evaluation receipt. It does not retrain or rescore the models.

**Important:** Do not rerun the final evaluation, modify the frozen artefacts or retune models using final-period results.

## Limitations

- The dataset ends before the COVID-era Broadway shutdown.
- The forecast concerns observed continuing productions, not new openings or unreported weeks.
- Gross revenue does not represent profit or establish audience demand.
- The project does not measure intervention effects or current Broadway market performance.
- Historical reporting timestamps are unavailable, so the previous week's report is assumed to be available at forecast time.
