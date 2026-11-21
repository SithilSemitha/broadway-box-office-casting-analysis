# Broadway box-office forecasting — technical coursework project

Forecast an observed continuing production's next reported weekly gross at a specific theatre, in **nominal USD**. Four consecutive earlier reported weeks are required. The prior week's report is assumed available at forecast time; the source has no reporting timestamps. This project produces code and evidence for independent verification. The student writes the coursework analysis and interpretation.

The data is Alex Cookson's Broadway Weekly Grosses, sourced from Playbill's reporting of Broadway League figures and released in [TidyTuesday, 28 April 2020](https://github.com/rfordatascience/tidytuesday/tree/main/data/2020/2020-04-28). See [upstream documentation](docs/upstream_readme.md) and [data dictionary](docs/data_dictionary.csv). Data ends before the COVID-era shutdown. Gross is revenue, and does not establish profit, demand, intervention effects or current-market performance. Spark demonstrates preprocessing that can scale; this CSV does not require a cluster.

## Environment and clean run

**Current delivery (20 November 2026):** the authorised, once-only final evaluation is complete. All 28 tests pass; both notebooks execute in evidence-review mode. The frozen data, model and source hashes still match. Run `python scripts/verify_artifacts.py --final` to verify the saved results. Preserve the evaluated data, model files and final receipt. The development commands below document a fresh reproduction, not a rerun to perform over this evaluated delivery.

| Final method | MAE (USD) | RMSE (USD) | Signed error (USD) | MAE reduction vs naive |
|---|---:|---:|---:|---:|
| Naive | 109,805.43 | 191,796.28 | 3,431.40 | 0.00% |
| Ridge | 103,087.47 | 171,631.36 | 1,120.15 | 6.12% |
| Histogram gradient boosting | 100,291.81 | 168,938.88 | -20,970.87 | 8.66% |

All methods cover 3,285 eligible rows across 113 weeks. Both ML candidates meet the example KPI on the final sample, but the development-selected method remains **naive**. The final comparison does not retroactively change the frozen selection. Full precision and subgroup evidence are saved under `outputs/tables/final_*`.

Use Python 3.11+ and Java 17. Dependencies are pinned in `requirements.txt`; actual verified runtime versions are recorded in `outputs/audit/spark_smoke.json` and `input_manifest.json`. PySpark 4.0.3's [installation requirements](https://spark.apache.org/docs/4.0.3/api/python/getting_started/install.html) specify Python 3.9+ and Java 17+. The run uses local Spark with two execution threads. No paid services or external database are needed.

From this directory, in PowerShell:

```powershell
python --version
java -version
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --index-url https://pypi.org/simple -r requirements.txt
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
# Point this at your actual JDK 17 installation if Java is not already configured:
$env:JAVA_HOME = 'C:\Program Files\Java\jdk-17'
# start_spark() resolves workers through this interpreter's directory on Windows.
.\.venv\Scripts\python.exe -m broadway smoke
.\.venv\Scripts\python.exe -m pytest -q --junitxml=outputs/audit/test_results.xml
.\.venv\Scripts\python.exe -m broadway download
.\.venv\Scripts\python.exe -m broadway preprocess
.\.venv\Scripts\python.exe -m broadway eda
.\.venv\Scripts\python.exe -m broadway validate
```

If PowerShell activation is disabled, the explicit interpreter commands above still work. In Linux/macOS use `python3 -m venv .venv`, `.venv/bin/python` and an appropriate `JAVA_HOME`. Avoid changing the global Python installation. If Spark or its Parquet filesystem smoke test fails, resolve that environment error before continuing; there is no pandas fallback. See the local run record in [verification.md](docs/verification.md) for any machine-specific setup or blockers discovered during verification.

On native Windows, Hadoop local filesystem access may require `hadoop.dll` and `winutils.exe`. The optional command below downloads **community-built** Hadoop 3.4.1 helpers from the named release, verifies the pinned archive SHA-256 and installs only those two files under `.tools/hadoop/bin`. Inspect `scripts/setup_windows_hadoop.py` and its source URL before using it. `start_spark()` discovers that project-local location without changing global system settings. This is not an Apache binary release. An existing explicit `HADOOP_HOME` takes precedence; use compatible helpers for that installation. Linux/Colab avoids this Windows step.

```powershell
.\.venv\Scripts\python.exe scripts/setup_windows_hadoop.py
.\.venv\Scripts\python.exe -m broadway smoke
```

Offline CSV route, in place of `download`:

```powershell
.\.venv\Scripts\python.exe -m broadway download --local-csv 'C:\path\to\grosses.csv'
```

Existing `data/raw/grosses.csv` is reused without replacement; a different supplied file stops the import. Preserve the old snapshot under another name before intentionally switching sources. The acquisition receipt records whether the file was downloaded or supplied locally. An existing local file without a receipt has an unknown upstream access date. The manifest records when it was checked. Downloads are not automatically updated.

## Notebook route

The notebooks review saved evidence by default and expose the same source functions as the CLI behind explicit reproduction switches:

1. `notebooks/01_spark_preprocessing.ipynb`: Spark smoke test, acquisition, audit, features and Parquet.
2. `notebooks/02_eda_and_modelling.ipynb`: six development figures, coverage, tuning and the locked final switch.

Select the `.venv` Python kernel in your notebook editor. To register and execute it headlessly:

```powershell
.\.venv\Scripts\python.exe -m ipykernel install --user --name broadway-cw --display-name 'Broadway CW (.venv)'
.\.venv\Scripts\python.exe scripts/execute_notebook.py notebooks/01_spark_preprocessing.ipynb
.\.venv\Scripts\python.exe scripts/execute_notebook.py notebooks/02_eda_and_modelling.ipynb
```

The execution script uses its own interpreter's kernel. Both notebooks now display audits, source excerpts, figure guidance, model comparisons and independent calculations. `REBUILD_PREPROCESSING=False` and `REBUILD_DEVELOPMENT=False` preserve saved evidence during Run All; their rebuild paths refuse to proceed after a final receipt exists. `scripts/make_notebooks.py` replaces notebook cells and clears outputs, so execute both after regeneration. `RUN_FINAL_TEST=False` stays off; the second notebook displays already saved final results when the completed receipt exists.

After a complete development run, `python scripts/verify_artifacts.py` checks the snapshot, figures, split boundaries, prediction/score consistency, subgroup counts/errors and frozen hashes without scoring the real final period. After a deliberately unlocked final run, `python scripts/verify_artifacts.py --final` verifies the saved final keys, source actuals, MAE/RMSE/signed error, KPI flags and subgroup tables against the frozen protocol and receipt. It does not fit or score models again. `python scripts/package_project.py` creates `dist/broadway_cw_project.zip` with code, notebooks, source data, Parquet and evidence. It omits `.venv`, native helpers and logs; recreate the environment after extracting. `outputs/audit/environment.freeze.txt` records the full local dependency snapshot; use the platform-independent `requirements.txt` instructions for Colab.

## Processing and quality policies

`Raw CSV → Spark explicit-schema ingestion → quality audit → continuity windows → Parquet → pandas/PyArrow plotting and sklearn pipelines`.

The source-order 14-column Spark schema is in `src/broadway/spark_pipeline.py`. A standard-library CSV check validates header order, field counts and date token shape. Spark parses values with the explicit schema and audits corrupt records; malformed input stops. Nulls, numeric NaNs and infinities are recorded. Exact duplicates are removed; conflicting show–theatre–week keys stop for inspection. Missing/blank keys, non-Sunday dates and missing, negative or nonfinite targets are excluded with individual reasons. Targets and keys are never imputed. Optional nonfinite measurements become null. Excluded target weeks create historical gaps.

Reported zero gross, zero theatre seating capacity, over-100% occupancy and unusually high gross are flagged and retained for review. Occupancy is a fraction and standing-room sales can exceed 1. The high-gross review threshold uses Q3 + 3 IQR estimated through 2014 only; it is a review flag, not an exclusion. Examples are limited to that exploratory period, while aggregate quality counts cover the entire source. No winsorising or automatic capping is applied. Optional ticket-price fields are excluded from the initial predictors.

An inferred run starts when a show–theatre pair has no previous row or a gap other than seven days. Lag features and the previous-four gross mean use only earlier rows within that run. Run identifiers are scoped to the show/theatre pair. Gaps and theatre moves split groups; repeated names and continuous revivals cannot be reliably resolved without official production IDs. `run_weeks_known` is observed earlier history, not official production age. All cleaned rows remain in Parquet; only eligible rows fit or score models.

The predictor allow-list is exactly: prior-week gross, seats sold, reported occupancy, performances and previews; prior-four-week gross mean; number of earlier observed run weeks; theatre; and known target calendar month. Current-week realised fields, market totals, year, eligibility and inferred run ID never enter model inputs. Calendar month is categorical.

## Development protocol and final lock

Exploration uses target weeks through 2014, with the lag relationship restricted to eligible rows. Month boxplots use 2010–2014. Representation tables also use only the exploratory period. Observed run-length bands in those tables are truncated at the EDA boundary. No final-period feature selection is supported.

All three methods use identical eligible rows. The naive benchmark repeats prior gross. Ridge uses fold-local median imputation, numeric scaling and dense one-hot encoding; alpha is 1, 10 or 100. Histogram gradient boosting uses fold-local median imputation and dense one-hot encoding, with learning rate 0.05/0.10, 15/31 leaves, 150 iterations, absolute error, no early stopping and seed 42. Unseen categories are ignored safely. Predictions are not clipped.

Expanding folds train through 2014/2015/2016 and validate on 2015/2016/2017 respectively. Complete target weeks stay together. Mean fold MAE (equal fold weights) chooses settings for each model, with deterministic tie breaking. The example KPI is **5% lower MAE than naive**, declared in `config.py` before evaluation. The validation decision rule chooses the better ML method only if it meets that threshold; otherwise it retains naive. It does not force an ML winner.

Validation refits each selected model on all eligible pre-2018 records and saves it. `outputs/models/frozen_protocol.json` records settings, decision rule, code/dependency/data/model hashes, fit size and runtime versions. It contains no final metrics. The final period is 1 January 2018 through 1 March 2020. Earlier reported holdout outcomes may serve as lags for subsequent weeks; they never fit or tune models. This is rolling one-week-ahead backtesting of observed continuing shows, and cannot score unobserved openings, missing weeks or closures. Production use additionally needs a future schedule and actual release timestamps.

**Original once-only evaluation command (already completed in this delivery; do not rerun):**

```powershell
.\.venv\Scripts\python.exe -m broadway final --final-test
.\.venv\Scripts\python.exe scripts/verify_artifacts.py --final
```

`python -m broadway final` alone refuses to score. An explicit unlock verifies the frozen protocol, source code, dependencies, Parquet and fitted model hashes, then creates `outputs/tables/final_evaluation_receipt.json` before scoring. A subsequent attempt or development retuning is refused while that receipt exists. If scoring fails, inspect the receipt and error first; do not delete it merely to experiment. Hash checking detects accidental changes, and is not a security barrier against someone deliberately editing the project.

Tests exercise the final scoring/output path on synthetic fixtures. The real holdout was preserved during implementation and evaluated once after authorisation on 20 November 2026. Do not tune on its outcomes or repeat the scoring. Every metrics table reports MAE, RMSE and prediction-minus-actual signed error in dollars; positive signed error means overprediction. MAE reduction is `100 * (baseline_mae - model_mae) / baseline_mae`, undefined for a zero baseline.

## Evidence locations

| Location | Technical evidence |
|---|---|
| `data/raw/grosses.csv`, `.source.json` | Preserved source bytes and acquisition receipt |
| `data/processed/broadway_features.parquet/` | All cleaned rows and time-safe features, including ineligible rows |
| `outputs/audit/input_manifest.json` | Attribution, access/verification timestamps, size/hash, date range, schema and actual versions |
| `outputs/audit/row_counts.csv`, `missingness.csv`, `duplicates.json` | Reconciled stage counts, null/NaN rates and conflict audit |
| `outputs/audit/excluded_rows.csv`, `parse_audit.json`, `malformed_csv_structure.csv` | Explicit exclusions and parse checks |
| `outputs/audit/anomaly_*` | Flag counts, development examples and retention policy |
| `outputs/audit/leakage_example.json`, `parquet_roundtrip.json` | Real four-week calculation and complete roundtrip verification |
| `outputs/audit/snapshot_crosschecks.csv` | Independent brief reference comparisons; no reference drives processing |
| `outputs/figures/figure_01_...png` through `figure_06_...png` | Six labelled, numbered figures |
| `outputs/tables/figure_*`, `lag_relationship.json` | Figure data, Spearman association and sample size |
| `outputs/tables/coverage_*`, `outlier_review_through_2014.csv` | Year/theatre/show/run-length representation and unusual gross review |
| `outputs/tables/validation_tuning.csv`, `validation_summary.csv`, `validation_fold_boundaries.csv` | Every parameter/fold score, mean scores and split evidence |
| `outputs/tables/validation_predictions.csv`, `validation_subgroups_*` | Selected-settings validation errors by keys, observed run age and theatre |
| `outputs/models/*_pre2018.joblib`, `frozen_protocol.json` | Two fitted ML pipelines and frozen protocol; load only trusted local artifacts |
| `outputs/tables/final_*` | Created only by the student's explicit final command |
| `outputs/audit/final_verification.json` | Created after successful verification of already saved final results |
| `docs/student_questions.md` | Prompts for independent interpretation, stakeholder planning and pilot design |

Subgroup tables include counts and week counts and flag groups under 30 observations for inspection; that flag is an example review convention, not a statistical guarantee. Repeated observations within long runs limit independent-information claims. The Spearman p-value is descriptive and not adjusted for within-run dependence.

## Google Colab route

This route is documented; whether it was executed is stated separately in `docs/verification.md`.

1. Upload the **whole project** as a zip or place it in Google Drive. Open a Colab notebook, mount Drive, and `%cd` to the extracted project root (which must contain `src/` and `requirements.txt`). Avoid copying only the notebooks.
2. Check `!python --version` and `!java -version`. If Java is absent or incompatible, use `!apt-get -qq update` and `!apt-get -qq install openjdk-17-jdk-headless`, then set `os.environ['JAVA_HOME']` to the actual installed JDK path (commonly `/usr/lib/jvm/java-17-openjdk-amd64`).
3. Run `%pip install -r requirements.txt` then `%pip install --no-deps -e .`. Restart the runtime if Colab requests it after replacing loaded packages; mount Drive and `%cd` again.
4. Run `!python -m broadway smoke`, `!python -m pytest -q`, then the same `download`, `preprocess`, `eda` and `validate` commands. The notebook source drivers also work from the project root. Leave `RUN_FINAL_TEST=False`.
5. For session retention, work directly in a persistent Drive project folder or copy **the whole** `data/processed/broadway_features.parquet/` directory plus raw data, metadata and `outputs/` to Drive. A Parquet directory is not one file. Restore it intact before another session's EDA/modelling. For reproducibility, preserve fitted models and the frozen protocol along with it. Rewriting/repartitioning Parquet changes the frozen byte fingerprint.
6. Colab CPU/RAM and installed Python may change; repeat the smoke test and inspect recorded versions. Only unlock the final holdout in the same frozen environment after your own review. A changed runtime requires restoring matching packages/runtime or revisiting development verification **before** accessing final results.

## Student handoff

The [focused coursework report](outputs/reports/Broadway_CW_Final_Report.docx) follows Tasks 1–5 with substantive discussion, final results, six figures and Harvard-style references, without appendices. It is approximately 2,465 words including task headings, table text and figure captions; the exact build estimate is in `outputs/reports/final_report_build_record.json`. It is AI-assisted text for student review and rewriting under the assessment policy, not independently authored student work. Complete the identity fields and check the final word count, references and pagination in Word. Rebuild with `python scripts/build_final_report.py` using the existing report dependencies. The longer technical reference below remains available for source-code explanations.

An expanded [technical working report in Word](outputs/reports/Broadway_CW_Technical_Working_Report.docx) and [Markdown copy](outputs/reports/Broadway_CW_Technical_Working_Report.md) follow the assessment's Tasks 1–5. They include the verified figures, audit/model tables and main source-code listings, with independent-analysis fields clearly identified. The complete document includes a substantial code appendix; check the coursework counting rules and write/edit your own assessed interpretation. The verified final results are included separately from validation. The report is built from saved artifacts by `scripts/build_technical_report.py`; it does not train or score models. To regenerate, install rendering-only dependencies using `python -m pip install --target .tools/report_deps python-docx==1.2.0`, then run `python scripts/build_technical_report.py`. This keeps the frozen modelling environment unchanged. DOCX content/package checks are recorded in `outputs/reports/report_build_record.json`; final pagination should be reviewed in Word.

See [assessment coverage](docs/assessment_coverage.md), [independent verification worksheet](docs/independent_verification.md), [actual execution and checkpoints](docs/verification.md) and [independent questions](docs/student_questions.md). The coverage checklist maps each assessment requirement to project evidence and identifies work the student must complete. The worksheet provides reproducible checks and blank fields for the student's own verification and decision planning. Independently inspect source provenance, audit decisions, leakage example, plots, folds and metrics. Choose and justify a practical stakeholder decision and KPI; design any low-risk shadow pilot yourself. Costs, intervention outcomes, profit and current-market validation are absent. Check the actual coursework deadline with the lecturer/LMS. The Word document is an expanded technical working reference with student-analysis fields; it is not a completed coursework submission.

The [submission workbook](outputs/reports/Broadway_CW_Submission_Workbook.docx) provides the required chapter structure, a suggested 2,500-word allocation, evidence pointers and fields for your own analysis. It is not a completed report. The expanded technical report exceeds the assessed word limit; keep it as a reference. See [current coursework review](docs/coursework_review.md) for the completed checks and remaining submission work.
