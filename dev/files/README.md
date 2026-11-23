# Stock Price Prediction MLOps — Databricks Asset Bundle

A fully live-data-driven stock regression MLOps project. Yahoo Finance is the primary OHLCV source; the first run bootstraps up to three years of daily history directly into Bronze Delta, and subsequent runs incrementally merge corrected/new bars. No Parquet upload or file-ingestion workflow is required.

The project retains Delta Lake medallion tables, point-in-time-safe features, MLflow experiment tracking, chronological model bake-off, evaluation gates, Unity Catalog champion/challenger aliases, batch predictions, inference-label backfill, drift monitoring, drift-triggered retraining, TEST→PROD promotion, CI/CD and tests.

## Layout

| Path | Purpose |
|---|---|
| `conf/project.yml` | Tables, canonical OHLCV columns, training, evaluation, drift and serving settings |
| `conf/live.yml` | Yahoo Finance provider, initial ticker universe, lookback and overlap settings |
| `conf/model_params.yml` | Hyper-parameter grids |
| `src/ingest_live.py` | First-run historical bootstrap and daily incremental Yahoo Finance ingestion into Bronze Delta |
| `src/features.py`, `src/features_lib.py` | Silver cleaning and technical indicators / next-session labels |
| `src/train.py`, `src/evaluate.py` | Chronological bake-off and quality gate |
| `src/register.py`, `src/serve.py` | UC registration and champion/challenger promotion / serving |
| `src/batch_inference.py`, `src/backfill_labels.py` | Next-session prediction and actual-close label backfill |
| `src/monitor.py`, `src/refresh_monitor.py`, `src/check_drift.py` | Monitoring, drift checks and conditional retraining |
| `resources/*.yml` | Setup, training, monitoring, promotion and scheduled live jobs |
| `tests/` | Unit and Spark feature tests |

## End-to-end workflow

1. `stock_setup_job` creates the catalog schemas.
2. `ingest_live.py` downloads the configured tickers from Yahoo Finance. On the first run it fetches up to `max_lookback_days` of history and creates Bronze Delta; later runs re-download an overlap window and MERGE on `(symbol, day)`.
3. `features.py` rebuilds Silver and the Feature Engineering table, including rolling indicators and `target_next_close`.
4. `train.py` reads accumulated labelled rows from the Delta feature table, performs a chronological fit/validation/test split and MLflow model bake-off.
5. `evaluate.py` enforces persistence-baseline and configured quality gates.
6. `register.py` registers the challenger and updates the champion alias only when evaluation and improvement rules pass.
7. `serve.py` updates the serving endpoint when a new champion is promoted (PROD target).
8. `batch_inference.py` scores each fresh symbol-day once per model version and stores the prediction with a null actual label.
9. On a later trading session, `backfill_labels.py` fills the actual next close from the feature table.
10. Monitoring computes performance/drift; a drift signal can trigger the training job, which first refreshes Yahoo Finance data and then trains on the accumulated Delta history. Evaluation and promotion gates still apply.

## Model improvement update

The current training revision compares regression and UP/DOWN classification models. It adds date-grouped expanding-window validation, broader point-in-time technical features, hyperparameter search, per-symbol regression metrics, and MLflow comparison artifacts. See [docs/IMPROVEMENTS.md](docs/IMPROVEMENTS.md) for the exact changes, selection rules, and limitations.

The production serving endpoint remains a **price-regression** endpoint. Direction classifiers are trained, evaluated, and logged separately in MLflow; they are not silently substituted into the price endpoint.

## Setup and first run

1. Edit the workspace hosts in `databricks.yml` and select the ticker universe in `conf/live.yml`. The default universe contains 15 liquid US tickers; Yahoo Finance is a public unofficial source and may rate-limit requests.
2. Ensure the Databricks workspace can reach Yahoo Finance over outbound HTTPS.
3. Install local dependencies and validate/deploy:

```powershell
uv sync --dev  # resolves the updated optional model dependencies and creates a fresh uv.lock
databricks bundle validate -t dev
databricks bundle deploy -t dev
databricks bundle run stock_setup_job -t dev
```

4. If `stock_demo.bronze.stock_raw` already contains old Parquet-loaded rows, run the one-time replacement job FIRST. It downloads Yahoo Finance history and overwrites the SAME Bronze table only after the download succeeds; it then rebuilds the existing Silver and Features tables. This removes the old Bronze rows, including symbols outside the configured Yahoo universe. Back up the table first if you may need the old data.

```powershell
databricks bundle run stock_yfinance_reset_job -t dev
```

5. Run initial model training on the rebuilt Yahoo Finance feature history:

```powershell
databricks bundle run stock_training_job -t dev
```

6. Once a `champion` alias exists, run the daily workflow manually once to validate predictions and monitoring:

```powershell
databricks bundle run stock_live_job -t dev
```

7. Deploy to TEST/PROD for scheduled execution. `resources/live_job.yml` is configured for 18:30 `America/New_York`, Monday-Friday, after the US cash-market close. Databricks development mode can pause schedules automatically; verify the deployed schedule is enabled in TEST/PROD.

## Verify live data and predictions

```sql
-- Live source rows in Bronze
SELECT _source_file, COUNT(*) AS rows, MAX(timestamp) AS newest_bar
FROM stock_demo.bronze.stock_raw
GROUP BY _source_file;

-- Latest features per symbol
SELECT symbol, MAX(trade_ts) AS newest_feature_date
FROM stock_demo.features.stock_features
GROUP BY symbol ORDER BY newest_feature_date DESC;

-- Predictions and later actual labels
SELECT symbol, trade_ts, prediction, actual_next_close, model_version, request_ts
FROM stock_demo.serving.stock_inference_log
ORDER BY request_ts DESC LIMIT 50;
```

`_source_file` should be `live:yfinance`. A prediction is for the next observed trading session after `trade_ts`; `actual_next_close` is populated only after that session's data arrives and the monitoring task runs.

## Important operating notes

- Daily ingestion/prediction is scheduled on weekdays; market holidays and weekends naturally produce no new trading bars.
- Initial history is capped by `max_lookback_days` (default 1,100 calendar days). Yahoo Finance limits and symbol availability may vary.
- Retraining is drift-triggered by default, not an unconditional daily retrain. You can also run `stock_training_job` manually.
- Retraining always uses the accumulated labelled rows in the Delta feature table; the newest unlabelled session is excluded from supervised training.
- The test split is chronological and untouched during model selection. The candidate must pass evaluation before registration/promotion.
- Yahoo Finance is not a licensed production feed. Use a contracted market-data provider for production trading systems. Predictions are not investment advice.

## Windows PowerShell quick start for this improved revision

Use Python 3.11 (recommended for Databricks Runtime 15.4 / Databricks Connect compatibility) rather than Python 3.13 for this project's current dependency constraints.

```powershell
# 1. Move into the extracted project folder
Set-Location "C:\path\to\stock_price_prediction_mlops"

# 2. Create and activate a clean environment
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip uv

# 3. Install project and development dependencies
uv sync --dev  # resolves the updated optional model dependencies and creates a fresh uv.lock

# 4. Run local unit tests (Spark/Databricks-only tests may be skipped locally)
pytest -q

# 5. Confirm your Databricks CLI authentication profile is valid
databricks auth profiles

# 6. Validate and deploy the dev Asset Bundle
databricks bundle validate -t dev
databricks bundle deploy -t dev

# 7. Bootstrap schemas, if this is the first deployment
databricks bundle run stock_setup_job -t dev

# 8. Run the full Yahoo Finance -> features -> regression + classifier training workflow
databricks bundle run stock_training_job -t dev
```

If your Bronze table still contains an old non-Yahoo dataset and you explicitly want to replace it, back up the table and run `databricks bundle run stock_yfinance_reset_job -t dev` once **before** the training job. This reset overwrites the existing Bronze table. Do not run it as part of routine retraining.

After training, open the MLflow experiment configured by `conf/project.yml` (default `/Shared/stock-price-prediction`). Open the parent run `stock_regression_and_direction_bakeoff` and its artifacts `regression_model_comparison.csv` and `direction_classifier_comparison.csv`. The winning regression run continues through the existing evaluation, Unity Catalog registration, and serving steps when it passes the configured gates.
