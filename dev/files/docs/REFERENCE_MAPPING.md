# Reference POC (fraud_mlops) → Stock POC mapping

| Fraud reference | Stock implementation |
|---|---|
| `conf/model_params.yml` grids | `conf/model_params.yml` (regressor grids) + `conf/project.yml` |
| credit-card CSV ingestion (row-hash MERGE) | Yahoo Finance OHLCV ingestion (Delta MERGE) |
| `features.py` + `fe.create_table` | `features.py` + `features_lib.py`, `fe.create_table` (PK symbol, trade_ts) |
| fraud classification bake-off (RandomizedSearchCV) | regression bake-off (RandomizedSearchCV + TimeSeriesSplit) |
| avg precision / ROC AUC / precision@recall | MAE / RMSE / R² / directional accuracy / persistence RMSE |
| tuned decision threshold + `operating_points.csv` | `per_symbol_test_metrics.csv` |
| `evaluate.py` AP gate | RMSE vs persistence + optional ceiling / directional floor |
| UC `fraud_detector`, champion/challenger | UC `stock_price_regressor`, champion/challenger on validation RMSE |
| `serve.py` (prod-only guard) | `serve.py` (prod-only guard) |
| training job: register → serve gate | training job: register → `update_serving` gate → serve; plus batch inference |
| `fraud_detector_payload` table | `serving.stock_inference_log` |
| classification monitor | regression monitor (`monitor.py`, idempotent) |
| monitoring job: refresh → check_drift → gate → retrain | backfill_labels → ensure_monitor → refresh_monitor → check_drift → gate → retrain |
| KS drift from monitor metrics | monitor KS **plus** PSI/KS vs training period |
| `simulate_drift.py` (scores baseline + drifted, logs to payload) | same, shifting returns/volume/volatility; refuses PROD |
| `promote_model.py` | `promote_model.py` (validation RMSE) |
| `promote_job.yml` | `promote_job.yml` |
| DEV/TEST/PROD targets, prod permissions, endpoint | same (`databricks.yml`) |
| GitHub CI/CD | PR lint+tests+validate → TEST deploy/train → PROD deploy/promote |
| `conftest.py` Databricks Connect fixtures, `fixtures/` | same, with graceful skip |
| `.vscode/`, `AGENTS.md`, `CLAUDE.md`, `.gitignore` | same |

## Extension beyond the reference
- `stock_live_job` / `ingest_live.py` / `conf/live.yml`: daily live-data refresh and scheduled prediction (the fraud reference has no equivalent).

## Deliberate differences
- Time-series split replaces stratified random split; there is no separate test table.
- Target is the next observed session's close; windows are partitioned by symbol.
- Drift triggers investigation/retraining; a retrained model must still pass the gate.
- Labels arrive later, so `backfill_labels.py` exists (no fraud equivalent).
- Not copied on purpose: `.databricks/` deployment state (workspace-specific) and the reference's broken `sample_taxis_test.py`.
