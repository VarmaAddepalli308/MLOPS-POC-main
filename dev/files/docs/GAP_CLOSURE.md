# Gap closure checklist (vs. fraud_mlops reference)

## A. Missing vs. reference
| # | Gap | Resolved in |
|---|---|---|
| 1 | Serving gate/step in training job | `resources/training_job.yml` (`update_serving` → `update_serving_run`) |
| 2 | `refresh_monitor` in monitoring job | `resources/monitoring_job.yml` |
| 3 | monitor/refresh/simulate/inspect not in jobs | monitor + refresh in monitoring job, inspect in ingestion job; `simulate_drift` documented in RUNBOOK (manual, as in reference) |
| 4 | Feature Engineering client | `src/features.py` |
| 5 | Drift simulation feeding the monitor | `src/simulate_drift.py` (writes scored windows to inference log) |
| 6 | Drift check on monitor output | `src/check_drift.py` (monitor KS + PSI/KS) |
| 7 | `model_params.yml` | `conf/model_params.yml`, read by `train.py` |
| 8 | Training widgets | `train.py` (selection_metric, search_n_iter, search_cv, val/test fractions, max rows) |
| 9 | Run tags/params/artifacts/task values | `train.py` (tags, parent_run_id, per-symbol artifact, sklearn pin, summary table, unknown-model warning) |
| 10 | AGENTS/CLAUDE/.vscode/fixtures/.gitignore | repo root |
| 11 | pyproject dev deps | `pyproject.toml` (databricks-connect, dlt, ipykernel; wheel package fix) |
| 12 | conftest Spark fixtures | `tests/conftest.py`, `tests/test_spark_features.py` |
| 13 | databricks.yml: schema var, uuid, prod permissions, user root_path | `databricks.yml` |

## B. Broken / misleading items
| # | Issue | Fix |
|---|---|---|
| 1 | Monitor target table never written | `batch_inference.py` / `simulate_drift.py` write `serving.stock_inference_log` with all monitor columns |
| 2 | No label back-fill | `src/backfill_labels.py` (+ monitoring job task) |
| 3 | `project.yml` unused | all notebooks read it via `utils.load_project()` |
| 4 | Schemas not created | `setup_environment.py` + `CREATE SCHEMA IF NOT EXISTS` in each writer |
| 5 | Gate params not wired | `evaluate/register/promote` read config; jobs pass `max_validation_rmse`, `min_improvement` via bundle variables |
| 6 | Dead/mislabeled code | `ema_12` removed; MACD documented as SMA variant; partial windows now NULL; unused `score()` removed; `utils.py` is now the shared module |
| 7 | Comment-only `serving_endpoint.yml` | removed (endpoint lives in PROD target) |
| 8 | Raw-price target risk | kept price target per decision; persistence gate on val **and** test; documented in README |

## Needs your input / not verifiable offline
- Canonical OHLCV field names are configured in `conf/project.yml`; Yahoo Finance is normalized into that schema.
- Workspace hosts (`databricks.yml`) and `authors` in `pyproject.toml`.
- Nothing here was executed on Databricks (no workspace access): `databricks bundle validate`, Feature Engineering, Model Serving and Lakehouse Monitoring calls must be validated on first deploy.
