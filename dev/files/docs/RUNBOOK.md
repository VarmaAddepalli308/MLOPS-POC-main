# Runbook

## One-time migration from Parquet-loaded Bronze data

If `stock_demo.bronze.stock_raw` currently contains Parquet-derived records, back up the table first if those records may be needed. Then deploy the bundle and run:

```powershell
databricks bundle validate -t dev
databricks bundle deploy -t dev
databricks bundle run stock_setup_job -t dev
databricks bundle run stock_yfinance_reset_job -t dev
```

`stock_yfinance_reset_job` downloads the configured Yahoo Finance history before it overwrites the existing `stock_demo.bronze.stock_raw` Delta table. It does not create a second live Bronze table. It removes all previous Bronze rows, including symbols not present in `conf/live.yml`, and writes rows with `_source_file = 'live:yfinance'`. It then rebuilds the existing Silver and Features tables. The reset job is intentionally unscheduled and must only be run once for this migration. If the Yahoo Finance download fails, the overwrite step is never reached and the existing Bronze table remains intact.

## Train after migration

```powershell
databricks bundle run stock_training_job -t dev
```

This trains/evaluates from the rebuilt Yahoo Finance feature history and applies the existing registration/promotion gates.

## First-time deployment without old Parquet data

If the Bronze table does not exist or is already Yahoo-only, run `stock_training_job` directly after setup; it bootstraps the table when needed.

## Daily operation

`stock_live_job` is configured for 18:30 America/New_York Monday-Friday in TEST/PROD. Confirm the deployed schedule is enabled. Databricks development mode may pause schedules. Daily ingestion uses incremental Delta MERGE and does not truncate history.

## Checks

- Bronze source: `_source_file = 'live:yfinance'`; no other source marker should remain after reset.
- Feature freshness: `MAX(trade_ts)` grouped by symbol in `features.stock_features`.
- Predictions and labels: `serving.stock_inference_log`; `actual_next_close` is backfilled after the next observed session.
- Training and promotion: inspect MLflow metrics, evaluation gate results, model versions and `champion` / `challenger` aliases.

## Recovery

If Yahoo Finance download fails, check outbound HTTPS access, ticker validity and provider rate limits. Re-run after the provider/network issue is resolved. The pipeline intentionally does not silently fall back to manually uploaded files.
