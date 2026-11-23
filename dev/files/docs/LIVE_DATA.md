# Live Yahoo Finance workflow

Yahoo Finance is the project's primary and only configured market-data source. The workflow has no Parquet upload, Volume, or file-ingestion dependency.

## One-time migration when Bronze contains Parquet-loaded rows

1. Configure `live_data.symbols` in `conf/live.yml`. The default is a 15-symbol liquid US equity universe.
2. Run `stock_setup_job` to create the catalog schemas.
3. Back up Bronze if the old data may be needed.
4. Run `stock_yfinance_reset_job` ONCE. It downloads up to `max_lookback_days` of Yahoo Finance daily OHLCV first, then overwrites the existing `bronze.stock_raw` table (same table name) with only `_source_file = 'live:yfinance'` rows. It then rebuilds the existing Silver and Features tables. It does not create a parallel live table and is not scheduled.
5. Run `stock_training_job` to train/evaluate/register from the rebuilt feature history.

If Bronze is absent or already contains only Yahoo Finance data, skip the reset job and run `stock_training_job` directly.

## Daily run

`stock_live_job` is scheduled for 18:30 America/New_York, Monday-Friday:

```text
ingest_live_data -> build_features -> batch_inference -> run_monitoring
```

- Ingestion re-downloads an overlap window to catch vendor corrections and MERGEs by symbol/session date.
- Features are rebuilt from retained Bronze Delta history.
- The champion scores the newest fresh row for each symbol.
- The next session's realised close is backfilled into prior inference rows.
- Monitoring checks model performance and drift; a drift signal can trigger retraining. The training job refreshes Yahoo data before training and uses the accumulated Delta history. Evaluation and champion/challenger promotion gates remain in force.

## Verification SQL

```sql
SELECT _source_file, COUNT(*) AS rows, MAX(timestamp) AS newest_bar
FROM stock_demo.bronze.stock_raw GROUP BY _source_file;

SELECT symbol, MAX(trade_ts) AS newest_feature_date
FROM stock_demo.features.stock_features GROUP BY symbol;

SELECT symbol, trade_ts, prediction, actual_next_close, model_version, request_ts
FROM stock_demo.serving.stock_inference_log ORDER BY request_ts DESC LIMIT 50;
```

Expected Bronze source marker: `live:yfinance`. The actual label is null until the next observed trading session has arrived and the monitoring job backfills it.

## Settings and limitations

- `symbols`: tickers to download. Set an explicit manageable universe; free Yahoo Finance endpoints are not designed for very large universes.
- `max_lookback_days`: initial historical lookback, default 1,100 calendar days.
- `overlap_days`: re-download overlap, default 5 calendar days.
- `auto_adjust`: must remain consistent across runs. Existing-price mismatch protection stops unsafe mixing of adjusted and unadjusted history.
- The scheduled job runs after the regular US market close; market holidays and provider outages may yield no new bar.
- Workspace outbound HTTPS access to Yahoo Finance is required. Yahoo Finance is unofficial and can rate-limit or change; use a licensed provider for production.
- Drift-triggered retraining is enabled by default; it is not an unconditional daily retrain. Manual retraining is available via `stock_training_job`.
