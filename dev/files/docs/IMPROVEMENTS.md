# Model improvement implementation

This revision keeps the existing Yahoo Finance -> Bronze Delta -> Silver Delta -> Feature Engineering -> MLflow -> Unity Catalog -> batch inference/serving -> monitoring -> Asset Bundle CI/CD flow.

## What changed

1. **Feature engineering** (`src/features_lib.py`, `src/utils.py`): added 3/10-day returns, 50-day SMA, 12/26/50-day moving averages, price-to-moving-average ratios, relative volume, overnight gap, Bollinger position, RSI 7/14/21, momentum 3/5/10/20, 10-day volatility and ATR percentage. Every feature is calculated from current/past observations within each symbol. The only forward-looking columns are `target_next_close` and `target_direction`.
2. **Date-grouped validation** (`src/train.py`): expanding-window CV now splits on unique trading dates, keeping all tickers from one date together. The outer fit/validation/test split remains chronological.
3. **Regression bake-off**: ElasticNet, Random Forest, Extra Trees, HistGradientBoosting, GradientBoosting, XGBoost, LightGBM and CatBoost. Optional boosting libraries are installed on the training task; the script skips an optional model if its import is unavailable.
4. **Direction classification**: Logistic Regression, Random Forest, Extra Trees, HistGradientBoosting and GradientBoosting classifiers. Each receives a grouped time-series hyperparameter search using balanced accuracy. Accuracy, balanced accuracy, precision/recall for UP, F1, ROC-AUC and class rates are logged.
5. **MLflow comparison artifacts**: regression and classifier comparison CSVs are logged in the parent run. Each model is logged as a nested run with its own model artifact and validation/test metrics.
6. **Unity Catalog and CI/CD**: the best regression model remains registered as `models.stock_price_regressor`; the best direction classifier is independently registered as `models.stock_direction_classifier`. Both use `champion`/`challenger` aliases and are copied from TEST to PROD by the existing promotion workflow. The existing serving endpoint and batch inference contract remain regression-only; the classifier is registered for downstream evaluation/consumption but is not substituted into the price endpoint.

## Selection and evaluation

`conf/project.yml` selects the regression candidate by **validation directional accuracy**. The regression hyperparameter search itself optimizes return RMSE, then the outer validation set chooses the champion candidate. The classifier candidates are tuned and ranked by validation balanced accuracy; classifier promotion also requires the configured minimum test balanced accuracy (0.50 by default). Test metrics are for final reporting and promotion/evaluation gates only; they must not be used to tune features or choose hyperparameters.

The evaluation gate now requires the regression model to beat the persistence baseline within the configured 2% tolerance and to reach at least 48% test directional accuracy. Adjust these thresholds only after examining the data and business tolerance. A failed gate is preferable to promoting an unstable model.

## Important notes

- Higher directional accuracy is not guaranteed. A result around 50% can be a realistic outcome for next-session equity direction.
- A pooled multi-symbol model still sees absolute OHLCV columns (including `close` and `volume`) because the existing regression wrapper needs `close` to convert predicted returns into prices. Relative features are added alongside them.
- The moving-average family `ma_12`, `ma_26`, and `ma_50` is deliberately named as moving averages; these are rolling means, not exponential moving averages. `macd` is therefore the rolling-average MACD-style spread, consistent with the Spark implementation.
- The feature table is rebuilt by `features.py`. The first run after deployment should run the full training job so the table schema and all downstream models use the same feature list.
- The three optional packages increase cluster startup/install time. If the workspace blocks PyPI egress, remove their task libraries and corresponding YAML model blocks; the scikit-learn models will still run.
- Never interpret backtest metrics as a guarantee of future returns or use this POC as an autonomous trading system.
