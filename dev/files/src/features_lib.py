"""Point-in-time-safe technical-indicator logic (PySpark). Imported by features.py and the Spark tests.

Rules:
  * every feature at row t uses only rows <= t of the same symbol;
  * rolling statistics are NULL until the window is full (no partial-window values);
  * the label (next observed session's close) is the only column that looks forward.
"""
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from utils import FEATURE_COLS


def standardize_ohlcv(src, mapping, normalize_to_day=True):
    """Select/cast/clean the raw columns into symbol, trade_ts, open, high, low, close, volume.

    * normalize_to_day: daily data only. Truncates timestamps to midnight so rows from the historical file and
      from the live download (which may carry a different time of day) land on the same session.
    * if several rows share (symbol, trade_ts) the most recently ingested one wins (`_ingest_ts` in bronze),
      so corrected live values replace older ones deterministically.
    """
    required = [mapping[k] for k in ["date_col", "open_col", "high_col", "low_col", "close_col", "volume_col"]]
    missing = [c for c in required if c not in src.columns]
    if missing:
        raise ValueError(f"Check the canonical OHLCV column mapping in conf/project.yml `data:`. Missing: {missing}")

    ts = F.to_timestamp(F.col(mapping["date_col"]))
    if normalize_to_day:
        ts = F.date_trunc("day", ts)
    selects = [
        ts.alias("trade_ts"),
        F.col(mapping["open_col"]).cast("double").alias("open"),
        F.col(mapping["high_col"]).cast("double").alias("high"),
        F.col(mapping["low_col"]).cast("double").alias("low"),
        F.col(mapping["close_col"]).cast("double").alias("close"),
        F.col(mapping["volume_col"]).cast("double").alias("volume"),
    ]
    if mapping["symbol_col"]:
        if mapping["symbol_col"] not in src.columns:
            raise ValueError(f"Symbol column not found: {mapping['symbol_col']}")
        selects.append(F.col(mapping["symbol_col"]).cast("string").alias("symbol"))
    else:
        selects.append(F.lit("SINGLE_ASSET").alias("symbol"))

    if "_ingest_ts" in src.columns:
        selects.append(F.col("_ingest_ts").cast("timestamp").alias("_ingest_ts"))
    else:
        selects.append(F.lit(None).cast("timestamp").alias("_ingest_ts"))

    clean = src.select(*selects).dropna(subset=["trade_ts", "open", "high", "low", "close", "volume", "symbol"])
    clean = clean.filter((F.col("open") > 0) & (F.col("high") > 0) &
                         (F.col("low") > 0) & (F.col("close") > 0) & (F.col("volume") >= 0))
    newest_first = Window.partitionBy("symbol", "trade_ts").orderBy(F.col("_ingest_ts").desc_nulls_last())
    clean = (clean.withColumn("_rn", F.row_number().over(newest_first))
                  .filter(F.col("_rn") == 1).drop("_rn", "_ingest_ts"))
    return clean.withColumn("trade_date", F.to_date("trade_ts"))


def build_feature_frame(clean):
    """Build point-in-time-safe normalized technical features and next-session labels.

    Every feature at t uses data at or before t. Labels alone use the next observed
    session. Price-relative features make a pooled multi-symbol model less dependent
    on the nominal price scale of each ticker.
    """
    w = Window.partitionBy("symbol").orderBy("trade_ts")

    def roll(n):
        return w.rowsBetween(-(n - 1), 0)

    def full(expr, col, n):
        return F.when(F.count(col).over(roll(n)) >= n, expr(col).over(roll(n)))

    avg = lambda c: F.avg(c)  # noqa: E731
    std = lambda c: F.stddev(c)  # noqa: E731
    feat = clean.withColumn("prev_close", F.lag("close", 1).over(w))
    feat = feat.withColumn("return_1d", F.col("close") / F.col("prev_close") - 1.0)
    for n in (3, 5, 10, 20):
        feat = feat.withColumn(f"return_{n}d", F.col("close") / F.lag("close", n).over(w) - 1.0)
    for n in (1, 2, 5):
        feat = feat.withColumn(f"lag_close_{n}", F.lag("close", n).over(w))
    feat = feat.withColumn("lag_volume_1", F.lag("volume", 1).over(w))
    for n in (5, 10, 20, 50):
        feat = feat.withColumn(f"sma_{n}", full(avg, "close", n))
    # Exponential-moving-average approximations use rolling averages in Spark SQL;
    # the feature names deliberately say ma_* so the implementation is not mislabeled.
    for n in (12, 26, 50):
        feat = feat.withColumn(f"ma_{n}", full(avg, "close", n))
    for n in (5, 10, 20):
        feat = feat.withColumn(f"volatility_{n}", full(std, "return_1d", n))
    feat = (feat
        .withColumn("volume_sma_5", full(avg, "volume", 5))
        .withColumn("relative_volume", F.col("volume") / F.col("volume_sma_5"))
        .withColumn("high_low_range", (F.col("high") - F.col("low")) / F.col("close"))
        .withColumn("open_close_change", (F.col("close") - F.col("open")) / F.col("open"))
        .withColumn("overnight_gap", F.col("open") / F.col("prev_close") - 1.0)
        .withColumn("price_to_sma_5", F.col("close") / F.col("sma_5") - 1.0)
        .withColumn("price_to_sma_20", F.col("close") / F.col("sma_20") - 1.0)
        .withColumn("price_to_sma_50", F.col("close") / F.col("sma_50") - 1.0)
        .withColumn("macd", F.col("ma_12") - F.col("ma_26"))
        .withColumn("macd_signal", full(avg, "macd", 9))
        .withColumn("macd_hist", F.col("macd") - F.col("macd_signal"))
        .withColumn("bb_std_20", full(std, "close", 20))
        .withColumn("bb_upper", F.col("sma_20") + 2.0 * F.col("bb_std_20"))
        .withColumn("bb_lower", F.col("sma_20") - 2.0 * F.col("bb_std_20"))
        .withColumn("bb_position", (F.col("close") - F.col("bb_lower")) /
                    F.when((F.col("bb_upper") - F.col("bb_lower")) == 0, F.lit(None))
                     .otherwise(F.col("bb_upper") - F.col("bb_lower")))
        .withColumn("delta", F.col("close") - F.col("prev_close"))
        .withColumn("gain", F.greatest(F.col("delta"), F.lit(0.0)))
        .withColumn("loss", F.greatest(-F.col("delta"), F.lit(0.0)))
    )
    for n in (7, 14, 21):
        feat = feat.withColumn(f"avg_gain_{n}", full(avg, "gain", n))
        feat = feat.withColumn(f"avg_loss_{n}", full(avg, "loss", n))
        feat = feat.withColumn(f"rsi_{n}", F.when(F.col(f"avg_loss_{n}") == 0, 100.0)
                               .otherwise(100.0 - 100.0 / (1.0 + F.col(f"avg_gain_{n}") / F.col(f"avg_loss_{n}"))))
    for n in (3, 5, 10, 20):
        feat = feat.withColumn(f"momentum_{n}", F.col("close") / F.lag("close", n).over(w) - 1.0)
    prev_close = F.col("prev_close")
    tr = F.greatest(F.col("high") - F.col("low"), F.abs(F.col("high") - prev_close),
                    F.abs(F.col("low") - prev_close))
    feat = (feat.withColumn("true_range", tr)
        .withColumn("atr_14", full(avg, "true_range", 14))
        .withColumn("atr_14_pct", F.col("atr_14") / F.col("close"))
        .withColumn("day_of_week", F.dayofweek("trade_date"))
        .withColumn("month", F.month("trade_date"))
        .withColumn("target_next_close", F.lead("close", 1).over(w))
        .withColumn("target_direction", (F.col("target_next_close") > F.col("close")).cast("int"))
    )
    # Keep the latest row for inference; training drops null labels in train.py.
    feat = feat.dropna(subset=FEATURE_COLS)
    keep = ["symbol", "trade_ts", "trade_date", *FEATURE_COLS, "target_next_close", "target_direction"]
    return feat.select(*keep)
