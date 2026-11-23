"""Spark-level tests (skipped automatically when no Databricks Connect session is available)."""
import pytest

pytestmark = pytest.mark.spark

MAPPING = dict(date_col="date", symbol_col="symbol", open_col="open", high_col="high",
               low_col="low", close_col="close", volume_col="volume")


def test_features_are_point_in_time_safe(load_fixture):
    from features_lib import build_feature_frame, standardize_ohlcv
    from utils import FEATURE_COLS

    raw = load_fixture("sample_ohlcv.csv")
    feats = build_feature_frame(standardize_ohlcv(raw, MAPPING)).toPandas().sort_values(["symbol", "trade_ts"])

    assert not feats[FEATURE_COLS].isna().any().any()
    for _, g in feats.groupby("symbol"):
        g = g.reset_index(drop=True)
        # label is the NEXT row's close; the final row has no label yet but is kept for scoring
        assert (g.target_next_close.iloc[:-1].values == g.close.iloc[1:].values).all()
        assert g.target_next_close.iloc[-1] != g.target_next_close.iloc[-1]   # NaN
        # lag features only look backwards
        assert (g.lag_close_1.iloc[1:].values == g.close.iloc[:-1].values).all()
        # warm-up rows are dropped: first rows with partial windows must not exist
        assert len(g) <= 80 - 25


def test_duplicates_and_bad_prices_removed(spark):
    from features_lib import standardize_ohlcv

    rows = [("A", "2024-01-02", 10.0, 11.0, 9.0, 10.5, 100.0),
            ("A", "2024-01-02", 10.0, 11.0, 9.0, 10.5, 100.0),     # duplicate
            ("A", "2024-01-03", -1.0, 11.0, 9.0, 10.5, 100.0)]     # invalid price
    df = spark.createDataFrame(rows, ["symbol", "date", "open", "high", "low", "close", "volume"])
    assert standardize_ohlcv(df, MAPPING).count() == 1


def test_newest_ingest_wins_and_timestamps_normalised_to_day(spark):
    from datetime import datetime

    from features_lib import standardize_ohlcv

    rows = [("A", datetime(2024, 1, 2, 5, 0), 10.0, 11.0, 9.0, 10.5, 100.0, datetime(2024, 2, 1)),    # historical file
            ("A", datetime(2024, 1, 2, 0, 0), 10.0, 11.0, 9.0, 12.0, 100.0, datetime(2024, 3, 1))]    # live correction
    df = spark.createDataFrame(rows, ["symbol", "date", "open", "high", "low", "close", "volume", "_ingest_ts"])
    out = standardize_ohlcv(df, MAPPING).collect()
    assert len(out) == 1 and out[0]["close"] == 12.0
