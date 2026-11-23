"""Shared, Spark-free helpers (importable from notebooks and unit tests).

Notebooks import this with `from utils import ...` (the notebook folder is on sys.path in
Databricks workspace files). Keep this module free of pyspark/dbutils so it stays testable.
"""
import os
import sys

import numpy as np

# --------------------------------------------------------------------------------------
# Single source of truth for the model input columns (features.py, train.py, batch_inference.py,
# simulate_drift.py and check_drift.py all use this list).
# --------------------------------------------------------------------------------------
FEATURE_COLS = [
    "open", "high", "low", "close", "volume",
    "return_1d", "return_3d", "return_5d", "return_10d", "return_20d",
    "lag_close_1", "lag_close_2", "lag_close_5", "lag_volume_1",
    "sma_5", "sma_10", "sma_20", "sma_50",
    "ma_12", "ma_26", "ma_50", "price_to_sma_5", "price_to_sma_20", "price_to_sma_50",
    "volatility_5", "volatility_10", "volatility_20", "volume_sma_5", "relative_volume",
    "high_low_range", "open_close_change", "overnight_gap",
    "macd", "macd_signal", "macd_hist", "bb_position", "rsi_7", "rsi_14", "rsi_21",
    "momentum_3", "momentum_5", "momentum_10", "momentum_20", "atr_14_pct",
    "day_of_week", "month",
]
TARGET_COL = "target_next_close"
CLASS_TARGET_COL = "target_direction"

INFERENCE_LOG_META_COLS = [
    "symbol", "trade_ts", "prediction", "actual_next_close",
    "request_ts", "model_version", "model_alias", "run_source",
]


# --------------------------------------------------------------------------------------
# Config loading
# --------------------------------------------------------------------------------------
def _conf_dir():
    here = os.getcwd()
    candidates = [os.path.join(here, "..", "conf"), os.path.join(here, "conf")]
    try:  # when imported as a regular module (tests)
        candidates.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "conf"))
    except NameError:
        pass
    for c in candidates:
        if os.path.isdir(c):
            return os.path.abspath(c)
    raise FileNotFoundError(f"conf/ directory not found; looked in {candidates}")


def load_yaml(name):
    import yaml
    with open(os.path.join(_conf_dir(), name)) as f:
        return yaml.safe_load(f)


def load_project():
    return load_yaml("project.yml")


def load_model_params():
    return load_yaml("model_params.yml")


def table(cfg, key, catalog):
    """Fully qualified table name: <catalog>.<schema>.<table> from the `tables` section."""
    return f"{catalog}.{cfg['tables'][key]}"


def model_name(cfg, catalog):
    return f"{catalog}.{cfg['model']['registered_name']}"


def classifier_model_name(cfg, catalog):
    """Fully qualified Unity Catalog name for the direction classifier."""
    return f"{catalog}.{cfg['model']['registered_classifier_name']}"


def widget_or(value, default):
    """Job widgets arrive as strings; an empty string means 'use the project.yml default'."""
    return default if value is None or str(value).strip() == "" else value


def as_bool(x):
    return str(x).strip().lower() in ("1", "true", "yes", "y")


# --------------------------------------------------------------------------------------
# Price / return maths
# --------------------------------------------------------------------------------------
def calculate_returns(prices):
    prices = np.asarray(prices, dtype=float)
    out = np.full(prices.shape, np.nan, dtype=float)
    out[1:] = prices[1:] / prices[:-1] - 1.0
    return out


# --------------------------------------------------------------------------------------
# Chronological split (no shuffling, split on distinct timestamps so symbols never straddle a cut)
# --------------------------------------------------------------------------------------
def chronological_cuts(unique_ts, validation_fraction=0.15, test_fraction=0.15):
    """Return (cut_fit_end, cut_val_end): fit <= cut1 < val <= cut2 < test."""
    ts = np.sort(np.asarray(unique_ts))
    if len(ts) < 30:
        raise ValueError("Need at least 30 distinct timestamps for chronological train/validation/test.")
    if validation_fraction <= 0 or test_fraction <= 0 or validation_fraction + test_fraction >= 0.9:
        raise ValueError("validation_fraction and test_fraction must be > 0 and sum to < 0.9")
    fit_frac = 1.0 - validation_fraction - test_fraction
    i1 = max(int(len(ts) * fit_frac) - 1, 0)
    i2 = max(int(len(ts) * (fit_frac + validation_fraction)) - 1, i1 + 1)
    return ts[i1], ts[i2]


def recent_window_cutoff(ts_counts, max_rows):
    """Earliest timestamp to KEEP so that the most recent whole timestamps total <= max_rows.

    ts_counts: iterable of (timestamp, row_count) pairs (one per distinct timestamp, any order).
    Returns None when everything already fits. Cuts only on whole timestamps, so no symbol ever
    straddles the cut and the time order is preserved. If even the newest timestamp alone exceeds
    max_rows it is still kept (a window can never be empty).
    """
    pairs = sorted(((t, int(c)) for t, c in ts_counts), key=lambda p: p[0], reverse=True)
    if not pairs:
        raise ValueError("No timestamps supplied.")
    if sum(c for _, c in pairs) <= max_rows:
        return None
    total, keep_from = 0, pairs[0][0]
    for t, c in pairs:
        if total + c > max_rows:
            break
        total += c
        keep_from = t
    return keep_from


# --------------------------------------------------------------------------------------
# Regression metrics
# --------------------------------------------------------------------------------------
def directional_accuracy(y_true, y_pred, ref_close):
    """Share of rows where the predicted move (vs today's close) has the same sign as the real move."""
    y_true, y_pred, ref = (np.asarray(a, dtype=float) for a in (y_true, y_pred, ref_close))
    return float(np.mean((y_pred > ref) == (y_true > ref)))


def rmse(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def regression_scores(y_true, y_pred, ref_close):
    """MAE / RMSE / R2 / directional accuracy / persistence-baseline RMSE for one split."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    persistence = rmse(y_true, ref_close)          # "tomorrow's close = today's close"
    model_rmse = rmse(y_true, y_pred)
    return {
        "mae": float(np.mean(np.abs(y_true - y_pred))),
        "rmse": model_rmse,
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "directional_accuracy": directional_accuracy(y_true, y_pred, ref_close),
        "persistence_rmse": persistence,
        "rmse_vs_persistence": model_rmse / persistence if persistence > 0 else float("nan"),
    }


# --------------------------------------------------------------------------------------
# Drift statistics
# --------------------------------------------------------------------------------------
def population_stability_index(reference, current, bins=10):
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    ref = ref[np.isfinite(ref)]
    cur = cur[np.isfinite(cur)]
    if len(ref) < 2 or len(cur) < 2:
        raise ValueError("PSI requires at least two finite observations in each sample.")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    p = np.histogram(ref, bins=edges)[0] / len(ref)
    q = np.histogram(cur, bins=edges)[0] / len(cur)
    p, q = np.clip(p, 1e-6, None), np.clip(q, 1e-6, None)
    return float(np.sum((q - p) * np.log(q / p)))


def psi_level(value, moderate=0.10, high=0.25):
    return "HIGH" if value >= high else ("MODERATE" if value >= moderate else "LOW")


def ks_statistic(reference, current):
    """Two-sample Kolmogorov-Smirnov statistic (max distance between the empirical CDFs)."""
    a = np.sort(np.asarray(reference, dtype=float))
    b = np.sort(np.asarray(current, dtype=float))
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        raise ValueError("KS requires non-empty samples.")
    grid = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, grid, side="right") / len(a)
    cdf_b = np.searchsorted(b, grid, side="right") / len(b)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def ensure_notebook_path():
    """Make `from utils import ...` work even if the runtime did not add the notebook dir to sys.path."""
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
