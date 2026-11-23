import numpy as np
import pytest
from utils import (FEATURE_COLS, as_bool, chronological_cuts, directional_accuracy, ks_statistic, psi_level,
                   recent_window_cutoff, regression_scores, widget_or)


def test_feature_cols_unique_and_exclude_labels():
    assert len(FEATURE_COLS) == len(set(FEATURE_COLS))
    assert not any(c.startswith("target") for c in FEATURE_COLS)


def test_chronological_cuts_are_ordered_and_leak_free():
    ts = np.arange(100)
    c1, c2 = chronological_cuts(ts, 0.15, 0.15)
    fit, val, test = ts[ts <= c1], ts[(ts > c1) & (ts <= c2)], ts[ts > c2]
    assert min(len(fit), len(val), len(test)) > 0
    assert fit.max() < val.min() and val.max() < test.min()
    assert abs(len(fit) / 100 - 0.70) < 0.03 and abs(len(test) / 100 - 0.15) < 0.03


def test_chronological_cuts_unsorted_input_and_too_short():
    c1, c2 = chronological_cuts(np.random.default_rng(0).permutation(60))
    assert c1 < c2
    with pytest.raises(ValueError):
        chronological_cuts(np.arange(10))


def test_directional_accuracy():
    close = np.array([10.0, 10.0, 10.0, 10.0])
    actual = np.array([11.0, 9.0, 11.0, 9.0])
    pred = np.array([12.0, 8.0, 9.0, 11.0])
    assert directional_accuracy(actual, pred, close) == 0.5


def test_regression_scores_persistence_baseline():
    close = np.array([10.0, 11.0, 12.0])
    actual = np.array([11.0, 12.0, 13.0])
    s = regression_scores(actual, actual, close)         # perfect model
    assert s["rmse"] == 0.0 and s["mae"] == 0.0 and s["r2"] == 1.0
    assert np.isclose(s["persistence_rmse"], 1.0)         # "tomorrow = today" is off by 1 each day
    assert s["rmse_vs_persistence"] == 0.0


def test_ks_statistic():
    rng = np.random.default_rng(1)
    a = rng.normal(size=3000)
    assert ks_statistic(a, a) == 0.0
    assert ks_statistic(a, a + 3) > 0.8


def test_psi_level_bands():
    assert psi_level(0.05) == "LOW" and psi_level(0.15) == "MODERATE" and psi_level(0.30) == "HIGH"


def test_widget_helpers():
    assert widget_or("", 5) == 5 and widget_or("7", 5) == "7" and widget_or(None, 5) == 5
    assert as_bool("True") and as_bool("1") and not as_bool("false")


def test_recent_window_cutoff_keeps_whole_recent_timestamps():
    counts = [(d, 100) for d in range(10)]                       # 10 days x 100 rows
    assert recent_window_cutoff(counts, 2000) is None            # everything fits
    assert recent_window_cutoff(counts, 350) == 7                # newest 3 days (300 rows), never a partial day
    assert recent_window_cutoff(reversed(counts), 350) == 7      # input order does not matter


def test_recent_window_cutoff_uneven_days_and_oversized_newest_day():
    counts = [(1, 500), (2, 100), (3, 100)]
    assert recent_window_cutoff(counts, 250) == 2                # days 3+2 = 200 rows; adding day 1 would be 700
    assert recent_window_cutoff([(1, 10), (2, 900)], 100) == 2   # newest day alone exceeds cap: still kept
    with pytest.raises(ValueError):
        recent_window_cutoff([], 10)
