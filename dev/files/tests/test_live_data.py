from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import pytest
from live_data import (bar_cutoff_date, build_symbol_map, compute_start_date, normalize_download_frame,
                       price_basis_mismatch, to_provider_ticker)


def _ohlcv(dates, base=100.0):
    n = len(dates)
    return pd.DataFrame({"Open": base + np.arange(n), "High": base + 2 + np.arange(n), "Low": base - 2 + np.arange(n),
                         "Close": base + 1 + np.arange(n), "Adj Close": base + np.arange(n), "Volume": 1000 + np.arange(n)},
                        index=pd.DatetimeIndex(dates, name="Date"))


DATES = pd.to_datetime(["2026-02-02", "2026-02-03", "2026-02-04"])


def test_ticker_mapping_roundtrip():
    assert to_provider_ticker(" brk.b ") == "BRK-B"
    assert to_provider_ticker("reliance.ns") == "RELIANCE.NS"
    assert to_provider_ticker("RPPL.BO") == "RPPL.BO"
    assert build_symbol_map(["BRK.B", "AAPL"]) == {"BRK-B": "BRK.B", "AAPL": "AAPL"}


def test_normalize_ticker_first_layout_and_symbol_restored():
    raw = pd.concat({"AAPL": _ohlcv(DATES), "BRK-B": _ohlcv(DATES, 300.0)}, axis=1)     # (ticker, field)
    out = normalize_download_frame(raw, build_symbol_map(["AAPL", "BRK.B"]))
    assert list(out.columns) == ["symbol", "timestamp", "open", "high", "low", "close", "volume"]
    assert set(out.symbol) == {"AAPL", "BRK.B"} and len(out) == 6
    assert out.volume.dtype == np.int64 and (out.timestamp == out.timestamp.dt.normalize()).all()


def test_normalize_field_first_layout_and_single_ticker():
    raw = pd.concat({"AAPL": _ohlcv(DATES)}, axis=1).swaplevel(0, 1, axis=1)              # (field, ticker)
    assert len(normalize_download_frame(raw, {"AAPL": "AAPL"})) == 3
    assert len(normalize_download_frame(_ohlcv(DATES), {"AAPL": "AAPL"})) == 3             # flat single ticker
    with pytest.raises(ValueError):
        normalize_download_frame(_ohlcv(DATES), {"A": "A", "B": "B"})


def test_normalize_drops_bad_rows_and_failed_tickers():
    good, bad = _ohlcv(DATES).astype(float), _ohlcv(DATES).astype(float)
    bad.iloc[:, :] = np.nan                                  # a ticker that returned nothing
    good.loc[DATES[1], "Close"] = -1.0                       # invalid price
    good.loc[DATES[2], "Volume"] = np.nan                    # missing volume
    raw = pd.concat({"AAPL": good, "ZZZZ": bad}, axis=1)
    out = normalize_download_frame(raw, {"AAPL": "AAPL", "ZZZZ": "ZZZZ"})
    assert list(out.symbol) == ["AAPL"] and len(out) == 1


def test_normalize_empty_and_tz_aware_index():
    assert normalize_download_frame(pd.DataFrame(), {}).empty
    tz = _ohlcv(DATES)
    tz.index = tz.index.tz_localize("America/New_York")
    out = normalize_download_frame(tz, {"AAPL": "AAPL"})
    assert out.timestamp.dt.tz is None and out.timestamp.iloc[0] == pd.Timestamp("2026-02-02")


def test_bar_cutoff_drops_a_still_forming_bar():
    midday = datetime(2026, 2, 4, 15, 0, tzinfo=timezone.utc)      # market open
    evening = datetime(2026, 2, 4, 23, 30, tzinfo=timezone.utc)    # market closed
    assert bar_cutoff_date(midday) == date(2026, 2, 4)             # today's bar excluded (date >= cutoff dropped)
    assert bar_cutoff_date(evening) == date(2026, 2, 5)            # today's bar kept
    with pytest.raises(ValueError):
        bar_cutoff_date(datetime(2026, 2, 4, 12, 0))


def test_compute_start_date():
    today = date(2026, 11, 11)
    assert compute_start_date([date(2026, 1, 21), date(2026, 1, 23)], 5, 1100, today) == date(2026, 1, 16)
    assert compute_start_date([date(2020, 1, 1)], 5, 100, today) == date(2026, 8, 3)        # capped by lookback
    assert compute_start_date([date(2026, 1, 21), None], 5, 100, today) == date(2026, 8, 3)  # unseen symbol
    assert compute_start_date([], 5, 100, today) == date(2026, 8, 3)


def test_price_basis_mismatch():
    assert not price_basis_mismatch([0.0001, -0.0002, 0.0, 0.5], 0.02)      # one outlier tolerated (median)
    assert price_basis_mismatch([0.05, 0.06, 0.04], 0.02)                   # systematic gap
    assert not price_basis_mismatch([], 0.02)
