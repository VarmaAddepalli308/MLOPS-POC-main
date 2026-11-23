"""Spark-free helpers for the live data download (importable from notebooks and unit tests).

Only end-of-day bars are used: the model predicts the next day's close from daily bars, so no streaming
data is needed.
"""
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

OHLCV_FIELDS = ["Open", "High", "Low", "Close", "Volume"]
OUT_COLUMNS = ["symbol", "timestamp", "open", "high", "low", "close", "volume"]

# US cash markets close at 16:00 New York time = 20:00 UTC (summer) / 21:00 UTC (winter).
# After 21:30 UTC the day's bar is complete in both cases. (No timezone database needed.)
_MARKET_DONE_UTC = (21, 30)


# Yahoo exchange suffixes that must keep their dot (Indian NSE / BSE): RELIANCE.NS stays RELIANCE.NS
EXCHANGE_SUFFIXES = (".NS", ".BO")


def to_provider_ticker(symbol):
    """Dataset symbol -> Yahoo ticker. US class shares use '-' (BRK.B -> BRK-B); exchange suffixes keep the dot."""
    s = str(symbol).strip().upper()
    if s.endswith(EXCHANGE_SUFFIXES):
        return s
    return s.replace(".", "-")


def build_symbol_map(symbols):
    """{provider ticker: original dataset symbol}, so downloaded rows are stored under the dataset's own names."""
    return {to_provider_ticker(s): str(s) for s in symbols}


def bar_cutoff_date(now_utc):
    """Bars dated >= the returned date are dropped as possibly incomplete.

    Before the market is surely closed (21:30 UTC) today's bar may still be forming, so today is excluded.
    After that today's bar is complete and kept.
    """
    if now_utc.tzinfo is None:
        raise ValueError("now_utc must be timezone-aware")
    now_utc = now_utc.astimezone(timezone.utc)
    today = now_utc.date()
    return today + timedelta(days=1) if (now_utc.hour, now_utc.minute) >= _MARKET_DONE_UTC else today


def compute_start_date(last_dates, overlap_days, max_lookback_days, today):
    """First date to download. last_dates: latest stored date per symbol (None = symbol not stored yet)."""
    last_dates = list(last_dates)
    floor = today - timedelta(days=int(max_lookback_days))
    if not last_dates or any(d is None for d in last_dates):
        return floor     # some symbol has no stored history yet: fetch the full lookback (the merge de-duplicates)
    return max(min(last_dates) - timedelta(days=int(overlap_days)), floor)


def _one_ticker(sub, symbol):
    sub = sub.copy()
    sub.columns = [str(c).strip().title() for c in sub.columns]
    missing = [c for c in OHLCV_FIELDS if c not in sub.columns]
    if missing:
        raise ValueError(f"Provider frame for {symbol} lacks columns {missing}; got {list(sub.columns)}")
    idx = pd.DatetimeIndex(sub.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    out = pd.DataFrame({
        "symbol": symbol,
        "timestamp": idx.normalize(),
        "open": pd.to_numeric(sub["Open"], errors="coerce").values,
        "high": pd.to_numeric(sub["High"], errors="coerce").values,
        "low": pd.to_numeric(sub["Low"], errors="coerce").values,
        "close": pd.to_numeric(sub["Close"], errors="coerce").values,
        "volume": pd.to_numeric(sub["Volume"], errors="coerce").values,
    })
    out = out.dropna(subset=["open", "high", "low", "close", "volume"])
    out = out[(out.open > 0) & (out.high > 0) & (out.low > 0) & (out.close > 0) & (out.volume >= 0)]
    out["volume"] = out["volume"].astype("int64")
    return out


def normalize_download_frame(raw, symbol_map):
    """Turn a yfinance.download() result into a tidy long frame (OUT_COLUMNS), one row per symbol-day.

    Handles (ticker, field) and (field, ticker) MultiIndex layouts and the single-ticker flat layout.
    Tickers that returned no usable data simply produce no rows (the caller reports them).
    """
    empty = pd.DataFrame({c: pd.Series(dtype=t) for c, t in zip(
        OUT_COLUMNS, ["object", "datetime64[ns]", "float64", "float64", "float64", "float64", "int64"])})
    if raw is None or len(raw) == 0:
        return empty

    df = raw.copy()
    frames = []
    if isinstance(df.columns, pd.MultiIndex):
        level0 = {str(v).strip().title() for v in df.columns.get_level_values(0)}
        if level0 & set(OHLCV_FIELDS):                       # (field, ticker) layout -> (ticker, field)
            df.columns = df.columns.swaplevel(0, 1)
        for tkr in dict.fromkeys(df.columns.get_level_values(0)):
            symbol = symbol_map.get(str(tkr).upper(), str(tkr))
            frames.append(_one_ticker(df[tkr], symbol))
    else:
        if len(symbol_map) != 1:
            raise ValueError("Flat provider frame is only valid for a single ticker.")
        frames.append(_one_ticker(df, next(iter(symbol_map.values()))))

    out = pd.concat(frames, ignore_index=True) if frames else empty
    out = out.drop_duplicates(subset=["symbol", "timestamp"], keep="last")
    return out.sort_values(["symbol", "timestamp"]).reset_index(drop=True)[OUT_COLUMNS]


def price_basis_mismatch(relative_deviations, tolerance=0.02):
    """True when stored and downloaded close prices disagree systematically on overlapping days.

    Typical causes: stored data is split/dividend-adjusted but the download is raw (or vice versa).
    Uses the median absolute deviation so a few genuine corrections do not trigger it.
    """
    dev = np.asarray(list(relative_deviations), dtype=float)
    dev = dev[np.isfinite(dev)]
    if len(dev) == 0:
        return False
    return float(np.median(np.abs(dev))) > float(tolerance)


def utc_now():
    return datetime.now(timezone.utc)
