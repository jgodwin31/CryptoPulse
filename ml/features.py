"""
features.py
-----------
Feature engineering for CryptoPulse ML pipeline.
Reads price history from PostgreSQL and computes technical indicators
used as inputs to the XGBoost prediction model.

Features computed:
    - Moving averages (7, 14, 30 periods)
    - Exponential moving averages (12, 26 periods)
    - MACD and MACD signal line
    - RSI (14 period)
    - Bollinger Bands (upper, lower, width)
    - Volume ratio (current vs rolling average)
    - Price momentum (rate of change)
    - Volatility (rolling std of returns)
    - Target label: price direction (1 = up, 0 = down) N periods ahead
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

logger = logging.getLogger("cryptopulse.features")

# ── Constants ─────────────────────────────────────────────────────────────────

TABLE_RT    = "crypto_market_rt"
TABLE_BATCH = "crypto_market"

# Minimum rows needed to compute all features reliably
MIN_ROWS = 50

# Default prediction horizon: how many periods ahead to predict direction
PREDICTION_HORIZON = 5

# ── Data Loading ──────────────────────────────────────────────────────────────

def load_price_history(
    engine: Engine,
    symbol: str,
    limit: int = 2000,
    table: str = TABLE_RT
) -> pd.DataFrame:
    """
    Load price history for a symbol from PostgreSQL, ordered oldest to newest.
    Falls back to batch table if real-time table is empty for this symbol.
    """
    query = text(f"""
        SELECT fetched_at, current_price, total_volume, price_change_pct, high_24h, low_24h
        FROM {table}
        WHERE symbol = :symbol
        ORDER BY fetched_at ASC
        LIMIT :limit
    """)
    with engine.connect() as conn:
        df = pd.read_sql_query(query, conn, params={"symbol": symbol, "limit": limit})

    if df.empty and table == TABLE_RT:
        logger.warning("No real-time data for %s, falling back to batch table", symbol)
        return load_price_history(engine, symbol, limit=limit, table=TABLE_BATCH)

    df["fetched_at"] = pd.to_datetime(df["fetched_at"])
    df = df.sort_values("fetched_at").reset_index(drop=True)
    logger.info("Loaded %d rows for %s", len(df), symbol)
    return df

# ── Technical Indicators ──────────────────────────────────────────────────────

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Compute Relative Strength Index.
    RSI > 70 → overbought (sell signal), RSI < 30 → oversold (buy signal).
    """
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)

    # Use exponential weighted mean for smoothing (Wilder's method)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """
    Compute MACD line, signal line, and histogram.
    Returns (macd, signal_line, histogram) as Series.
    """
    ema_fast   = series.ewm(span=fast,   adjust=False).mean()
    ema_slow   = series.ewm(span=slow,   adjust=False).mean()
    macd       = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram  = macd - signal_line
    return macd, signal_line, histogram


def compute_bollinger_bands(series: pd.Series, window: int = 20, num_std: float = 2.0):
    """
    Compute Bollinger Bands.
    Returns (upper_band, lower_band, bandwidth) as Series.
    Bandwidth = (upper - lower) / middle — measures volatility.
    """
    middle = series.rolling(window=window).mean()
    std    = series.rolling(window=window).std()
    upper  = middle + num_std * std
    lower  = middle - num_std * std
    bandwidth = (upper - lower) / middle.replace(0, np.nan)
    return upper, lower, bandwidth


def compute_features(df: pd.DataFrame, horizon: int = PREDICTION_HORIZON) -> pd.DataFrame:
    """
    Given a price history DataFrame, compute all ML features and the target label.

    Input df columns expected: fetched_at, current_price, total_volume, high_24h, low_24h
    Returns a new DataFrame with all features + target column, NaN rows dropped.
    """
    if len(df) < MIN_ROWS:
        raise ValueError(
            f"Need at least {MIN_ROWS} rows to compute features, got {len(df)}. "
            "Run the ingestion pipeline longer to collect more data."
        )

    f = df.copy()
    price  = f["current_price"]
    volume = f["total_volume"]

    # ── Moving Averages ───────────────────────────────────────────────────────
    f["ma_7"]  = price.rolling(7).mean()
    f["ma_14"] = price.rolling(14).mean()
    f["ma_30"] = price.rolling(30).mean()

    # Price relative to MA (normalised distance)
    f["price_to_ma7"]  = price / f["ma_7"]  - 1
    f["price_to_ma14"] = price / f["ma_14"] - 1
    f["price_to_ma30"] = price / f["ma_30"] - 1

    # ── Exponential Moving Averages ───────────────────────────────────────────
    f["ema_12"] = price.ewm(span=12, adjust=False).mean()
    f["ema_26"] = price.ewm(span=26, adjust=False).mean()
    f["ema_cross"] = f["ema_12"] - f["ema_26"]  # golden/death cross indicator

    # ── MACD ─────────────────────────────────────────────────────────────────
    f["macd"], f["macd_signal"], f["macd_hist"] = compute_macd(price)

    # ── RSI ───────────────────────────────────────────────────────────────────
    f["rsi_14"] = compute_rsi(price, period=14)
    f["rsi_7"]  = compute_rsi(price, period=7)

    # ── Bollinger Bands ───────────────────────────────────────────────────────
    f["bb_upper"], f["bb_lower"], f["bb_width"] = compute_bollinger_bands(price)
    f["bb_position"] = (price - f["bb_lower"]) / (f["bb_upper"] - f["bb_lower"] + 1e-10)
    # bb_position: 0 = at lower band, 1 = at upper band

    # ── Volume Features ───────────────────────────────────────────────────────
    f["volume_ma14"]  = volume.rolling(14).mean()
    f["volume_ratio"] = volume / f["volume_ma14"].replace(0, np.nan)
    # volume spike = volume_ratio > 2 (often precedes large price move)

    # ── Momentum / Rate of Change ─────────────────────────────────────────────
    f["roc_1"]  = price.pct_change(1)   # 1-period return
    f["roc_5"]  = price.pct_change(5)   # 5-period momentum
    f["roc_14"] = price.pct_change(14)  # 14-period momentum

    # ── Volatility ────────────────────────────────────────────────────────────
    returns = price.pct_change()
    f["volatility_7"]  = returns.rolling(7).std()
    f["volatility_14"] = returns.rolling(14).std()

    # ── High-Low Range ────────────────────────────────────────────────────────
    if "high_24h" in f.columns and "low_24h" in f.columns:
        f["hl_range"] = (f["high_24h"] - f["low_24h"]) / price.replace(0, np.nan)
    else:
        f["hl_range"] = np.nan

    # ── Target Label ─────────────────────────────────────────────────────────
    # 1 if price is higher N periods from now, 0 if lower/same
    f["target"] = (price.shift(-horizon) > price).astype(int)

    # ── Drop NaN rows (edges from rolling windows + target lookahead) ─────────
    feature_cols = [
        "ma_7", "ma_14", "ma_30",
        "price_to_ma7", "price_to_ma14", "price_to_ma30",
        "ema_12", "ema_26", "ema_cross",
        "macd", "macd_signal", "macd_hist",
        "rsi_14", "rsi_7",
        "bb_upper", "bb_lower", "bb_width", "bb_position",
        "volume_ratio",
        "roc_1", "roc_5", "roc_14",
        "volatility_7", "volatility_14",
        "hl_range",
        "target"
    ]
    f = f.dropna(subset=feature_cols)

    logger.info(
        "Feature matrix: %d rows, %d features (horizon=%d periods)",
        len(f), len(feature_cols) - 1, horizon
    )
    return f


def get_feature_columns() -> list:
    """Return the list of feature column names (excludes target)."""
    return [
        "ma_7", "ma_14", "ma_30",
        "price_to_ma7", "price_to_ma14", "price_to_ma30",
        "ema_12", "ema_26", "ema_cross",
        "macd", "macd_signal", "macd_hist",
        "rsi_14", "rsi_7",
        "bb_upper", "bb_lower", "bb_width", "bb_position",
        "volume_ratio",
        "roc_1", "roc_5", "roc_14",
        "volatility_7", "volatility_14",
        "hl_range",
    ]


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    from sqlalchemy import create_engine

    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    engine = create_engine(
        f"postgresql+psycopg2://{os.getenv('PGUSER')}:{os.getenv('PGPASSWORD')}"
        f"@{os.getenv('PGHOST','localhost')}:{os.getenv('PGPORT',5432)}/{os.getenv('PGDATABASE')}",
        future=True
    )

    df_raw = load_price_history(engine, symbol="BTC")
    df_feat = compute_features(df_raw)
    print(df_feat[get_feature_columns() + ["target"]].tail(10).to_string())