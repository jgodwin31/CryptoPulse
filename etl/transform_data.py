import pandas as pd
import logging
from typing import List, Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

def market_list_to_df(market_list: List[Dict]) -> pd.DataFrame:
    """
    Convert CoinGecko market list to a cleaned DataFrame with derived metrics.
    Expects the API response items to include:
      id, symbol, name, current_price, market_cap, total_volume, price_change_percentage_24h, last_updated
    """
    df = pd.json_normalize(market_list)
    # keep only fields we need
    keep = [
        "id", "symbol", "name",
        "current_price", "market_cap", "total_volume",
        "price_change_percentage_24h", "last_updated"
    ]
    for col in keep:
        if col not in df.columns:
            df[col] = None
    df = df[keep].copy()
    # type conversions
    df["current_price"] = pd.to_numeric(df["current_price"], errors="coerce")
    df["market_cap"] = pd.to_numeric(df["market_cap"], errors="coerce")
    df["total_volume"] = pd.to_numeric(df["total_volume"], errors="coerce")
    df["price_change_percentage_24h"] = pd.to_numeric(df["price_change_percentage_24h"], errors="coerce")
    # convert last_updated to timestamp
    df["last_updated"] = pd.to_datetime(df["last_updated"], errors="coerce")
    # derive timestamp for ingestion
    df["fetched_at"] = pd.Timestamp.utcnow()
    # add a simple short name (uppercase symbol)
    df["symbol"] = df["symbol"].str.upper()
    # compute a derived price_change (alternate if API doesn't provide it)
    df["price_change_pct"] = df["price_change_percentage_24h"]
    # add a placeholder for volatility (populated later using historical data)
    df["volatility_index"] = pd.NA
    logger.debug("Transformed market list into DataFrame with %d rows", len(df))
    return df

def compute_volatility_from_series(price_series: pd.Series, window: int = 24) -> Optional[float]:
    """
    Compute a rolling volatility index (stddev of price changes) for the latest window.
    price_series is ordered from oldest to newest.
    Returns the latest rolling std (or None if insufficient data).
    """
    if price_series is None or len(price_series.dropna()) < 2:
        return None
    # compute percent returns
    returns = price_series.pct_change().dropna()
    if len(returns) < 1:
        return None
    # rolling std of percent returns; use last window
    if len(returns) < window:
        # compute std over available data
        return float(returns.std())
    else:
        return float(returns.rolling(window=window).std().iloc[-1])

if __name__ == "__main__":
    # quick test
    import etl.extract_data as ex
    logging.basicConfig(level=logging.INFO)
    coins = ex.fetch_top_coins( per_page=5 )
    df = market_list_to_df(coins)
    print(df.head().to_string())
