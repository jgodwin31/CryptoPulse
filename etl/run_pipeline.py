"""
Orchestrator to run the full ETL pipeline: extract -> transform -> compute volatility -> load
This script is intended to be run periodically (cron, systemd timer, or Airflow task).
"""

import logging
import os
import time
from typing import Optional
import pandas as pd
import dotenv

from etl.extract_data import fetch_top_coins
from etl.transform_data import market_list_to_df, compute_volatility_from_series
from etl.load_data import get_db_engine, init_schema, insert_market_snapshot, DEFAULT_TABLE

logger = logging.getLogger(__name__)

dotenv.load_dotenv()  # loads .env

def compute_volatility_for_df(engine, df: pd.DataFrame, window: int = 24) -> pd.DataFrame:
    """
    For each coin row in df, fetch recent historical price points from DB and compute volatility,
    then set volatility_index column.
    """
    if df.empty:
        return df
    symbols = df["symbol"].unique().tolist()
    vols = {}
    with engine.connect() as conn:
        for sym in symbols:
            # fetch last (window * 3) price snapshots; adjust as needed
            q = f"""
                SELECT current_price, fetched_at
                FROM {DEFAULT_TABLE}
                WHERE symbol = :sym
                ORDER BY fetched_at ASC
                LIMIT :limit
            """
            # we want oldest to newest, but we limit to recent N (so we do a subquery if needed)
            # simpler: fetch last N and then sort
            recent = conn.execute(
                f"""
                SELECT current_price, fetched_at FROM {DEFAULT_TABLE}
                WHERE symbol = :sym
                ORDER BY fetched_at DESC
                LIMIT :limit
                """, {"sym": sym, "limit": max(window*3, 100)}
            ).fetchall()
            # recent is newest->oldest; convert to pandas Series oldest->newest
            if not recent:
                vols[sym] = None
                continue
            prices = pd.Series([r[0] for r in reversed(recent)])
            vol = compute_volatility_from_series(prices, window=window)
            vols[sym] = vol
    # map vols into df
    df["volatility_index"] = df["symbol"].map(vols)
    return df

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger.info("Starting CryptoPulse ETL run")

    # DB config from env
    pg_user = os.getenv("PGUSER")
    pg_pass = os.getenv("PGPASSWORD")
    pg_host = os.getenv("PGHOST", "localhost")
    pg_port = int(os.getenv("PGPORT", 5432))
    pg_db = os.getenv("PGDATABASE")

    if not (pg_user and pg_pass and pg_db):
        logger.error("Missing Postgres credentials in environment. See .env.example")
        return

    engine = get_db_engine(pg_user, pg_pass, pg_host, pg_port, pg_db)

    # ensure schema exists
    init_schema(engine)

    # 1. Extract
    try:
        raw = fetch_top_coins(per_page=10)
    except Exception as e:
        logger.exception("Extraction failed: %s", e)
        return

    # 2. Transform
    df = market_list_to_df(raw)

    # 3. Volatility computation (bonus)
    try:
        df = compute_volatility_for_df(engine, df, window=24)
    except Exception as e:
        logger.exception("Volatility computation failed: %s", e)
        # continue without volatility

    # 4. Load
    try:
        insert_market_snapshot(engine, df)
    except Exception as e:
        logger.exception("Load failed: %s", e)
        return

    logger.info("ETL run completed successfully")

if __name__ == "__main__":
    main()
