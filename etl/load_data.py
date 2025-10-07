import os
import logging
from typing import Optional
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)

DEFAULT_TABLE = "crypto_market"

def get_db_engine(user: str, password: str, host: str, port: int, dbname: str, echo: bool = False) -> Engine:
    """
    Return SQLAlchemy engine. driver uses psycopg2.
    """
    url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}"
    engine = create_engine(url, echo=echo, future=True)
    return engine

def init_schema(engine: Engine, table_name: str = DEFAULT_TABLE):
    """
    Create table if it doesn't exist. Simple schema to record latest snapshot per fetch.
    """
    create_sql = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
      id BIGSERIAL PRIMARY KEY,
      coin_id TEXT NOT NULL,
      symbol TEXT NOT NULL,
      name TEXT,
      fetched_at TIMESTAMP WITH TIME ZONE NOT NULL,
      current_price DOUBLE PRECISION,
      market_cap DOUBLE PRECISION,
      total_volume DOUBLE PRECISION,
      price_change_pct DOUBLE PRECISION,
      volatility_index DOUBLE PRECISION,
      last_updated TIMESTAMP WITH TIME ZONE,
      UNIQUE(coin_id, fetched_at)
    );
    CREATE INDEX IF NOT EXISTS idx_{table_name}_symbol ON {table_name} (symbol);
    CREATE INDEX IF NOT EXISTS idx_{table_name}_fetched_at ON {table_name} (fetched_at);
    """
    with engine.begin() as conn:
        conn.execute(text(create_sql))
    logger.info("Schema initialized (table: %s)", table_name)

def insert_market_snapshot(engine: Engine, df: pd.DataFrame, table_name: str = DEFAULT_TABLE):
    """
    Insert DataFrame rows into Postgres table.
    Uses df.to_sql for simplicity (upsert not implemented; duplicates avoided by unique constraint on coin_id+fetched_at).
    """
    if df.empty:
        logger.info("No rows to insert")
        return

    # map columns to db schema
    to_insert = df[[
        "id", "symbol", "name", "fetched_at",
        "current_price", "market_cap", "total_volume",
        "price_change_pct", "volatility_index", "last_updated"
    ]].rename(columns={"id": "coin_id"})

    try:
        # use pandas to_sql (SQLAlchemy engine)
        to_insert.to_sql(table_name, engine, if_exists="append", index=False, method="multi")
        logger.info("Inserted %d rows into %s", len(to_insert), table_name)
    except SQLAlchemyError as e:
        logger.exception("Failed to insert rows: %s", e)
        # You might want to implement row-by-row fallback or upsert behavior here.

if __name__ == "__main__":
    # manual quick test (requires DB)
    import dotenv, os
    dotenv.load_dotenv()
    engine = get_db_engine(
        os.getenv("PGUSER"), os.getenv("PGPASSWORD"),
        os.getenv("PGHOST", "localhost"), int(os.getenv("PGPORT", 5432)),
        os.getenv("PGDATABASE")
    )
    init_schema(engine)
