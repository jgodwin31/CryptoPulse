"""
ingest_binance.py
-----------------
Real-time cryptocurrency price ingestion via Binance WebSocket.
Streams live ticker data for configured symbols and writes to PostgreSQL.

Usage:
    python ingest_binance.py

Runs indefinitely. Press Ctrl+C to stop.
Designed to run alongside (or replace) run_pipeline.py for real-time ingestion.
"""

import asyncio
import json
import logging
import os
import signal
import time
from datetime import datetime, timezone
from typing import List

import pandas as pd
import websockets
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s"
)
logger = logging.getLogger("cryptopulse.binance")

# ── Config ────────────────────────────────────────────────────────────────────

# Top coins to track — Binance symbol format (always USDT pairs)
SYMBOLS: List[str] = [
    "btcusdt", "ethusdt", "bnbusdt", "solusdt", "xrpusdt",
    "adausdt", "dogeusdt", "avaxusdt", "linkusdt", "dotusdt"
]

# Binance combined stream endpoint
# Subscribes to individual symbol mini-tickers (24hr rolling stats + last price)
BINANCE_WS_BASE = "wss://stream.binance.com:9443/stream?streams="

# How many rows to buffer before bulk-inserting into Postgres
BATCH_SIZE = 20

# Reconnect delay on connection drop (seconds)
RECONNECT_DELAY = 5

TABLE_NAME = "crypto_market_rt"  # separate table from batch pipeline

# ── Database ──────────────────────────────────────────────────────────────────

def get_engine() -> Engine:
    url = (
        f"postgresql+psycopg2://{os.getenv('PGUSER')}:{os.getenv('PGPASSWORD')}"
        f"@{os.getenv('PGHOST', 'localhost')}:{os.getenv('PGPORT', 5432)}"
        f"/{os.getenv('PGDATABASE')}"
    )
    return create_engine(url, future=True, pool_pre_ping=True)


def init_schema(engine: Engine):
    """Create real-time table if not exists."""
    ddl = f"""
    CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
        id              BIGSERIAL PRIMARY KEY,
        symbol          TEXT NOT NULL,
        coin_id         TEXT NOT NULL,
        fetched_at      TIMESTAMP WITH TIME ZONE NOT NULL,
        current_price   DOUBLE PRECISION,
        price_change_pct DOUBLE PRECISION,
        high_24h        DOUBLE PRECISION,
        low_24h         DOUBLE PRECISION,
        total_volume    DOUBLE PRECISION,
        market_cap      DOUBLE PRECISION DEFAULT NULL,
        volatility_index DOUBLE PRECISION DEFAULT NULL,
        last_updated    TIMESTAMP WITH TIME ZONE,
        UNIQUE(symbol, fetched_at)
    );
    CREATE INDEX IF NOT EXISTS idx_rt_symbol     ON {TABLE_NAME} (symbol);
    CREATE INDEX IF NOT EXISTS idx_rt_fetched_at ON {TABLE_NAME} (fetched_at DESC);
    """
    with engine.begin() as conn:
        conn.execute(text(ddl))
    logger.info("Schema ready: %s", TABLE_NAME)


def bulk_insert(engine: Engine, rows: List[dict]):
    """Bulk insert buffered rows. Ignores duplicates via ON CONFLICT DO NOTHING."""
    if not rows:
        return
    df = pd.DataFrame(rows)
    # Use raw SQL for upsert — pandas to_sql doesn't support ON CONFLICT
    insert_sql = f"""
        INSERT INTO {TABLE_NAME}
            (symbol, coin_id, fetched_at, current_price, price_change_pct,
             high_24h, low_24h, total_volume, last_updated)
        VALUES
            (:symbol, :coin_id, :fetched_at, :current_price, :price_change_pct,
             :high_24h, :low_24h, :total_volume, :last_updated)
        ON CONFLICT (symbol, fetched_at) DO NOTHING
    """
    with engine.begin() as conn:
        conn.execute(text(insert_sql), rows)
    logger.info("Inserted %d rows into %s", len(rows), TABLE_NAME)

# ── WebSocket Stream ──────────────────────────────────────────────────────────

def build_stream_url(symbols: List[str]) -> str:
    """
    Build Binance combined stream URL.
    Uses miniTicker streams — lightweight, updates every second per symbol.
    """
    streams = "/".join(f"{s}@miniTicker" for s in symbols)
    return BINANCE_WS_BASE + streams


def parse_mini_ticker(data: dict) -> dict | None:
    """
    Parse a Binance miniTicker message into a DB row dict.

    Binance miniTicker fields:
        e  - event type ("24hrMiniTicker")
        E  - event time (ms timestamp)
        s  - symbol (e.g. "BTCUSDT")
        c  - close price (current price)
        o  - open price
        h  - high price
        l  - low price
        v  - base asset volume
        q  - quote asset volume (USD volume for USDT pairs)
    """
    try:
        stream_data = data.get("data", data)  # handle combined stream wrapper
        if stream_data.get("e") != "24hrMiniTicker":
            return None

        symbol_raw = stream_data["s"]  # e.g. "BTCUSDT"
        symbol = symbol_raw.replace("USDT", "")  # e.g. "BTC"
        close_price = float(stream_data["c"])
        open_price  = float(stream_data["o"])
        high        = float(stream_data["h"])
        low         = float(stream_data["l"])
        volume      = float(stream_data["q"])  # quote volume = USD volume
        event_time  = datetime.fromtimestamp(stream_data["E"] / 1000, tz=timezone.utc)

        # price change % vs open
        price_change_pct = ((close_price - open_price) / open_price * 100) if open_price else None

        return {
            "symbol":           symbol,
            "coin_id":          symbol.lower(),
            "fetched_at":       datetime.now(tz=timezone.utc),
            "current_price":    close_price,
            "price_change_pct": round(price_change_pct, 4) if price_change_pct else None,
            "high_24h":         high,
            "low_24h":          low,
            "total_volume":     volume,
            "last_updated":     event_time,
        }
    except (KeyError, ValueError, TypeError) as e:
        logger.warning("Failed to parse ticker: %s | raw: %s", e, data)
        return None


async def stream(engine: Engine, symbols: List[str]):
    """
    Main WebSocket coroutine. Connects to Binance, parses tickers,
    buffers rows, and bulk-inserts every BATCH_SIZE messages.
    Reconnects automatically on disconnect.
    """
    url = build_stream_url(symbols)
    buffer: List[dict] = []

    logger.info("Connecting to Binance WebSocket: %d symbols", len(symbols))
    logger.info("Tracking: %s", [s.upper() for s in symbols])

    while True:  # outer reconnect loop
        try:
            async with websockets.connect(
                url,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=5,
            ) as ws:
                logger.info("WebSocket connected ✓")
                async for raw_msg in ws:
                    try:
                        data = json.loads(raw_msg)
                    except json.JSONDecodeError:
                        logger.warning("Non-JSON message received, skipping")
                        continue

                    row = parse_mini_ticker(data)
                    if row:
                        buffer.append(row)
                        logger.debug(
                            "%-6s $%-12.4f  Δ24h: %+.2f%%",
                            row["symbol"], row["current_price"],
                            row["price_change_pct"] or 0
                        )

                    # flush buffer to DB
                    if len(buffer) >= BATCH_SIZE:
                        try:
                            bulk_insert(engine, buffer)
                        except Exception as e:
                            logger.error("DB insert failed: %s", e)
                        finally:
                            buffer.clear()

        except websockets.ConnectionClosed as e:
            logger.warning("WebSocket closed (%s). Reconnecting in %ds...", e, RECONNECT_DELAY)
        except OSError as e:
            logger.error("Network error: %s. Reconnecting in %ds...", e, RECONNECT_DELAY)
        except Exception as e:
            logger.exception("Unexpected error: %s. Reconnecting in %ds...", e, RECONNECT_DELAY)

        # flush remaining buffer before reconnect
        if buffer:
            try:
                bulk_insert(engine, buffer)
            except Exception:
                pass
            buffer.clear()

        await asyncio.sleep(RECONNECT_DELAY)

# ── Graceful Shutdown ─────────────────────────────────────────────────────────

_shutdown = False

def _handle_signal(sig, frame):
    global _shutdown
    logger.info("Shutdown signal received (%s). Stopping...", signal.Signals(sig).name)
    _shutdown = True

# ── Entry Point ───────────────────────────────────────────────────────────────

async def main():
    signal.signal(signal.SIGINT,  _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    engine = get_engine()
    init_schema(engine)

    try:
        await stream(engine, SYMBOLS)
    except asyncio.CancelledError:
        pass
    finally:
        logger.info("CryptoPulse Binance ingestion stopped.")
        engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())