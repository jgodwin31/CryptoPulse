"""
ingest_kraken.py
----------------
Real-time cryptocurrency price ingestion via Kraken WebSocket.
Streams live ticker data for configured symbols and writes to PostgreSQL.

Kraken WebSocket v2 docs: https://docs.kraken.com/api/docs/websocket-v2/ticker

Usage:
    python -m ingestion.ingest_kraken

Runs indefinitely. Press Ctrl+C to stop.
"""

import asyncio
import json
import logging
import os
import signal
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
logger = logging.getLogger("cryptopulse.kraken")

# ── Config ─────────────────────────────────────────────────────────────────────

# Kraken uses format "BTC/USD" for pairs
SYMBOLS_KRAKEN = [
    "BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD", "ADA/USD",
    "DOGE/USD", "AVAX/USD", "LINK/USD", "DOT/USD", "MATIC/USD"
]

# Clean symbol map: "BTC/USD" → "BTC"
def clean_symbol(pair: str) -> str:
    return pair.split("/")[0]

KRAKEN_WS_URL  = "wss://ws.kraken.com/v2"
BATCH_SIZE     = 20
RECONNECT_DELAY = 5
TABLE_NAME     = "crypto_market_rt"

# ── Database ───────────────────────────────────────────────────────────────────

def get_engine() -> Engine:
    url = (
        f"postgresql+psycopg2://{os.getenv('PGUSER')}:{os.getenv('PGPASSWORD')}"
        f"@{os.getenv('PGHOST', 'localhost')}:{os.getenv('PGPORT', 5432)}"
        f"/{os.getenv('PGDATABASE')}"
    )
    return create_engine(url, future=True, pool_pre_ping=True)


def init_schema(engine: Engine):
    ddl = f"""
    CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
        id               BIGSERIAL PRIMARY KEY,
        symbol           TEXT NOT NULL,
        coin_id          TEXT NOT NULL,
        fetched_at       TIMESTAMP WITH TIME ZONE NOT NULL,
        current_price    DOUBLE PRECISION,
        price_change_pct DOUBLE PRECISION,
        high_24h         DOUBLE PRECISION,
        low_24h          DOUBLE PRECISION,
        total_volume     DOUBLE PRECISION,
        market_cap       DOUBLE PRECISION DEFAULT NULL,
        volatility_index DOUBLE PRECISION DEFAULT NULL,
        last_updated     TIMESTAMP WITH TIME ZONE,
        UNIQUE(symbol, fetched_at)
    );
    CREATE INDEX IF NOT EXISTS idx_rt_symbol     ON {TABLE_NAME} (symbol);
    CREATE INDEX IF NOT EXISTS idx_rt_fetched_at ON {TABLE_NAME} (fetched_at DESC);
    """
    with engine.begin() as conn:
        conn.execute(text(ddl))
    logger.info("Schema ready: %s", TABLE_NAME)


def bulk_insert(engine: Engine, rows: List[dict]):
    if not rows:
        return
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

# ── WebSocket ──────────────────────────────────────────────────────────────────

def build_subscribe_msg(symbols: List[str]) -> dict:
    """
    Kraken v2 subscription message for ticker channel.
    """
    return {
        "method": "subscribe",
        "params": {
            "channel": "ticker",
            "symbol":  symbols,
        }
    }


def parse_ticker(msg: dict) -> List[dict]:
    """
    Parse a Kraken v2 ticker message into DB row dicts.

    Kraken v2 ticker fields (inside data array):
        symbol      - e.g. "BTC/USD"
        last         - last trade price
        high         - 24h high
        low          - 24h low
        volume       - 24h volume (base currency)
        vwap         - 24h volume-weighted average price
        change       - price change vs 24h ago
        change_pct   - % price change vs 24h ago
    """
    rows = []
    try:
        if msg.get("channel") != "ticker":
            return rows
        if msg.get("type") not in ("snapshot", "update"):
            return rows

        for item in msg.get("data", []):
            symbol_raw = item.get("symbol", "")        # e.g. "BTC/USD"
            symbol     = clean_symbol(symbol_raw)       # e.g. "BTC"
            price      = item.get("last")
            high       = item.get("high")
            low        = item.get("low")
            volume     = item.get("volume")
            change_pct = item.get("change_pct")

            if price is None:
                continue

            rows.append({
                "symbol":           symbol,
                "coin_id":          symbol.lower(),
                "fetched_at":       datetime.now(tz=timezone.utc),
                "current_price":    float(price),
                "price_change_pct": round(float(change_pct), 4) if change_pct is not None else None,
                "high_24h":         float(high)   if high   is not None else None,
                "low_24h":          float(low)    if low    is not None else None,
                "total_volume":     float(volume) if volume is not None else None,
                "last_updated":     datetime.now(tz=timezone.utc),
            })
    except (KeyError, ValueError, TypeError) as e:
        logger.warning("Failed to parse ticker: %s | raw: %s", e, msg)

    return rows


async def stream(engine: Engine, symbols: List[str]):
    """
    Main WebSocket coroutine. Connects to Kraken, subscribes to ticker,
    buffers rows, and bulk-inserts every BATCH_SIZE messages.
    Auto-reconnects on disconnect.
    """
    subscribe_msg = build_subscribe_msg(symbols)
    buffer: List[dict] = []

    logger.info("Connecting to Kraken WebSocket...")
    logger.info("Tracking: %s", [clean_symbol(s) for s in symbols])

    while True:
        try:
            async with websockets.connect(
                KRAKEN_WS_URL,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=5,
            ) as ws:
                logger.info("WebSocket connected ✓")

                # Send subscription
                await ws.send(json.dumps(subscribe_msg))
                logger.info("Subscribed to ticker channel for %d pairs", len(symbols))

                async for raw_msg in ws:
                    try:
                        data = json.loads(raw_msg)
                    except json.JSONDecodeError:
                        continue

                    # Log subscription confirmations
                    if data.get("method") == "subscribe":
                        if data.get("success"):
                            logger.info("Subscription confirmed ✓")
                        else:
                            logger.warning("Subscription failed: %s", data)
                        continue

                    # Parse ticker updates
                    rows = parse_ticker(data)
                    for row in rows:
                        buffer.append(row)
                        logger.debug(
                            "%-8s $%-12.4f  Δ24h: %+.2f%%",
                            row["symbol"],
                            row["current_price"],
                            row["price_change_pct"] or 0
                        )

                    # Flush to DB when buffer is full
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

        # Flush remaining buffer before reconnect
        if buffer:
            try:
                bulk_insert(engine, buffer)
            except Exception:
                pass
            buffer.clear()

        await asyncio.sleep(RECONNECT_DELAY)

# ── Graceful Shutdown ──────────────────────────────────────────────────────────

def _handle_signal(sig, frame):
    logger.info("Shutdown signal received. Stopping...")
    raise SystemExit(0)

# ── Entry Point ────────────────────────────────────────────────────────────────

async def main():
    signal.signal(signal.SIGINT,  _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    engine = get_engine()
    init_schema(engine)

    try:
        await stream(engine, SYMBOLS_KRAKEN)
    except (asyncio.CancelledError, SystemExit):
        pass
    finally:
        logger.info("CryptoPulse Kraken ingestion stopped.")
        engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())