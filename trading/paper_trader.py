"""
paper_trader.py
---------------
Virtual paper trading engine for CryptoPulse.

Simulates trades based on ML model signals with a virtual cash balance.
Tracks positions, P&L, trade history, and portfolio value over time.
Persists state to PostgreSQL so the dashboard can read it live.

Usage:
    python paper_trader.py --run          # run one trading cycle
    python paper_trader.py --backtest     # backtest on historical data
    python paper_trader.py --status       # print portfolio status
    python paper_trader.py --reset        # reset portfolio to initial state
"""

import argparse
import logging
import os
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from features import compute_features, get_feature_columns, load_price_history
from model import load_artifacts, predict_latest

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from ml.features import compute_features, get_feature_columns, load_price_history
from ml.model import load_artifacts, predict_latest

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("cryptopulse.trader")

# ── Config ─────────────────────────────────────────────────────────────────────

STARTING_CASH      = 10_000.0   # USD
TRADE_FRACTION     = 0.20       # invest 20% of available cash per BUY signal
MAX_POSITION_COINS = 3          # max simultaneous open positions
STOP_LOSS_PCT      = 0.05       # exit if position drops 5%
TAKE_PROFIT_PCT    = 0.10       # exit if position gains 10%
CONFIDENCE_MIN     = 0.60       # minimum model confidence to act on signal
MAKER_FEE          = 0.001      # 0.1% simulated trading fee per side

SYMBOLS = ["BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "DOGE", "AVAX", "LINK", "DOT"]

# ── DB Schema ──────────────────────────────────────────────────────────────────

def init_tables(engine: Engine):
    ddl = """
    -- Portfolio state (one row per session, updated in-place)
    CREATE TABLE IF NOT EXISTS paper_portfolio (
        id           SERIAL PRIMARY KEY,
        cash         DOUBLE PRECISION NOT NULL DEFAULT 10000,
        total_value  DOUBLE PRECISION,
        total_pnl    DOUBLE PRECISION,
        total_pnl_pct DOUBLE PRECISION,
        updated_at   TIMESTAMP WITH TIME ZONE DEFAULT NOW()
    );

    -- Open positions
    CREATE TABLE IF NOT EXISTS paper_positions (
        id           SERIAL PRIMARY KEY,
        symbol       TEXT NOT NULL UNIQUE,
        quantity     DOUBLE PRECISION NOT NULL,
        entry_price  DOUBLE PRECISION NOT NULL,
        entry_cost   DOUBLE PRECISION NOT NULL,
        opened_at    TIMESTAMP WITH TIME ZONE DEFAULT NOW()
    );

    -- Trade history (closed trades)
    CREATE TABLE IF NOT EXISTS paper_trades (
        id           SERIAL PRIMARY KEY,
        symbol       TEXT NOT NULL,
        side         TEXT NOT NULL,       -- BUY or SELL
        quantity     DOUBLE PRECISION,
        price        DOUBLE PRECISION,
        cost         DOUBLE PRECISION,
        fee          DOUBLE PRECISION,
        pnl          DOUBLE PRECISION,    -- NULL for buys
        pnl_pct      DOUBLE PRECISION,    -- NULL for buys
        signal       TEXT,               -- model signal that triggered
        confidence   DOUBLE PRECISION,
        executed_at  TIMESTAMP WITH TIME ZONE DEFAULT NOW()
    );

    -- Portfolio value over time (for chart)
    CREATE TABLE IF NOT EXISTS paper_portfolio_history (
        id           SERIAL PRIMARY KEY,
        total_value  DOUBLE PRECISION,
        cash         DOUBLE PRECISION,
        positions_value DOUBLE PRECISION,
        recorded_at  TIMESTAMP WITH TIME ZONE DEFAULT NOW()
    );
    """
    with engine.begin() as conn:
        conn.execute(text(ddl))
    logger.info("Paper trading tables ready")

# ── Portfolio State ─────────────────────────────────────────────────────────────

class Portfolio:
    def __init__(self, engine: Engine):
        self.engine = engine
        self._load()

    def _load(self):
        """Load portfolio state from DB."""
        with self.engine.connect() as conn:
            # Cash balance
            row = conn.execute(text(
                "SELECT cash FROM paper_portfolio ORDER BY id DESC LIMIT 1"
            )).fetchone()
            self.cash = row[0] if row else STARTING_CASH

            # Open positions
            rows = conn.execute(text(
                "SELECT symbol, quantity, entry_price, entry_cost, opened_at FROM paper_positions"
            )).fetchall()
            self.positions = {
                r[0]: {
                    "symbol":     r[0],
                    "quantity":   r[1],
                    "entry_price": r[2],
                    "entry_cost": r[3],
                    "opened_at":  r[4],
                }
                for r in rows
            }

    def _save(self, total_value: float):
        """Persist portfolio state to DB."""
        positions_value = total_value - self.cash
        with self.engine.begin() as conn:
            # Upsert portfolio row
            conn.execute(text("""
                INSERT INTO paper_portfolio (cash, total_value, total_pnl, total_pnl_pct, updated_at)
                VALUES (:cash, :tv, :pnl, :pnl_pct, NOW())
            """), {
                "cash":    self.cash,
                "tv":      total_value,
                "pnl":     total_value - STARTING_CASH,
                "pnl_pct": (total_value - STARTING_CASH) / STARTING_CASH * 100,
            })
            # History snapshot
            conn.execute(text("""
                INSERT INTO paper_portfolio_history (total_value, cash, positions_value, recorded_at)
                VALUES (:tv, :cash, :pv, NOW())
            """), {"tv": total_value, "cash": self.cash, "pv": positions_value})

    def value(self, current_prices: dict) -> float:
        """Calculate total portfolio value given current prices."""
        positions_val = sum(
            pos["quantity"] * current_prices.get(sym, pos["entry_price"])
            for sym, pos in self.positions.items()
        )
        return self.cash + positions_val

    def buy(self, symbol: str, price: float, signal: str, confidence: float) -> Optional[dict]:
        """
        Execute a BUY if conditions are met:
        - Signal is BUY with sufficient confidence
        - Not already holding this symbol
        - Have available cash
        - Haven't exceeded max positions
        """
        if symbol in self.positions:
            logger.debug("%s: already holding, skip BUY", symbol)
            return None
        if len(self.positions) >= MAX_POSITION_COINS:
            logger.debug("Max positions (%d) reached, skip BUY %s", MAX_POSITION_COINS, symbol)
            return None
        if confidence < CONFIDENCE_MIN:
            logger.debug("%s: confidence %.2f below threshold, skip", symbol, confidence)
            return None

        invest = min(self.cash * TRADE_FRACTION, self.cash)
        if invest < 1.0:
            logger.info("Insufficient cash ($%.2f) to buy %s", self.cash, symbol)
            return None

        fee      = invest * MAKER_FEE
        cost     = invest - fee
        quantity = cost / price

        self.cash -= invest
        self.positions[symbol] = {
            "symbol":      symbol,
            "quantity":    quantity,
            "entry_price": price,
            "entry_cost":  cost,
            "opened_at":   datetime.now(tz=timezone.utc),
        }

        trade = {
            "symbol":      symbol,
            "side":        "BUY",
            "quantity":    quantity,
            "price":       price,
            "cost":        invest,
            "fee":         fee,
            "pnl":         None,
            "pnl_pct":     None,
            "signal":      signal,
            "confidence":  confidence,
        }
        self._record_trade(trade)
        self._sync_position(symbol, upsert=True)
        logger.info("BUY  %-6s | qty=%.6f @ $%.2f | cost=$%.2f | cash_left=$%.2f",
                    symbol, quantity, price, invest, self.cash)
        return trade

    def sell(self, symbol: str, price: float, signal: str, confidence: float, reason: str = "SIGNAL") -> Optional[dict]:
        """Execute a SELL for an open position."""
        if symbol not in self.positions:
            return None

        pos      = self.positions[symbol]
        quantity = pos["quantity"]
        proceeds = quantity * price
        fee      = proceeds * MAKER_FEE
        net      = proceeds - fee
        pnl      = net - pos["entry_cost"]
        pnl_pct  = pnl / pos["entry_cost"] * 100

        self.cash += net
        del self.positions[symbol]

        trade = {
            "symbol":     symbol,
            "side":       "SELL",
            "quantity":   quantity,
            "price":      price,
            "cost":       proceeds,
            "fee":        fee,
            "pnl":        pnl,
            "pnl_pct":    pnl_pct,
            "signal":     f"{signal} ({reason})",
            "confidence": confidence,
        }
        self._record_trade(trade)
        self._sync_position(symbol, upsert=False)
        logger.info("SELL %-6s | qty=%.6f @ $%.2f | P&L=$%.2f (%.2f%%) | [%s]",
                    symbol, quantity, price, pnl, pnl_pct, reason)
        return trade

    def _record_trade(self, trade: dict):
        with self.engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO paper_trades
                    (symbol, side, quantity, price, cost, fee, pnl, pnl_pct, signal, confidence, executed_at)
                VALUES
                    (:symbol, :side, :quantity, :price, :cost, :fee, :pnl, :pnl_pct, :signal, :confidence, NOW())
            """), trade)

    def _sync_position(self, symbol: str, upsert: bool):
        with self.engine.begin() as conn:
            if upsert and symbol in self.positions:
                pos = self.positions[symbol]
                conn.execute(text("""
                    INSERT INTO paper_positions (symbol, quantity, entry_price, entry_cost, opened_at)
                    VALUES (:symbol, :quantity, :entry_price, :entry_cost, :opened_at)
                    ON CONFLICT (symbol) DO UPDATE
                        SET quantity=EXCLUDED.quantity, entry_price=EXCLUDED.entry_price,
                            entry_cost=EXCLUDED.entry_cost
                """), pos)
            else:
                conn.execute(text("DELETE FROM paper_positions WHERE symbol=:symbol"), {"symbol": symbol})

    def status(self, current_prices: dict) -> pd.DataFrame:
        """Return positions DataFrame with current values and unrealised P&L."""
        rows = []
        for sym, pos in self.positions.items():
            cp   = current_prices.get(sym, pos["entry_price"])
            val  = pos["quantity"] * cp
            pnl  = val - pos["entry_cost"]
            rows.append({
                "Symbol":      sym,
                "Qty":         round(pos["quantity"], 6),
                "Entry $":     round(pos["entry_price"], 2),
                "Current $":   round(cp, 2),
                "Value $":     round(val, 2),
                "P&L $":       round(pnl, 2),
                "P&L %":       round(pnl / pos["entry_cost"] * 100, 2),
            })
        return pd.DataFrame(rows)

# ── Trading Cycle ──────────────────────────────────────────────────────────────

def run_cycle(engine: Engine, portfolio: Portfolio):
    """
    One full trading cycle:
    1. Get ML prediction for each symbol
    2. Check stop-loss / take-profit on open positions
    3. Execute BUY/SELL signals
    4. Save portfolio state
    """
    current_prices = {}
    predictions    = []

    logger.info("─" * 50)
    logger.info("Running trading cycle: %d symbols", len(SYMBOLS))

    for sym in SYMBOLS:
        try:
            pred = predict_latest(sym, engine)
            predictions.append(pred)
            current_prices[sym] = pred["price"]
        except FileNotFoundError:
            logger.warning("No model for %s — skipping", sym)
        except Exception as e:
            logger.error("Prediction error for %s: %s", sym, e)

    # 1. Check stop-loss / take-profit on existing positions first
    for sym, pos in list(portfolio.positions.items()):
        cp    = current_prices.get(sym)
        if not cp:
            continue
        pnl_pct = (cp - pos["entry_price"]) / pos["entry_price"]

        if pnl_pct <= -STOP_LOSS_PCT:
            portfolio.sell(sym, cp, "STOP_LOSS", 1.0, reason="STOP-LOSS")
        elif pnl_pct >= TAKE_PROFIT_PCT:
            portfolio.sell(sym, cp, "TAKE_PROFIT", 1.0, reason="TAKE-PROFIT")

    # 2. Act on model signals
    for pred in predictions:
        sym        = pred["symbol"]
        price      = pred["price"]
        signal     = pred["signal"]
        confidence = pred["confidence"]

        if signal == "BUY":
            portfolio.buy(sym, price, signal, confidence)
        elif signal == "SELL" and sym in portfolio.positions:
            portfolio.sell(sym, price, signal, confidence, reason="SIGNAL")

    # 3. Save and log summary
    total_value = portfolio.value(current_prices)
    portfolio._save(total_value)

    pnl     = total_value - STARTING_CASH
    pnl_pct = pnl / STARTING_CASH * 100

    logger.info("─" * 50)
    logger.info("Portfolio Value: $%.2f | Cash: $%.2f | P&L: $%.2f (%.2f%%)",
                total_value, portfolio.cash, pnl, pnl_pct)
    logger.info("Open Positions: %d", len(portfolio.positions))

    status = portfolio.status(current_prices)
    if not status.empty:
        logger.info("\n%s", status.to_string(index=False))

    return predictions, total_value

# ── Backtest ───────────────────────────────────────────────────────────────────

def backtest(symbol: str, engine: Engine) -> pd.DataFrame:
    """
    Walk-forward backtest on historical data for a single symbol.
    Simulates trading signals chronologically and tracks virtual P&L.
    """
    logger.info("Backtesting %s...", symbol)

    try:
        model, scaler = load_artifacts(symbol)
    except FileNotFoundError:
        logger.error("Train a model for %s first: python model.py --symbol %s --train", symbol, symbol)
        return pd.DataFrame()

    feature_cols = get_feature_columns()
    df_raw  = load_price_history(engine, symbol=symbol, limit=5000)
    df_feat = compute_features(df_raw)

    if len(df_feat) < 100:
        logger.error("Not enough data for backtest")
        return pd.DataFrame()

    # Walk-forward: start from 60% of data, simulate on remainder
    start_idx = int(len(df_feat) * 0.6)

    cash     = STARTING_CASH
    position = None   # {"qty": float, "entry": float, "cost": float}
    trades   = []

    X = scaler.transform(df_feat[feature_cols].values)

    for i in range(start_idx, len(df_feat) - 1):
        row   = df_feat.iloc[i]
        price = row["current_price"]
        prob_up = float(model.predict_proba(X[i:i+1])[0, 1])

        if prob_up >= CONFIDENCE_MIN and position is None:
            # BUY
            invest   = cash * TRADE_FRACTION
            fee      = invest * MAKER_FEE
            cost     = invest - fee
            qty      = cost / price
            cash    -= invest
            position = {"qty": qty, "entry": price, "cost": cost}

        elif (1 - prob_up) >= CONFIDENCE_MIN and position is not None:
            # SELL
            proceeds = position["qty"] * price
            fee      = proceeds * MAKER_FEE
            net      = proceeds - fee
            pnl      = net - position["cost"]
            cash    += net
            trades.append({
                "timestamp": row["fetched_at"],
                "price":     price,
                "pnl":       pnl,
                "pnl_pct":   pnl / position["cost"] * 100,
                "cash":      cash,
            })
            position = None

        # stop-loss check
        if position and (price - position["entry"]) / position["entry"] <= -STOP_LOSS_PCT:
            proceeds = position["qty"] * price
            fee      = proceeds * MAKER_FEE
            net      = proceeds - fee
            pnl      = net - position["cost"]
            cash    += net
            trades.append({
                "timestamp": row["fetched_at"],
                "price":     price,
                "pnl":       pnl,
                "pnl_pct":   pnl / position["cost"] * 100,
                "cash":      cash,
                "note":      "STOP-LOSS",
            })
            position = None

    results = pd.DataFrame(trades)
    if not results.empty:
        total_pnl  = results["pnl"].sum()
        win_rate   = (results["pnl"] > 0).mean() * 100
        final_val  = cash + (position["qty"] * df_feat["current_price"].iloc[-1] if position else 0)
        logger.info("═" * 50)
        logger.info("BACKTEST RESULTS — %s", symbol)
        logger.info("Trades:       %d", len(results))
        logger.info("Win rate:     %.1f%%", win_rate)
        logger.info("Total P&L:    $%.2f", total_pnl)
        logger.info("Final value:  $%.2f (started $%.2f)", final_val, STARTING_CASH)
        logger.info("Return:       %.2f%%", (final_val - STARTING_CASH) / STARTING_CASH * 100)
        logger.info("═" * 50)
    else:
        logger.info("No trades executed in backtest")

    return results

# ── Portfolio Reset ────────────────────────────────────────────────────────────

def reset_portfolio(engine: Engine):
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM paper_portfolio"))
        conn.execute(text("DELETE FROM paper_positions"))
        conn.execute(text("DELETE FROM paper_trades"))
        conn.execute(text("DELETE FROM paper_portfolio_history"))
        conn.execute(text(
            "INSERT INTO paper_portfolio (cash, total_value, total_pnl, total_pnl_pct) "
            f"VALUES ({STARTING_CASH}, {STARTING_CASH}, 0, 0)"
        ))
    logger.info("Portfolio reset to $%.2f", STARTING_CASH)

# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CryptoPulse Paper Trader")
    parser.add_argument("--run",      action="store_true", help="Run one trading cycle")
    parser.add_argument("--backtest", action="store_true", help="Backtest on historical data")
    parser.add_argument("--status",   action="store_true", help="Print portfolio status")
    parser.add_argument("--reset",    action="store_true", help="Reset portfolio")
    parser.add_argument("--symbol",   default="BTC",       help="Symbol for backtest")
    args = parser.parse_args()

    engine = create_engine(
        f"postgresql+psycopg2://{os.getenv('PGUSER')}:{os.getenv('PGPASSWORD')}"
        f"@{os.getenv('PGHOST','localhost')}:{os.getenv('PGPORT',5432)}/{os.getenv('PGDATABASE')}",
        future=True
    )
    init_tables(engine)

    if args.reset:
        reset_portfolio(engine)

    elif args.run:
        portfolio = Portfolio(engine)
        run_cycle(engine, portfolio)

    elif args.backtest:
        backtest(args.symbol.upper(), engine)

    elif args.status:
        portfolio = Portfolio(engine)
        logger.info("Cash: $%.2f | Positions: %d", portfolio.cash, len(portfolio.positions))
        if portfolio.positions:
            print(portfolio.status({}).to_string(index=False))