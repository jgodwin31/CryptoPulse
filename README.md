# ⚡ CryptoPulse

> Real-time cryptocurrency intelligence platform with ML-powered price direction prediction and paper trading simulation.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15+-336791)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Overview

CryptoPulse is an end-to-end data engineering and ML pipeline that ingests real-time cryptocurrency market data, engineers technical indicators, trains an XGBoost price direction classifier, and executes a paper trading simulation — all visualised in a live Streamlit dashboard.

Built as a portfolio project demonstrating:
- Real-time streaming data ingestion via WebSocket
- Production-grade ETL pipeline with idempotent schema design
- Feature engineering for time-series financial data
- Machine learning with proper time-series cross-validation
- Automated paper trading with risk management

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                               │
│                                                                 │
│  Kraken WebSocket ──► ingest_kraken.py ──► PostgreSQL           │
│  (real-time ticks)     (batch insert,        (crypto_market_rt) │
│                         auto-reconnect)                         │
│                                                                 │
│  CoinGecko API ──────► etl/extract_data.py ► PostgreSQL         │
│  (batch fallback)       transform_data.py    (crypto_market)    │
│                         load_data.py                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        ML LAYER                                 │
│                                                                 │
│  features.py ──► 25 technical indicators                        │
│    • Moving averages (MA7, MA14, MA30)                          │
│    • Exponential MAs (EMA12, EMA26)                             │
│    • MACD + signal line + histogram                             │
│    • RSI (7-period, 14-period)                                  │
│    • Bollinger Bands (upper, lower, width, position)            │
│    • Volume ratio, momentum (ROC1, ROC5, ROC14)                 │
│    • Volatility (7-period, 14-period rolling std)               │
│    • High-low range                                             │
│                                                                 │
│  model.py ──► XGBoost binary classifier                         │
│    • Target: price direction N periods ahead                    │
│    • Time-series cross-validation (no data leakage)             │
│    • Early stopping to prevent overfitting                      │
│    • BTC AUC: 0.669 | ETH: 0.609 | SOL: 0.658                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     TRADING LAYER                               │
│                                                                 │
│  paper_trader.py                                                │
│    • $10,000 virtual starting balance                           │
│    • 20% position sizing per BUY signal                         │
│    • Max 3 simultaneous open positions                          │
│    • 5% stop-loss / 10% take-profit                             │
│    • 0.1% simulated maker fee per side                          │
│    • BUY/SELL/HOLD signals at ≥60% model confidence            │
│    • Full trade history persisted to PostgreSQL                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     DASHBOARD LAYER                             │
│                                                                 │
│  dashboard/dashboard.py (Streamlit)                             │
│    • Live price table (Kraken WebSocket feed)                   │
│    • ML signal panel — BUY/SELL/HOLD with confidence %          │
│    • Portfolio overview — value, cash, P&L                      │
│    • Portfolio value chart vs $10k baseline                     │
│    • Open positions with unrealised P&L                         │
│    • Trade history with signal attribution                      │
│    • Auto-refresh every 30 seconds                              │
└─────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.10 |
| Database | PostgreSQL 15 |
| ORM / DB client | SQLAlchemy 2.0, psycopg2 |
| Real-time ingestion | Kraken WebSocket v2, websockets |
| Data processing | pandas, numpy |
| Feature engineering | Custom (RSI, MACD, Bollinger Bands) |
| ML model | XGBoost, scikit-learn |
| Dashboard | Streamlit, Plotly |
| Config | python-dotenv |

---

## Project Structure

```
CryptoPulse/
├── .env                        # DB credentials (not committed)
├── .env.example                # Template for credentials
├── requirements.txt
├── README.md
│
├── etl/                        # Batch pipeline (CoinGecko)
│   ├── __init__.py
│   ├── extract_data.py         # API client with retry/backoff
│   ├── transform_data.py       # Cleaning, type casting, volatility
│   ├── load_data.py            # PostgreSQL schema + batch insert
│   └── run_pipeline.py         # Orchestrator (cron-compatible)
│
├── ingestion/                  # Real-time streaming
│   └── ingest_kraken.py        # Kraken WebSocket → PostgreSQL
│
├── ml/                         # Machine learning
│   ├── __init__.py
│   ├── features.py             # 25 technical indicator features
│   ├── model.py                # XGBoost trainer + predictor
│   └── models/                 # Saved .pkl artifacts (gitignored)
│
├── trading/                    # Paper trading engine
│   └── paper_trader.py         # Signal execution + P&L tracking
│
├── dashboard/                  # Visualisation
│   └── dashboard.py            # Streamlit live dashboard
│
└── sql/
    └── init_schema.sql         # Base schema definitions
```

---

## Getting Started

### Prerequisites
- Python 3.10+
- PostgreSQL 15+
- Git

### Installation

```bash
git clone https://github.com/jgodwin31/CryptoPulse.git
cd CryptoPulse
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### Configuration

```bash
cp .env.example .env
```

Edit `.env`:
```
PGUSER=your_postgres_user
PGPASSWORD=your_postgres_password
PGHOST=localhost
PGPORT=5432
PGDATABASE=cryptopulse
```

### Running the Pipeline

**Step 1 — Start real-time ingestion (leave running):**
```bash
python -m ingestion.ingest_kraken
```

**Step 2 — Train ML models (after ~30 min of data collection):**
```bash
python -m ml.model --symbol BTC --train
python -m ml.model --symbol ETH --train
python -m ml.model --symbol SOL --train
```

**Step 3 — Start paper trading (run every 5 min):**
```bash
python -m trading.paper_trader --reset
# Windows loop:
while ($true) { python -m trading.paper_trader --run; Start-Sleep -Seconds 300 }
```

**Step 4 — Launch dashboard:**
```bash
streamlit run dashboard/dashboard.py
```

Open `http://localhost:8501`

---

## ML Model Performance

Models trained on ~15 hours of tick data with 5-fold time-series cross-validation:

| Symbol | Rows | CV AUC | Test AUC | Best Iter |
|--------|------|--------|----------|-----------|
| BTC    | 20,903 | 0.654 ±0.024 | **0.669** | 43 |
| ETH    | 10,124 | 0.616 ±0.086 | **0.609** | 42 |
| SOL    | 7,764  | 0.562 ±0.013 | **0.658** | 37 |
| XRP    | 7,143  | 0.529 ±0.046 | 0.546 | 16 |

> AUC of 0.5 = random. AUC of 0.7+ = strong signal. BTC and SOL test AUC above 0.65 is meaningful for short-horizon crypto prediction.

**Top predictive features (BTC):** `price_to_ma7`, `roc_1`, `roc_5`, `volume_ratio`, `volatility_7`

Short-term momentum and price-relative-to-MA dominate — the model is picking up mean-reversion and momentum patterns at the tick level.

---

## Key Engineering Decisions

**Idempotent ingestion:** `UNIQUE(symbol, fetched_at)` constraint with `ON CONFLICT DO NOTHING` ensures duplicate ticks never corrupt the dataset regardless of reconnects.

**Time-series cross-validation:** Standard k-fold CV would leak future data into training folds. `TimeSeriesSplit` respects temporal order — each fold trains on past data only and validates on future data, matching real-world deployment conditions.

**Early stopping:** XGBoost trains until validation logloss stops improving for 30 consecutive rounds, preventing overfitting to a single market session. Best iterations ranged from 16–43 rounds, far below the 500 estimator ceiling.

**Separate real-time table:** `crypto_market_rt` (WebSocket ticks) is kept separate from `crypto_market` (CoinGecko batch). This allows the ML pipeline to use high-frequency tick data while the batch pipeline continues to run independently.

**Position sizing:** 20% of available cash per signal with a 3-position cap limits maximum drawdown to ~60% even if all positions hit stop-loss simultaneously.

---

## Roadmap

- [ ] Retrain models on 7+ days of data for stronger generalisation
- [ ] Add AVAX, LINK, DOT models as data accumulates
- [ ] Sentiment feature layer (news/social signal integration)
- [ ] Backtesting module with Sharpe ratio and max drawdown metrics
- [ ] Model drift detection — auto-retrain when AUC degrades
- [ ] Deploy to cloud (AWS EC2 + RDS) for 24/7 operation

---

## Author

**Joshua Godwin**
[Portfolio](https://joshuagodwin.vercel.app) · [LinkedIn](https://linkedin.com/in/joshua-godwin-charles) · [GitHub](https://github.com/jgodwin31)

*DataCamp Certified Data Engineer (Feb 2026 · ID: DE0012368460067)*