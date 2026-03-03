# CryptoPulse 📈

A real-time cryptocurrency data engineering pipeline with market analytics and an interactive dashboard. Built end-to-end with Python, PostgreSQL, and Streamlit — demonstrating production data engineering patterns including batch ingestion, schema design, feature engineering, and live visualization.

---

## Architecture

```
CoinGecko API
      │
      ▼
┌─────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│  extract_   │────▶│  transform_data  │────▶│     load_data       │
│  data.py    │     │  .py             │     │     .py             │
│             │     │  - Type casting  │     │  - PostgreSQL       │
│  - Retry    │     │  - Normalization │     │  - Schema init      │
│  - Backoff  │     │  - Volatility    │     │  - Dedup via UNIQUE │
└─────────────┘     └──────────────────┘     └─────────────────────┘
                                                        │
                                                        ▼
                                              ┌─────────────────────┐
                                              │   crypto_market     │
                                              │   (PostgreSQL)      │
                                              │                     │
                                              │  coin_id, symbol,   │
                                              │  price, market_cap, │
                                              │  volume, vol_index, │
                                              │  fetched_at         │
                                              └─────────────────────┘
                                                        │
                                                        ▼
                                              ┌─────────────────────┐
                                              │  Streamlit Dashboard│
                                              │  - Live price table │
                                              │  - Time-series chart│
                                              │  - Asset comparison │
                                              │  - Volatility index │
                                              └─────────────────────┘
```

---

## Features

- **End-to-end ETL pipeline** — extracts top 10 cryptos by market cap from CoinGecko, transforms and validates data, loads into PostgreSQL with idempotent design
- **Retry logic with exponential backoff** — handles API rate limits and transient failures gracefully
- **Volatility index** — computed from rolling standard deviation of price returns using historical DB data, not just API-provided metrics
- **Schema with indexing** — PostgreSQL table with indexes on `symbol` and `fetched_at` for fast time-series queries; `UNIQUE(coin_id, fetched_at)` prevents duplicate ingestion
- **Interactive Streamlit dashboard** — real-time price table, configurable time-range charts (24h / 7d / 30d / 90d), multi-asset overlay, and side-by-side comparison
- **Cron/scheduler-ready orchestrator** — `run_pipeline.py` designed to be triggered by cron, systemd timer, or Airflow

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11+ |
| Data extraction | `requests`, CoinGecko API |
| Transformation | `pandas`, `numpy` |
| Database | PostgreSQL + `psycopg2` |
| ORM / query | SQLAlchemy 2.0 |
| Dashboard | Streamlit + Plotly |
| Orchestration | `run_pipeline.py` (cron-compatible) |
| Config | `python-dotenv` |

---

## Project Structure

```
CryptoPulse/
├── etl/
│   ├── extract_data.py       # CoinGecko API client with retry/backoff
│   ├── transform_data.py     # Data cleaning, type casting, volatility computation
│   └── load_data.py          # PostgreSQL schema init + batch insert
├── dashboard/
│   └── dashboard.py          # Streamlit dashboard
├── run_pipeline.py           # ETL orchestrator (cron-ready)
├── .env.example              # Environment variable template
├── requirements.txt          # Python dependencies
└── README.md
```

---

## Database Schema

```sql
CREATE TABLE crypto_market (
  id              BIGSERIAL PRIMARY KEY,
  coin_id         TEXT NOT NULL,
  symbol          TEXT NOT NULL,
  name            TEXT,
  fetched_at      TIMESTAMP WITH TIME ZONE NOT NULL,
  current_price   DOUBLE PRECISION,
  market_cap      DOUBLE PRECISION,
  total_volume    DOUBLE PRECISION,
  price_change_pct DOUBLE PRECISION,
  volatility_index DOUBLE PRECISION,
  last_updated    TIMESTAMP WITH TIME ZONE,
  UNIQUE(coin_id, fetched_at)
);

CREATE INDEX idx_crypto_market_symbol    ON crypto_market (symbol);
CREATE INDEX idx_crypto_market_fetched_at ON crypto_market (fetched_at);
```

---

## Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/jgodwin31/CryptoPulse.git
cd CryptoPulse
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
# Fill in your PostgreSQL credentials
```

`.env` variables:

```
PGUSER=your_pg_user
PGPASSWORD=your_pg_password
PGHOST=localhost
PGPORT=5432
PGDATABASE=cryptopulse
```

### 4. Run the pipeline

```bash
python run_pipeline.py
```

This will create the schema on first run, fetch the top 10 cryptos, transform and load them into PostgreSQL.

### 5. Launch the dashboard

```bash
streamlit run dashboard/dashboard.py
```

### 6. Schedule recurring runs (optional)

Add to crontab to run every 5 minutes:

```bash
*/5 * * * * cd /path/to/CryptoPulse && python run_pipeline.py >> logs/pipeline.log 2>&1
```

---

## Key Engineering Decisions

**Idempotent ingestion** — The `UNIQUE(coin_id, fetched_at)` constraint means re-running the pipeline never creates duplicate records. Safe to run on a schedule without extra deduplication logic.

**Volatility from historical data** — Rather than relying solely on CoinGecko's 24h change metric, CryptoPulse computes a rolling volatility index from the local price history in PostgreSQL. This gives a more accurate, custom measure of price stability over time.

**Exponential backoff on extraction** — The extractor retries up to 3 times with `2^attempt` second delays, preventing cascading failures from temporary API issues.

**Schema-first design** — The database schema is initialized programmatically by the pipeline on first run, making the project self-contained and easy to deploy fresh.

---

## Roadmap

- [ ] Binance WebSocket integration for true real-time (sub-second) price streaming
- [ ] XGBoost price prediction model with RSI, moving averages, and volume features
- [ ] Trading signal engine (BUY / SELL / HOLD) based on model output
- [ ] Paper trading simulator with virtual portfolio and live P&L tracking
- [ ] Airflow DAG for production-grade orchestration
- [ ] Dockerized deployment

---

## Author

**Joshua Godwin** — Data Engineer & AI Specialist

- Portfolio: [joshuagodwin.vercel.app](https://joshuagodwin.vercel.app)
- LinkedIn: [linkedin.com/in/joshua-godwin-charles](https://linkedin.com/in/joshua-godwin-charles)
- GitHub: [github.com/jgodwin31](https://github.com/jgodwin31)