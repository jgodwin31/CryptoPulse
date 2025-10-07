-- Optional: initial schema SQL, same as created in load_data.init_schema()
CREATE TABLE IF NOT EXISTS crypto_market (
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
CREATE INDEX IF NOT EXISTS idx_crypto_market_symbol ON crypto_market (symbol);
CREATE INDEX IF NOT EXISTS idx_crypto_market_fetched_at ON crypto_market (fetched_at);
