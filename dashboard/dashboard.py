"""
Streamlit dashboard to visualize CryptoPulse data.

Run:
streamlit run dashboard/dashboard.py
"""

import os
from datetime import datetime, timedelta
from functools import lru_cache

import pandas as pd
import sqlalchemy
import streamlit as st
import plotly.express as px
import dotenv

dotenv.load_dotenv()

# DB helper
def get_engine():
    user = os.getenv("PGUSER")
    password = os.getenv("PGPASSWORD")
    host = os.getenv("PGHOST", "localhost")
    port = os.getenv("PGPORT", "5432")
    db = os.getenv("PGDATABASE")
    url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"
    return sqlalchemy.create_engine(url, future=True)

@st.cache_data(ttl=30)
def load_latest_snapshot(limit: int = 1000):
    engine = get_engine()
    query = f"""
      SELECT coin_id, symbol, name, fetched_at, current_price, market_cap, total_volume, price_change_pct, volatility_index, last_updated
      FROM crypto_market
      ORDER BY fetched_at DESC, market_cap DESC
      LIMIT :limit
    """
    with engine.connect() as conn:
        df = pd.read_sql_query(sqlalchemy.text(query), conn, params={"limit": limit})
    # convert timestamps
    df["fetched_at"] = pd.to_datetime(df["fetched_at"])
    return df

@st.cache_data(ttl=60)
def load_series(symbols: list, start: datetime, end: datetime):
    engine = get_engine()
    q = """
      SELECT symbol, fetched_at, current_price
      FROM crypto_market
      WHERE fetched_at BETWEEN :start AND :end
        AND symbol = ANY(:symbols)
      ORDER BY fetched_at ASC
    """
    with engine.connect() as conn:
        df = pd.read_sql_query(sqlalchemy.text(q), conn, params={"start": start, "end": end, "symbols": symbols})
    df["fetched_at"] = pd.to_datetime(df["fetched_at"])
    return df

# Streamlit UI
st.set_page_config(page_title="CryptoPulse", page_icon=":chart_with_upwards_trend:", layout="wide")
st.markdown(
    "<h1 style='text-align:center;'>CryptoPulse — Real-time Cryptocurrency Dashboard</h1>",
    unsafe_allow_html=True
)

# top bar: show latest
latest = load_latest_snapshot(limit=2000)
if latest.empty:
    st.error("No data found in database. Run the ETL pipeline first.")
    st.stop()

# derive unique symbols and default selection
symbols = sorted(latest["symbol"].unique().tolist())
default_selection = symbols[:4] if len(symbols) >= 4 else symbols

# Sidebar controls
st.sidebar.header("Controls")
sel_symbols = st.sidebar.multiselect("Select cryptocurrencies", symbols, default=default_selection)
date_range = st.sidebar.selectbox("Time range", ["24 hours", "7 days", "30 days", "90 days", "All"], index=1)
now = datetime.utcnow()
if date_range == "24 hours":
    start = now - timedelta(days=1)
elif date_range == "7 days":
    start = now - timedelta(days=7)
elif date_range == "30 days":
    start = now - timedelta(days=30)
elif date_range == "90 days":
    start = now - timedelta(days=90)
else:
    # choose earliest
    start = latest["fetched_at"].min()

end = now

# Top metrics: current prices and 24h changes (most recent snapshot per symbol)
st.markdown("### Current Prices & % Change")
# get latest per symbol
latest_per_symbol = latest.sort_values("fetched_at").groupby("symbol").last().reset_index()
display_df = latest_per_symbol[["symbol", "name", "current_price", "price_change_pct", "volatility_index", "market_cap", "total_volume"]]
display_df = display_df.sort_values("market_cap", ascending=False)
# format numbers
display_df["current_price"] = display_df["current_price"].map(lambda v: f"${v:,.2f}" if pd.notna(v) else "N/A")
display_df["price_change_pct"] = display_df["price_change_pct"].map(lambda v: f"{v:.2f}%" if pd.notna(v) else "N/A")
display_df["volatility_index"] = display_df["volatility_index"].map(lambda v: f"{v:.6f}" if pd.notna(v) else "N/A")
st.dataframe(display_df, use_container_width=True)

# Fetch series for selected symbols
if not sel_symbols:
    st.warning("Select at least one cryptocurrency on the left")
    st.stop()

series_df = load_series(sel_symbols, start, end)
if series_df.empty:
    st.warning("No historical data for selection and range.")
else:
    # line chart: price over time by symbol
    st.markdown("### Price over time")
    fig = px.line(series_df, x="fetched_at", y="current_price", color="symbol",
                  labels={"fetched_at": "Time", "current_price": "Price (USD)"}, template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    # interactive comparison: choose two symbols to compare (default first two)
    st.markdown("### Compare two assets")
    left, right = st.columns(2)
    with left:
        comp_a = st.selectbox("Asset A", sel_symbols, index=0)
    with right:
        comp_b = st.selectbox("Asset B", sel_symbols, index=min(1, max(0, len(sel_symbols)-1)))
    comp_df = series_df[series_df["symbol"].isin([comp_a, comp_b])]
    if not comp_df.empty:
        fig2 = px.line(comp_df, x="fetched_at", y="current_price", color="symbol", template="plotly_dark")
        st.plotly_chart(fig2, use_container_width=True)

# Table of live market stats (last snapshot)
st.markdown("### Live Market Stats (most recent snapshot)")
st.table(display_df.head(20))

# Footer / notes
st.markdown("""
*Data source: CoinGecko public API. Refresh frequency depends on ETL scheduler (e.g. cron).*
""")
