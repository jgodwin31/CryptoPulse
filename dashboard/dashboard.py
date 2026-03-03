"""
dashboard.py  —  CryptoPulse v2
--------------------------------
Real-time crypto dashboard with:
  • Live prices & 24h stats from Kraken ingestion
  • ML model signals (BUY / SELL / HOLD) with confidence
  • Paper trading portfolio — value, P&L, open positions
  • Trade history
  • Price & portfolio charts

Run:
    streamlit run dashboard/dashboard.py
"""

import os
import sys
from datetime import datetime, timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sqlalchemy
import streamlit as st
from dotenv import load_dotenv

# Allow imports from project root
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

load_dotenv()

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CryptoPulse",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg:        #080b10;
    --surface:   #0d1117;
    --border:    #1c2333;
    --green:     #00ff88;
    --red:       #ff3b5c;
    --yellow:    #f5c842;
    --blue:      #4d9fff;
    --muted:     #8b9ab1;
    --text:      #e2e8f0;
    --mono:      'Space Mono', monospace;
    --sans:      'DM Sans', sans-serif;
}

html, body, [class*="css"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--sans) !important;
}

/* Header */
.cp-header {
    display: flex;
    align-items: baseline;
    gap: 12px;
    padding: 8px 0 24px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 24px;
}
.cp-logo {
    font-family: var(--mono);
    font-size: 28px;
    font-weight: 700;
    color: var(--green);
    letter-spacing: -1px;
}
.cp-sub {
    font-family: var(--sans);
    font-size: 13px;
    color: var(--muted);
    font-weight: 300;
}
.cp-live {
    margin-left: auto;
    display: flex;
    align-items: center;
    gap: 6px;
    font-family: var(--mono);
    font-size: 11px;
    color: var(--green);
}
.cp-dot {
    width: 7px; height: 7px;
    background: var(--green);
    border-radius: 50%;
    animation: pulse 1.5s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: 0.4; transform: scale(0.8); }
}

/* Metric cards */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 24px;
}
.metric-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px 20px;
}
.metric-label {
    font-size: 11px;
    font-family: var(--mono);
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 6px;
}
.metric-value {
    font-family: var(--mono);
    font-size: 22px;
    font-weight: 700;
    color: var(--text);
}
.metric-value.green  { color: var(--green); }
.metric-value.red    { color: var(--red); }
.metric-value.yellow { color: var(--yellow); }

/* Signal badge */
.signal-buy  { color: var(--green); font-family: var(--mono); font-weight: 700; font-size: 13px; }
.signal-sell { color: var(--red);   font-family: var(--mono); font-weight: 700; font-size: 13px; }
.signal-hold { color: var(--yellow);font-family: var(--mono); font-weight: 700; font-size: 13px; }

/* Section header */
.section-header {
    font-family: var(--mono);
    font-size: 11px;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 2px;
    border-bottom: 1px solid var(--border);
    padding-bottom: 8px;
    margin: 24px 0 16px;
}

/* Streamlit overrides */
[data-testid="stDataFrame"] { border: 1px solid var(--border) !important; border-radius: 8px; }
[data-testid="stSidebar"] { background: var(--surface) !important; border-right: 1px solid var(--border) !important; }
div[data-testid="metric-container"] { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 12px 16px; }
.stSelectbox > div, .stMultiSelect > div { background: var(--surface) !important; border-color: var(--border) !important; }
</style>
""", unsafe_allow_html=True)

# ── DB ─────────────────────────────────────────────────────────────────────────

@st.cache_resource
def get_engine():
    url = (
        f"postgresql+psycopg2://{os.getenv('PGUSER')}:{os.getenv('PGPASSWORD')}"
        f"@{os.getenv('PGHOST','localhost')}:{os.getenv('PGPORT',5432)}"
        f"/{os.getenv('PGDATABASE')}"
    )
    return sqlalchemy.create_engine(url, future=True)


def table_exists(engine, name: str) -> bool:
    q = "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name=:t)"
    with engine.connect() as c:
        return c.execute(sqlalchemy.text(q), {"t": name}).scalar()


@st.cache_data(ttl=10)
def load_latest(_engine) -> pd.DataFrame:
    q = """
        SELECT DISTINCT ON (symbol)
            symbol, current_price, price_change_pct, high_24h, low_24h,
            total_volume, fetched_at
        FROM crypto_market_rt
        ORDER BY symbol, fetched_at DESC
    """
    with _engine.connect() as c:
        df = pd.read_sql_query(sqlalchemy.text(q), c)
    return df


@st.cache_data(ttl=15)
def load_series(_engine, symbols: list, hours: int = 24) -> pd.DataFrame:
    start = datetime.utcnow() - timedelta(hours=hours)
    q = """
        SELECT symbol, fetched_at, current_price, total_volume
        FROM crypto_market_rt
        WHERE fetched_at >= :start AND symbol = ANY(:syms)
        ORDER BY fetched_at ASC
    """
    with _engine.connect() as c:
        df = pd.read_sql_query(sqlalchemy.text(q), c, params={"start": start, "syms": symbols})
    df["fetched_at"] = pd.to_datetime(df["fetched_at"])
    return df


@st.cache_data(ttl=10)
def load_portfolio(_engine) -> dict:
    if not table_exists(_engine, "paper_portfolio"):
        return {}
    q = "SELECT cash, total_value, total_pnl, total_pnl_pct FROM paper_portfolio ORDER BY id DESC LIMIT 1"
    with _engine.connect() as c:
        row = c.execute(sqlalchemy.text(q)).fetchone()
    if not row:
        return {}
    return {"cash": row[0], "total_value": row[1], "total_pnl": row[2], "total_pnl_pct": row[3]}


@st.cache_data(ttl=10)
def load_positions(_engine) -> pd.DataFrame:
    if not table_exists(_engine, "paper_positions"):
        return pd.DataFrame()
    q = "SELECT symbol, quantity, entry_price, entry_cost, opened_at FROM paper_positions ORDER BY opened_at DESC"
    with _engine.connect() as c:
        return pd.read_sql_query(sqlalchemy.text(q), c)


@st.cache_data(ttl=15)
def load_trades(_engine, limit: int = 50) -> pd.DataFrame:
    if not table_exists(_engine, "paper_trades"):
        return pd.DataFrame()
    q = f"""
        SELECT symbol, side, quantity, price, cost, fee, pnl, pnl_pct, signal, executed_at
        FROM paper_trades
        ORDER BY executed_at DESC
        LIMIT {limit}
    """
    with _engine.connect() as c:
        return pd.read_sql_query(sqlalchemy.text(q), c)


@st.cache_data(ttl=15)
def load_portfolio_history(_engine) -> pd.DataFrame:
    if not table_exists(_engine, "paper_portfolio_history"):
        return pd.DataFrame()
    q = "SELECT total_value, cash, positions_value, recorded_at FROM paper_portfolio_history ORDER BY recorded_at ASC"
    with _engine.connect() as c:
        df = pd.read_sql_query(sqlalchemy.text(q), c)
    df["recorded_at"] = pd.to_datetime(df["recorded_at"])
    return df


def load_signals(_engine) -> pd.DataFrame:
    """Try to load latest ML signals — gracefully returns empty if model not run yet."""
    try:
        from ml.model import predict_all
        from ingestion.ingest_kraken import SYMBOLS_KRAKEN, clean_symbol
        syms = [clean_symbol(s) for s in SYMBOLS_KRAKEN]
        return predict_all(syms, _engine)
    except Exception:
        return pd.DataFrame()

# ── Header ─────────────────────────────────────────────────────────────────────

now_str = datetime.utcnow().strftime("%H:%M:%S UTC")
st.markdown(f"""
<div class="cp-header">
    <span class="cp-logo">⚡ CRYPTOPULSE</span>
    <span class="cp-sub">Real-time intelligence & paper trading</span>
    <span class="cp-live">
        <span class="cp-dot"></span>
        LIVE · {now_str}
    </span>
</div>
""", unsafe_allow_html=True)

# ── Engine & data ──────────────────────────────────────────────────────────────

engine  = get_engine()
latest  = load_latest(engine)
portf   = load_portfolio(engine)

if latest.empty:
    st.error("⚠️  No data yet. Run `python -m ingestion.ingest_kraken` first.")
    st.stop()

# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Controls")
    symbols_all  = sorted(latest["symbol"].unique().tolist())
    sel_symbols  = st.multiselect("Coins", symbols_all, default=symbols_all[:5])
    time_options = {"1h": 1, "6h": 6, "24h": 24, "3d": 72, "7d": 168}
    sel_range    = st.selectbox("Time range", list(time_options.keys()), index=2)
    hours        = time_options[sel_range]
    auto_refresh = st.checkbox("Auto-refresh (30s)", value=True)
    st.markdown("---")
    st.markdown("### 💼 Paper Portfolio")
    if portf:
        pnl_color = "green" if portf.get("total_pnl", 0) >= 0 else "red"
        st.markdown(f"""
        <div style='font-family:monospace;font-size:13px;line-height:2'>
        💵 Cash: <b>${portf['cash']:,.2f}</b><br>
        📊 Value: <b>${portf['total_value']:,.2f}</b><br>
        <span class='{pnl_color}'>
        {'▲' if portf['total_pnl'] >= 0 else '▼'} P&L: ${portf['total_pnl']:+,.2f} ({portf['total_pnl_pct']:+.2f}%)
        </span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.caption("Run paper_trader.py to see portfolio")
    st.markdown("---")
    st.caption("Data: Kraken WebSocket · Model: XGBoost · DB: PostgreSQL")

if auto_refresh:
    st.empty()

# ── Portfolio metrics row ──────────────────────────────────────────────────────

if portf:
    st.markdown('<div class="section-header">Portfolio Overview</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Value", f"${portf['total_value']:,.2f}")
    with c2:
        st.metric("Cash Available", f"${portf['cash']:,.2f}")
    with c3:
        delta_str = f"{portf['total_pnl']:+,.2f}"
        st.metric("Total P&L", f"${portf['total_pnl']:,.2f}", delta=delta_str)
    with c4:
        st.metric("Return", f"{portf['total_pnl_pct']:+.2f}%")

# ── Live prices table ──────────────────────────────────────────────────────────

st.markdown('<div class="section-header">Live Prices</div>', unsafe_allow_html=True)

display = latest[latest["symbol"].isin(sel_symbols)].copy() if sel_symbols else latest.copy()
display = display.sort_values("current_price", ascending=False)

# Format for display
def fmt_price(v):
    if v is None or (isinstance(v, float) and v != v):
        return "—"
    return f"${v:,.4f}" if v < 1 else f"${v:,.2f}"

def fmt_pct(v):
    if v is None or (isinstance(v, float) and v != v):
        return "—"
    sign = "▲" if v >= 0 else "▼"
    return f"{sign} {abs(v):.2f}%"

def fmt_vol(v):
    if v is None or (isinstance(v, float) and v != v):
        return "—"
    if v >= 1e9: return f"${v/1e9:.2f}B"
    if v >= 1e6: return f"${v/1e6:.2f}M"
    return f"${v:,.0f}"

price_display = display.copy()
price_display["Price"]      = price_display["current_price"].apply(fmt_price)
price_display["24h Change"] = price_display["price_change_pct"].apply(fmt_pct)
price_display["24h High"]   = price_display["high_24h"].apply(fmt_price)
price_display["24h Low"]    = price_display["low_24h"].apply(fmt_price)
price_display["Volume"]     = price_display["total_volume"].apply(fmt_vol)
price_display["Updated"]    = pd.to_datetime(price_display["fetched_at"]).dt.strftime("%H:%M:%S")

st.dataframe(
    price_display[["symbol", "Price", "24h Change", "24h High", "24h Low", "Volume", "Updated"]]
        .rename(columns={"symbol": "Symbol"}),
    use_container_width=True,
    hide_index=True,
)

# ── ML Signals ─────────────────────────────────────────────────────────────────

st.markdown('<div class="section-header">ML Signals — XGBoost Price Direction</div>', unsafe_allow_html=True)

signals_df = load_signals(engine)

if signals_df.empty:
    st.info("📊 No signals yet. Train the model first:\n```\npython -m ml.model --symbol BTC --train\n```")
else:
    sig_filtered = signals_df[signals_df["symbol"].isin(sel_symbols)] if sel_symbols else signals_df

    def signal_badge(s):
        if s == "BUY":  return "🟢 BUY"
        if s == "SELL": return "🔴 SELL"
        return "🟡 HOLD"

    sig_display = sig_filtered.copy()
    sig_display["Signal"]     = sig_display["signal"].apply(signal_badge)
    sig_display["Confidence"] = sig_display["confidence"].apply(lambda v: f"{v*100:.1f}%")
    sig_display["P(Up)"]      = sig_display["prob_up"].apply(lambda v: f"{v*100:.1f}%")
    sig_display["P(Down)"]    = sig_display["prob_down"].apply(lambda v: f"{v*100:.1f}%")
    sig_display["Price"]      = sig_display["price"].apply(fmt_price)
    sig_display["Horizon"]    = sig_display["horizon"].apply(lambda v: f"{v} periods")

    st.dataframe(
        sig_display[["symbol", "Price", "Signal", "Confidence", "P(Up)", "P(Down)", "Horizon"]]
            .rename(columns={"symbol": "Symbol"}),
        use_container_width=True,
        hide_index=True,
    )

    # Signal summary bar
    counts = signals_df["signal"].value_counts()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"<div style='text-align:center;font-family:monospace;color:#00ff88;font-size:24px;font-weight:700'>{counts.get('BUY',0)}</div><div style='text-align:center;font-size:11px;color:#8b9ab1'>BUY SIGNALS</div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div style='text-align:center;font-family:monospace;color:#f5c842;font-size:24px;font-weight:700'>{counts.get('HOLD',0)}</div><div style='text-align:center;font-size:11px;color:#8b9ab1'>HOLD SIGNALS</div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div style='text-align:center;font-family:monospace;color:#ff3b5c;font-size:24px;font-weight:700'>{counts.get('SELL',0)}</div><div style='text-align:center;font-size:11px;color:#8b9ab1'>SELL SIGNALS</div>", unsafe_allow_html=True)

# ── Price Charts ───────────────────────────────────────────────────────────────

st.markdown('<div class="section-header">Price History</div>', unsafe_allow_html=True)

if sel_symbols:
    series_df = load_series(engine, sel_symbols, hours=hours)

    if series_df.empty:
        st.info(f"No historical data for selected range ({sel_range}). Data collects over time.")
    else:
        tab1, tab2 = st.tabs(["All Coins", "Compare Two"])

        with tab1:
            fig = px.line(
                series_df, x="fetched_at", y="current_price", color="symbol",
                labels={"fetched_at": "", "current_price": "Price (USD)", "symbol": ""},
                template="plotly_dark",
                color_discrete_sequence=["#00ff88", "#4d9fff", "#f5c842", "#ff3b5c", "#a78bfa",
                                         "#fb923c", "#34d399", "#f472b6", "#60a5fa", "#facc15"],
            )
            fig.update_layout(
                paper_bgcolor="#0d1117", plot_bgcolor="#080b10",
                font=dict(family="DM Sans", color="#8b9ab1"),
                legend=dict(bgcolor="#0d1117", bordercolor="#1c2333", borderwidth=1),
                xaxis=dict(gridcolor="#1c2333", showgrid=True),
                yaxis=dict(gridcolor="#1c2333", showgrid=True),
                margin=dict(l=0, r=0, t=10, b=0),
                height=380,
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            ca, cb = st.columns(2)
            with ca:
                coin_a = st.selectbox("Asset A", sel_symbols, index=0, key="cmp_a")
            with cb:
                coin_b = st.selectbox("Asset B", sel_symbols, index=min(1, len(sel_symbols)-1), key="cmp_b")

            cmp = series_df[series_df["symbol"].isin([coin_a, coin_b])]
            if not cmp.empty and coin_a != coin_b:
                # Normalise to % change from first value for fair comparison
                def normalise(g):
                    g = g.copy()
                    base = g["current_price"].iloc[0]
                    g["pct_change"] = (g["current_price"] - base) / base * 100
                    return g
                cmp_n = cmp.groupby("symbol", group_keys=False).apply(normalise)
                fig2 = px.line(
                    cmp_n, x="fetched_at", y="pct_change", color="symbol",
                    labels={"fetched_at": "", "pct_change": "% Change from start", "symbol": ""},
                    template="plotly_dark",
                    color_discrete_sequence=["#00ff88", "#4d9fff"],
                )
                fig2.update_layout(
                    paper_bgcolor="#0d1117", plot_bgcolor="#080b10",
                    font=dict(family="DM Sans", color="#8b9ab1"),
                    xaxis=dict(gridcolor="#1c2333"),
                    yaxis=dict(gridcolor="#1c2333"),
                    margin=dict(l=0, r=0, t=10, b=0),
                    height=320,
                )
                st.plotly_chart(fig2, use_container_width=True)

# ── Portfolio History Chart ────────────────────────────────────────────────────

port_hist = load_portfolio_history(engine)
if not port_hist.empty and len(port_hist) > 1:
    st.markdown('<div class="section-header">Portfolio Value Over Time</div>', unsafe_allow_html=True)

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=port_hist["recorded_at"], y=port_hist["total_value"],
        mode="lines", name="Total Value",
        line=dict(color="#00ff88", width=2),
        fill="tozeroy",
        fillcolor="rgba(0,255,136,0.05)",
    ))
    fig3.add_trace(go.Scatter(
        x=port_hist["recorded_at"], y=port_hist["cash"],
        mode="lines", name="Cash",
        line=dict(color="#4d9fff", width=1.5, dash="dot"),
    ))
    fig3.add_hline(
        y=10000, line_dash="dash", line_color="#8b9ab1",
        annotation_text="Starting $10,000", annotation_position="bottom right",
    )
    fig3.update_layout(
        paper_bgcolor="#0d1117", plot_bgcolor="#080b10",
        font=dict(family="DM Sans", color="#8b9ab1"),
        legend=dict(bgcolor="#0d1117", bordercolor="#1c2333", borderwidth=1),
        xaxis=dict(gridcolor="#1c2333"),
        yaxis=dict(gridcolor="#1c2333", tickprefix="$"),
        margin=dict(l=0, r=0, t=10, b=0),
        height=300,
    )
    st.plotly_chart(fig3, use_container_width=True)

# ── Open Positions ─────────────────────────────────────────────────────────────

positions = load_positions(engine)
if not positions.empty:
    st.markdown('<div class="section-header">Open Positions</div>', unsafe_allow_html=True)

    # Enrich with current prices
    price_map = dict(zip(latest["symbol"], latest["current_price"]))
    pos_display = positions.copy()
    pos_display["current_price"] = pos_display["symbol"].map(price_map)
    pos_display["current_value"] = pos_display["quantity"] * pos_display["current_price"]
    pos_display["unrealised_pnl"] = pos_display["current_value"] - pos_display["entry_cost"]
    pos_display["pnl_pct"] = pos_display["unrealised_pnl"] / pos_display["entry_cost"] * 100

    pos_fmt = pos_display[["symbol", "quantity", "entry_price", "current_price", "current_value", "unrealised_pnl", "pnl_pct"]].copy()
    pos_fmt.columns = ["Symbol", "Quantity", "Entry $", "Current $", "Value $", "Unrealised P&L", "P&L %"]
    pos_fmt["Entry $"]      = pos_fmt["Entry $"].apply(lambda v: f"${v:,.4f}" if v < 1 else f"${v:,.2f}")
    pos_fmt["Current $"]    = pos_fmt["Current $"].apply(lambda v: f"${v:,.4f}" if v and v < 1 else (f"${v:,.2f}" if v else "—"))
    pos_fmt["Value $"]      = pos_fmt["Value $"].apply(lambda v: f"${v:,.2f}" if v else "—")
    pos_fmt["Unrealised P&L"] = pos_fmt["Unrealised P&L"].apply(lambda v: f"${v:+,.2f}" if v else "—")
    pos_fmt["P&L %"]        = pos_fmt["P&L %"].apply(lambda v: f"{v:+.2f}%" if v else "—")
    pos_fmt["Quantity"]     = pos_fmt["Quantity"].apply(lambda v: f"{v:.6f}")

    st.dataframe(pos_fmt, use_container_width=True, hide_index=True)

# ── Trade History ──────────────────────────────────────────────────────────────

trades = load_trades(engine)
if not trades.empty:
    st.markdown('<div class="section-header">Trade History</div>', unsafe_allow_html=True)

    trades_fmt = trades.copy()
    trades_fmt["Price"]  = trades_fmt["price"].apply(fmt_price)
    trades_fmt["Cost"]   = trades_fmt["cost"].apply(lambda v: f"${v:,.2f}")
    trades_fmt["Fee"]    = trades_fmt["fee"].apply(lambda v: f"${v:,.4f}")
    trades_fmt["P&L"]    = trades_fmt["pnl"].apply(lambda v: f"${v:+,.2f}" if v is not None and v == v else "—")
    trades_fmt["P&L %"]  = trades_fmt["pnl_pct"].apply(lambda v: f"{v:+.2f}%" if v is not None and v == v else "—")
    trades_fmt["Side"]   = trades_fmt["side"].apply(lambda s: "🟢 BUY" if s == "BUY" else "🔴 SELL")
    trades_fmt["Time"]   = pd.to_datetime(trades_fmt["executed_at"]).dt.strftime("%m/%d %H:%M")

    st.dataframe(
        trades_fmt[["symbol", "Side", "Price", "Cost", "Fee", "P&L", "P&L %", "signal", "Time"]]
            .rename(columns={"symbol": "Symbol", "signal": "Signal"}),
        use_container_width=True,
        hide_index=True,
    )

# ── Footer ─────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown(
    "<div style='text-align:center;font-size:11px;color:#8b9ab1;font-family:monospace;padding:8px'>"
    "CRYPTOPULSE · Data: Kraken WebSocket · Model: XGBoost · "
    f"Last render: {datetime.utcnow().strftime('%H:%M:%S UTC')}"
    "</div>",
    unsafe_allow_html=True,
)

# Auto-refresh
if auto_refresh:
    import time
    time.sleep(30)
    st.rerun()