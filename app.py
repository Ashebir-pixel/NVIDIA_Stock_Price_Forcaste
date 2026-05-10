import pandas as pd
import numpy as np
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.arima.model import ARIMA
from sqlalchemy import create_engine
import urllib.parse
from datetime import timedelta

# --- 1. SETTINGS & STYLING ---
st.set_page_config(page_title="NVIDIA Prediction Engine", layout="wide", page_icon="📈")

st.markdown("""
<style>
    .stMetric { background-color: #1e2130; padding: 15px; border-radius: 10px; border: 1px solid #3e4150; }
    .reportview-container { background: #0e1117; }
</style>
""", unsafe_allow_html=True)

# --- 2. ROBUST DATA LOADER (UPDATED WITH API) ---
@st.cache_data(ttl=3600) # Cache for 1 hour to keep data fresh
def load_data():
    """Fetches data from API (Primary), DB, or CSV fallbacks"""
    ticker_symbol = "NVDA"
    
    # --- STRATEGY 1: API CALL (PRIMARY) ---
    try:
        # Fetching historical data up to the current date in 2026
        # Fetching data directly from the API
        # We use a date range that includes today's date in 2026
        df = yf.download(ticker_symbol, start="1999-01-01", end="2026-05-10")
        if df.empty:
            raise ValueError("API returned empty DataFrame")
        # IMPORTANT: yfinance 2026 returns a Multi-Index header. 
        # We must flatten it so the rest of your code (ARIMA, etc.) works.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df = df.reset_index()
        st.sidebar.success("✅ Connected to Live Yahoo Finance API")
        return df.rename(columns={"Date": "Date", "Close": "Close"})
    
    except Exception as e:
        st.sidebar.warning(f"⚠️ API Connection Failed: {e}. Falling back to CSV.")
        # Only if the API fails, it goes to the CSV
        df = pd.read_csv("nvidia_stock_data_1999_2026.csv")
        return df
    except Exception as e:
        st.sidebar.warning(f"⚠️ API Connection Failed: {e}. Falling back to CSV.")
        # Only if the API fails, it goes to the CSV
        df = pd.read_csv("nvidia_stock_data_1999_2026.csv")
        return df
    # --- STRATEGY 2: DATABASE FALLBACK ---
    try:
        db_pass = st.secrets.get("db_password")
        if db_pass:
            safe_pass = urllib.parse.quote_plus(db_pass)
            engine = create_engine(f"mysql+pymysql://root:{safe_pass}@127.0.0.1/stock_data")
            df = pd.read_sql("SELECT * FROM nvidia_historical", engine)
            st.sidebar.success("📡 Database: Online")
        else:
            raise ValueError("No Credentials")
    except Exception:
        # --- STRATEGY 3: LOCAL CSV FALLBACK ---
        try:
            df = pd.read_csv("nvidia_stock_data_1999_2026.csv")
            st.sidebar.info("📂 Using Local CSV File")
        except Exception:
            # --- STRATEGY 4: SIMULATION ---
            dates = pd.date_range(end="2026-05-08", periods=1000)
            df = pd.DataFrame({
                "Date": dates,
                "Close": 100 + np.cumsum(np.random.normal(0.5, 2.5, 1000)),
            })
            st.sidebar.warning("⚠️ Simulation Mode Active")

    df.columns = df.columns.str.lower().str.strip()
    date_col = next((c for c in df.columns if "date" in c), df.columns[0])
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col)
    p_col = next((c for c in df.columns if "close" in c or "price" in c), "close")
    return df.rename(columns={date_col: "Date", p_col: "Close"})

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=period).mean()
    loss = -delta.clip(upper=0).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# --- 3. CORE LOGIC ---
df = load_data()

# Technical Indicators
df["SMA50"] = df["Close"].rolling(window=50).mean()
df["SMA200"] = df["Close"].rolling(window=200).mean()
df["DailyReturn"] = df["Close"].pct_change()
df["Volatility20"] = df["DailyReturn"].rolling(20).std() * np.sqrt(252) * 100
df["RSI14"] = compute_rsi(df["Close"])

# Sidebar Controls
st.sidebar.header("Forecast Settings")
scenario = st.sidebar.selectbox("Market Scenario", ["Neutral", "Bullish", "Bearish"])
days_to_predict = st.sidebar.slider("Days to Predict", 7, 365, 30)
show_sma50 = st.sidebar.checkbox("Show SMA 50", value=True)
show_rsi = st.sidebar.checkbox("Show RSI Panel", value=True)

with st.sidebar.expander("Advanced ARIMA Parameters"):
    p = st.slider("AR (p)", 0, 5, 1)
    d = st.slider("I (d)", 0, 2, 1)
    q = st.slider("MA (q)", 0, 5, 1)

# ARIMA Modeling
training_df = df.tail(500)
y = training_df["Close"].values

try:
    model = ARIMA(y, order=(p, d, q))
    model_fit = model.fit()

    drift_map = {"Neutral": 0, "Bullish": 0.05, "Bearish": -0.05}
    forecast_res = model_fit.get_forecast(steps=days_to_predict)
    future_preds = forecast_res.predicted_mean * (1 + drift_map[scenario])
    conf_int = forecast_res.conf_int(alpha=0.05)

    last_date = df["Date"].max()
    future_dates = pd.bdate_range(start=last_date + timedelta(days=1), periods=days_to_predict)
    # Create a DataFrame for the exported data
    forecast_export = pd.DataFrame({
        "Date": future_dates,
        "Predicted_Close": future_preds,
        "Lower_95_Confidence": conf_int[:, 0],
        "Upper_95_Confidence": conf_int[:, 1],
        "Scenario": scenario,
        "ARIMA_Order": f"({p},{d},{q})"
    })
    # --- 4. UI SECTIONS ---
    st.title("📈 NVIDIA STOCK PRICE PREDICTION ENGINE")
    c1, c2, c3, c4, c5 = st.columns(5)
    current_px = float(df["Close"].iloc[-1])
    target_px = float(future_preds[-1])
    roi = ((target_px - current_px) / current_px) * 100

    c1.metric("Current Price (Live)", f"${current_px:.2f}")
    c2.metric("Target Projection", f"${target_px:.2f}", f"{roi:.1f}%")
    c3.metric("SMA 200", f"${df['SMA200'].iloc[-1]:.2f}")
    c4.metric("20D Volatility", f"{df['Volatility20'].iloc[-1]:.2f}%")
    c5.metric("RSI (14)", f"{df['RSI14'].iloc[-1]:.1f}")

    # Visualization
    if show_rsi:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.03)
    else:
        fig = make_subplots(rows=1, cols=1)

    hist_view = df.tail(200)
    fig.add_trace(go.Scatter(x=hist_view["Date"], y=hist_view["Close"], name="Historical", line=dict(color="#10b981")), row=1, col=1)
    
    if show_sma50:
        fig.add_trace(go.Scatter(x=hist_view["Date"], y=hist_view["SMA50"], name="SMA 50", line=dict(color="#60a5fa", width=1, dash="dot")), row=1, col=1)

    fig.add_trace(go.Scatter(x=future_dates, y=future_preds, name="Forecast", line=dict(color="#f59e0b", width=3, dash="dash")), row=1, col=1)

    fig.update_layout(template="plotly_dark", height=700 if show_rsi else 600)
    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"Error in Engine: {e}")