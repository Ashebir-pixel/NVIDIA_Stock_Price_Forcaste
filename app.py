import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from sqlalchemy import create_engine
import urllib.parse
from datetime import datetime, timedelta

# --- 1. SETTINGS & STYLING ---
st.set_page_config(page_title="NVIDIA Prediction Engine", layout="wide", page_icon="📈")

# Dark Theme Styling
st.markdown("""
<style>
    .stMetric { background-color: #1e2130; padding: 15px; border-radius: 10px; border: 1px solid #3e4150; }
    .reportview-container { background: #0e1117; }
</style>
""", unsafe_allow_html=True)

# --- 2. ROBUST DATA LOADER ---
@st.cache_data
def load_data():
    """Fetches data from DB or local CSV with fallback to simulation"""
    df = None
    try:
        # DB Connection Attempt
        db_pass = st.secrets.get("db_password")
        if db_pass:
            safe_pass = urllib.parse.quote_plus(db_pass)
            engine = create_engine(f"mysql+pymysql://root:{safe_pass}@127.0.0.1/stock_data")
            df = pd.read_sql("SELECT * FROM nvidia_historical", engine)
            st.sidebar.success("📡 Database: Online")
        else:
            raise ValueError("No Credentials")
    except:
        try:
            # CSV Fallback
            df = pd.read_csv('nvidia_stock_data_1999_2026.csv')
            st.sidebar.info("📂 Using Local CSV File")
        except:
            # Emergency Simulation (Prevents code crash)
            dates = pd.date_range(end='2026-05-08', periods=1000)
            df = pd.DataFrame({
                'Date': dates,
                'Close': 100 + np.cumsum(np.random.normal(0.5, 2.5, 1000))
            })
            st.sidebar.warning("⚠️ Simulation Mode Active")

    # FIX: Standardize Columns (Lowercase + Strip)
    df.columns = df.columns.str.lower().str.strip()
    
    # FIX: Date Handling
    date_col = next((c for c in df.columns if 'date' in c), df.columns[0])
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col]).sort_values(date_col)
    
    # FIX: Price Identification
    p_col = next((c for c in df.columns if 'close' in c or 'price' in c), 'close')
    return df.rename(columns={date_col: 'Date', p_col: 'Close'})

# --- 3. CORE LOGIC ---
df = load_data()

# Sidebar: Forecast Controls
st.sidebar.header("Forecast Settings")
scenario = st.sidebar.selectbox("Market Scenario", ["Neutral", "Bullish", "Bearish"])
days_to_predict = st.sidebar.slider("Days to Predict", 7, 365, 30)

with st.sidebar.expander("Advanced ARIMA Parameters"):
    p = st.slider("AR (p)", 0, 5, 1)
    d = st.slider("I (d)", 0, 2, 1)
    q = st.slider("MA (q)", 0, 5, 1)

# Feature 1: Moving Averages
df['SMA50'] = df['Close'].rolling(window=50).mean()
df['SMA200'] = df['Close'].rolling(window=200).mean()

# ARIMA Modeling
training_df = df.tail(500) # Use last 500 days for speed/stability
y = training_df['Close'].values

try:
    # Model Fit
    model = ARIMA(y, order=(p, d, q))
    model_fit = model.fit()
    
    # Adjustment for Scenarios (Drift)
    drift_map = {"Neutral": 0, "Bullish": 0.05, "Bearish": -0.05}
    forecast_res = model_fit.get_forecast(steps=days_to_predict)
    # Apply scenario drift to predicted mean
    future_preds = forecast_res.predicted_mean * (1 + drift_map[scenario])
    conf_int = forecast_res.conf_int(alpha=0.05)
    
    # Generate Future Dates (Business Days only)
    last_date = df['Date'].max()
    future_dates = pd.bdate_range(start=last_date + timedelta(days=1), periods=days_to_predict)

    # --- 4. UI SECTIONS ---
    st.title("📈 NVIDIA Forecast Console")
    
    # Metrics
    c1, c2, c3, c4 = st.columns(4)
    current_px = df['Close'].iloc[-1]
    target_px = future_preds[-1]
    roi = ((target_px - current_px) / current_px) * 100
    
    c1.metric("Current Price", f"${current_px:.2f}")
    c2.metric("Target Projection", f"${target_px:.2f}", f"{roi:.1f}%")
    c3.metric("SMA 200", f"${df['SMA200'].iloc[-1]:.2f}")
    c4.metric("Status", scenario, delta_color="normal")

    # Charting with Plotly
    fig = go.Figure()
    
    # Historical Trace (Last 200 days)
    hist_view = df.tail(200)
    fig.add_trace(go.Scatter(x=hist_view['Date'], y=hist_view['Close'], name='Historical', line=dict(color='#10b981')))
    
    # SMA Overlays
    fig.add_trace(go.Scatter(x=hist_view['Date'], y=hist_view['SMA200'], name='SMA 200', line=dict(color='purple', width=1, dash='dot')))
    
    # Forecast Trace
    fig.add_trace(go.Scatter(x=future_dates, y=future_preds, name='Forecast', line=dict(color='#f59e0b', width=3, dash='dash')))
    
    # Confidence Intervals (Shaded)
    fig.add_trace(go.Scatter(
        x=future_dates.tolist() + future_dates.tolist()[::-1],
        y=conf_int[:, 1].tolist() + conf_int[:, 0].tolist()[::-1],
        fill='toself',
        fillcolor='rgba(245, 158, 11, 0.1)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        name='Confidence Interval'
    ))

    fig.update_layout(template="plotly_dark", height=600, margin=dict(l=0, r=0, t=20, b=0))
    st.plotly_chart(fig, use_container_width=True)

    # Economic Insight
    st.info(f"Analysis: The model predicts a {scenario} trend reaching ${target_px:.2f} by {future_dates[-1].date()}.")
    
except Exception as e:
    st.error(f"Prediction Error: {e}")