import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.arima.model import ARIMA
from sqlalchemy import create_engine
import urllib.parse
from datetime import timedelta

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
        db_pass = st.secrets.get("db_password")
        if db_pass:
            safe_pass = urllib.parse.quote_plus(db_pass)
            engine = create_engine(f"mysql+pymysql://root:{safe_pass}@127.0.0.1/stock_data")
            df = pd.read_sql("SELECT * FROM nvidia_historical", engine)
            st.sidebar.success("📡 Database: Online")
        else:
            raise ValueError("No Credentials")
    except Exception:
        try:
            df = pd.read_csv("nvidia_stock_data_1999_2026.csv")
            st.sidebar.info("📂 Using Local CSV File")
        except Exception:
            dates = pd.date_range(end="2026-05-08", periods=1000)
            df = pd.DataFrame(
                {
                    "Date": dates,
                    "Close": 100 + np.cumsum(np.random.normal(0.5, 2.5, 1000)),
                }
            )
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

# Sidebar: Forecast Controls
st.sidebar.header("Forecast Settings")
scenario = st.sidebar.selectbox("Market Scenario", ["Neutral", "Bullish", "Bearish"])
days_to_predict = st.sidebar.slider("Days to Predict", 7, 365, 30)
show_sma50 = st.sidebar.checkbox("Show SMA 50", value=True)
show_rsi = st.sidebar.checkbox("Show RSI Panel", value=True)

with st.sidebar.expander("Advanced ARIMA Parameters"):
    p = st.slider("AR (p)", 0, 5, 1)
    d = st.slider("I (d)", 0, 2, 1)
    q = st.slider("MA (q)", 0, 5, 1)

# Technical Indicators
df["SMA50"] = df["Close"].rolling(window=50).mean()
df["SMA200"] = df["Close"].rolling(window=200).mean()
df["DailyReturn"] = df["Close"].pct_change()
df["Volatility20"] = df["DailyReturn"].rolling(20).std() * np.sqrt(252) * 100
df["RSI14"] = compute_rsi(df["Close"])

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

    # Quick backtest metric
    test_size = min(30, max(7, len(training_df) // 10))
    train_bt = training_df["Close"].iloc[:-test_size]
    test_bt = training_df["Close"].iloc[-test_size:]
    bt_fit = ARIMA(train_bt.values, order=(p, d, q)).fit()
    bt_pred = bt_fit.forecast(steps=test_size)
    mae = np.mean(np.abs(test_bt.values - bt_pred))
    mape = np.mean(np.abs((test_bt.values - bt_pred) / test_bt.values)) * 100

    # --- 4. UI SECTIONS ---
    st.title("📈 NVIDIA Forecast Console")

    c1, c2, c3, c4, c5 = st.columns(5)
    current_px = df["Close"].iloc[-1]
    target_px = future_preds[-1]
    roi = ((target_px - current_px) / current_px) * 100

    c1.metric("Current Price", f"${current_px:.2f}")
    c2.metric("Target Projection", f"${target_px:.2f}", f"{roi:.1f}%")
    c3.metric("SMA 200", f"${df['SMA200'].iloc[-1]:.2f}")
    c4.metric("20D Volatility", f"{df['Volatility20'].iloc[-1]:.2f}%")
    c5.metric("Backtest MAPE", f"{mape:.2f}%")

    # Dual-panel chart when RSI enabled
    if show_rsi:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.03)
    else:
        fig = make_subplots(rows=1, cols=1)

    hist_view = df.tail(200)
    fig.add_trace(go.Scatter(x=hist_view["Date"], y=hist_view["Close"], name="Historical", line=dict(color="#10b981")), row=1, col=1)
    fig.add_trace(go.Scatter(x=hist_view["Date"], y=hist_view["SMA200"], name="SMA 200", line=dict(color="purple", width=1, dash="dot")), row=1, col=1)

    if show_sma50:
        fig.add_trace(go.Scatter(x=hist_view["Date"], y=hist_view["SMA50"], name="SMA 50", line=dict(color="#60a5fa", width=1, dash="dot")), row=1, col=1)

    fig.add_trace(go.Scatter(x=future_dates, y=future_preds, name="Forecast", line=dict(color="#f59e0b", width=3, dash="dash")), row=1, col=1)
    fig.add_trace(
        go.Scatter(
            x=future_dates.tolist() + future_dates.tolist()[::-1],
            y=conf_int[:, 1].tolist() + conf_int[:, 0].tolist()[::-1],
            fill="toself",
            fillcolor="rgba(245, 158, 11, 0.1)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            name="Confidence Interval",
        ),
        row=1,
        col=1,
    )

    if show_rsi:
        fig.add_trace(go.Scatter(x=hist_view["Date"], y=hist_view["RSI14"], name="RSI 14", line=dict(color="#f43f5e")), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

    fig.update_layout(template="plotly_dark", height=700 if show_rsi else 600, margin=dict(l=0, r=0, t=20, b=0))
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        f"Analysis: The model predicts a {scenario} trend reaching ${target_px:.2f} by {future_dates[-1].date()} "
        f"(backtest MAE: ${mae:.2f}, MAPE: {mape:.2f}%)."
    )

    forecast_export = pd.DataFrame(
        {
            "Date": future_dates,
            "Predicted_Close": future_preds,
            "Lower_95": conf_int[:, 0],
            "Upper_95": conf_int[:, 1],
            "Scenario": scenario,
            "ARIMA_Order": f"({p},{d},{q})",
        }
    )
    st.download_button(
        "⬇️ Download Forecast CSV",
        data=forecast_export.to_csv(index=False).encode("utf-8"),
        file_name=f"nvidia_forecast_{scenario.lower()}_{days_to_predict}d.csv",
        mime="text/csv",
    )

except Exception as e:
    st.error(f"Prediction Error: {e}")