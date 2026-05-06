import pandas as pd
import streamlit as st
import numpy as np
from sqlalchemy import create_engine
from statsmodels.tsa.arima.model import ARIMA
from datetime import timedelta
import urllib.parse
# Page configuration
st.set_page_config(page_title="NVIDIA Stock Forecast", layout="wide")
st.title("📈 Stock Price Forecast Dashboard")

# 1. Database Connection
# Recommendation: In production, use st.secrets for credentials
db_pass = st.secrets ["db_password"]
safe_pass = urllib.parse.quote_plus(db_pass)
engine = create_engine(f"mysql+pymysql://root:{safe_pass}@localhost/wegagen_db")

@st.cache_data
def load_data():
    try:
        # Load from DB
        query = "SELECT * FROM nvidia_stocks"
        df = pd.read_sql(query, engine)
        
        # FIX 1: Standardize all columns to lowercase early
        df.columns = [c.lower() for c in df.columns] 
        df['date'] = pd.to_datetime(df['date'])
        return df.sort_values('date')
    except Exception as e:
        st.error(f"Error connecting to database: {e}")
        # Fallback logic for testing without DB
        return pd.DataFrame()

df = load_data()
# 2. Sidebar for User Input
st.sidebar.header("Forecast Settings")
if st.sidebar.button("Refresh Data"):
    st.cache_data.clear()
days_to_forecast = st.sidebar.slider("Days to Predict", 1, 365, 30)
# NEW: ARIMA parameters
p = st.sidebar.slider("AR (p)", 0, 5, 1)
d = st.sidebar.slider("Differencing (d)", 0, 2, 1)
q = st.sidebar.slider("MA (q)", 0, 5, 1)

# NEW: Date filter
start_date = st.sidebar.date_input("Start Date", df['date'].min())
end_date = st.sidebar.date_input("End Date", df['date'].max())
# 3. Training the Model
# FIX 2: Use recent data (last 100 days) to prevent "old trend" bias
# This ensures a 2027 prediction isn't heavily skewed by 2023 data
recent_df = df.tail(100).copy()
y = recent_df['close']
# ARIMA Model setup
recent_df = df.tail(100).copy()
y = recent_df['close']

model = ARIMA(y, order=(p, d, q))
model_fit = model.fit()
# 4. Generating Future Dates
last_date = df['date'].max()
future_dates = pd.bdate_range(start=last_date, periods=days_to_forecast+1)[1:]
# 5. Predicting
# FIX 3: Predict using the 'days_to_forecast' slider value, not a hardcoded '10'
forecast = model_fit.get_forecast(steps=days_to_forecast)
future_preds = forecast.predicted_mean
conf_int = forecast.conf_int()
# FIX 4: Maintain consistent column casing (lowercase) for the forecast dataframe
forecast_df = pd.DataFrame({
    'date': future_dates, 
    'close': future_preds.values 
})
# 6. Visualizing (Combine Historical + Forecast)
df['type'] = 'Historical'
forecast_df['type'] = 'Forecast'

# FIX 5: Ensure column names match perfectly for concatenation
combined_df = pd.concat([df[['date', 'close', 'type']], forecast_df])
st.subheader(f"NVIDIA Price Projection for the next {days_to_forecast} days")

# FIX 6: Use standardized lowercase names for charting
chart_data = combined_df.set_index('date')['close']
csv = combined_df.to_csv(index=False)

st.download_button(
    "Download Forecast Data",
    csv,
    "forecast.csv",
    "text/csv"
)
import plotly.graph_objects as go

fig = go.Figure()

fig.add_trace(go.Scatter(x=df['date'], y=df['close'], name='Historical'))
fig.add_trace(go.Scatter(x=future_dates, y=future_preds, name='Forecast'))

fig.add_trace(go.Scatter(
    x=future_dates,
    y=conf_int.iloc[:, 0],
    line=dict(width=0),
    showlegend=False
))

fig.add_trace(go.Scatter(
    x=future_dates,
    y=conf_int.iloc[:, 1],
    fill='tonexty',
    line=dict(width=0),
    name='Confidence Interval'
))

st.plotly_chart(fig, use_container_width=True)
st.subheader("Model Summary")
st.text(model_fit.summary())

st.subheader("Residuals")
st.line_chart(model_fit.resid)

# 7. Economic Insight
st.info(f"Target Forecast: Reach ${future_preds.values[-1]:.2f} by {future_dates[-1].date()}")

# Optional: Show Data Summary
with st.expander("View Data Summary"):
    st.write(combined_df.tail(days_to_forecast + 5))