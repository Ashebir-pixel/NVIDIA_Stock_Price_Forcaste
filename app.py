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

if df.empty:
    st.warning("No data found. Please check your database connection.")
    st.stop()

# 2. Sidebar for User Input
st.sidebar.header("Forecast Settings")
days_to_forecast = st.sidebar.slider("Days to Predict", 1, 365, 30)

# 3. Training the Model
# FIX 2: Use recent data (last 100 days) to prevent "old trend" bias
# This ensures a 2027 prediction isn't heavily skewed by 2023 data
recent_df = df.tail(100).copy()
y = recent_df['close']
# ARIMA Model setup
model = ARIMA(y, order=(1,1,1))
model_fit = model.fit()
# 4. Generating Future Dates
last_date = df['date'].max()
future_dates = [last_date + timedelta(days=i) for i in range(1, days_to_forecast + 1)]
# 5. Predicting
# FIX 3: Predict using the 'days_to_forecast' slider value, not a hardcoded '10'
future_preds = model_fit.forecast(steps=days_to_forecast)
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
st.line_chart(chart_data)

# 7. Economic Insight
st.info(f"Target Forecast: Reach ${future_preds.values[-1]:.2f} by {future_dates[-1].date()}")

# Optional: Show Data Summary
with st.expander("View Data Summary"):
    st.write(combined_df.tail(days_to_forecast + 5))