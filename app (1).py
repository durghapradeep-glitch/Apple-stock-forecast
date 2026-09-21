import streamlit as st
import pandas as pd
import numpy as np
from pandas.tseries.offsets import BDay
from statsmodels.tsa.statespace.sarimax import SARIMAX

st.set_page_config(page_title="Apple Stock Forecast", layout="wide")
st.title("Apple Stock Price Forecasting (SARIMA)")

@st.cache_data
def load_data():
    df = pd.read_csv("EDA_Processed_Apple_Stock.csv", parse_dates=["timestamp"])
    df.set_index("timestamp", inplace=True)
    df = df.asfreq("B").ffill()
    return df

df = load_data()
y = df["stock_price"]

@st.cache_resource
def train_model(series):
    model = SARIMAX(
        series,
        order=(1,1,1),
        seasonal_order=(0,1,1,5),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    return model.fit(disp=False)

sarima_fit = train_model(y)

def generate_future_trading_times(last_timestamp, n_days=30):
    business_days = pd.bdate_range(start=last_timestamp + BDay(1), periods=n_days)
    intraday_times = pd.date_range("09:30", "16:00", freq="30min").time
    return pd.DatetimeIndex(
        pd.Timestamp.combine(day.date(), t)
        for day in business_days
        for t in intraday_times
    )

if st.button("Predict Next 30 Trading Days"):
    future_index = generate_future_trading_times(y.index[-1], 30)
    daily_forecast = sarima_fit.forecast(steps=30)
    intraday_per_day = len(pd.date_range("09:30", "16:00", freq="30min"))
    expanded = np.repeat(daily_forecast.values, intraday_per_day)

    result = pd.DataFrame({
        "timestamp": future_index,
        "predicted_stock_price": expanded
    })

    st.subheader("Prediction Table")
    st.dataframe(result, use_container_width=True)

import matplotlib.pyplot as plt

st.subheader("Prediction Visualization (Intraday Expanded Daily SARIMA)")

fig, ax = plt.subplots(figsize=(14, 5))

ax.plot(
    result["timestamp"],
    result["predicted_stock_price"],
    linewidth=2,
    label="Predicted Stock Price"
)

ax.set_title("Apple Stock Price Forecast")
ax.set_xlabel("Time")
ax.set_ylabel("Predicted Stock Price")

ax.grid(True, linestyle="--", alpha=0.5)
ax.legend()

plt.xticks(rotation=45)
plt.tight_layout()

st.pyplot(fig)


st.subheader("Visualization")
st.line_chart(result.set_index("timestamp"))


