import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from data_loader import load_stock_data
from train_model import train_model


# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="NIFTY50 Stock Predictor",
    page_icon="📈",
    layout="wide"
)

# -----------------------------
# Cute UI Styling
# -----------------------------

st.markdown("""
<style>
.stApp {
    background: linear-gradient(120deg,#1f4037,#99f2c8);
}

h1 {
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Title
# -----------------------------

st.title("📊 Stock Price Prediction System")

st.markdown(
"Predict **future stock prices** of NIFTY50 companies using **Machine Learning + Yahoo Finance API**"
)

# -----------------------------
# NIFTY50 Companies
# -----------------------------

nifty50_stocks = {
    "Reliance Industries": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "Infosys": "INFY.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "State Bank of India": "SBIN.NS",
    "Larsen & Toubro": "LT.NS",
    "Bharti Airtel": "BHARTIARTL.NS",
    "ITC": "ITC.NS",
    "HCL Technologies": "HCLTECH.NS",
    "Kotak Mahindra Bank": "KOTAKBANK.NS",
    "Axis Bank": "AXISBANK.NS",
    "Asian Paints": "ASIANPAINT.NS",
    "Maruti Suzuki": "MARUTI.NS",
    "Titan Company": "TITAN.NS",
    "Sun Pharma": "SUNPHARMA.NS",
    "UltraTech Cement": "ULTRACEMCO.NS",
    "Bajaj Finance": "BAJFINANCE.NS",
    "Bajaj Finserv": "BAJAJFINSV.NS",
    "Wipro": "WIPRO.NS",
    "Nestle India": "NESTLEIND.NS",
    "Power Grid": "POWERGRID.NS",
    "NTPC": "NTPC.NS",
    "JSW Steel": "JSWSTEEL.NS",
    "Tata Steel": "TATASTEEL.NS"
}

company = st.selectbox("🏢 Select NIFTY50 Company", list(nifty50_stocks.keys()))
stock_symbol = nifty50_stocks[company]

# -----------------------------
# LOAD DATA
# -----------------------------

if st.button("📥 Load Stock Data"):

    data = load_stock_data(stock_symbol)

    col1, col2, col3 = st.columns(3)

    col1.metric("Company", company)
    col2.metric("Total Records", len(data))
    col3.metric("Latest Price ₹", round(data["Close"].iloc[-1],2))

    st.subheader("📋 Recent Stock Data")
    st.dataframe(data.tail())

    st.subheader("📊 Quick Stock Insights")

    col1, col2 = st.columns(2)

    # Chart 1 — Trend
    with col1:
        fig, ax = plt.subplots(figsize=(4,2.5))
        ax.plot(data['Close'].tail(100), linewidth=2)
        ax.set_title("Last 100 Days Trend")
        st.pyplot(fig)

    # Chart 2 — Distribution
    with col2:
        fig, ax = plt.subplots(figsize=(4,2.5))
        ax.hist(data['Close'], bins=25)
        ax.set_title("Price Distribution")
        st.pyplot(fig)

    col3, col4 = st.columns(2)

    # Chart 3 — Moving Average
    with col3:
        data['MA20'] = data['Close'].rolling(20).mean()
        fig, ax = plt.subplots(figsize=(4,2.5))
        ax.plot(data['Close'].tail(100))
        ax.plot(data['MA20'].tail(100))
        ax.set_title("Moving Average")
        st.pyplot(fig)

    # Chart 4 — Box Plot
    with col4:
        fig, ax = plt.subplots(figsize=(4,2.5))
        ax.boxplot(data['Close'])
        ax.set_title("Price Volatility")
        st.pyplot(fig)


# -----------------------------
# PREDICTION SECTION
# -----------------------------

if st.button("🤖 Predict Future Price"):

    model, accuracy = train_model()

    st.success(f"Model Accuracy: {accuracy:.4f}")

    data = pd.read_csv("stock_data.csv")
    data = data[['Close']]

    future_days = 30

    x_future = np.array(data[['Close']])[-future_days:]

    prediction = model.predict(x_future)

    prediction_df = pd.DataFrame({
        "Day": range(1, future_days+1),
        "Predicted Price (₹)": np.round(prediction,4)
    })

    st.subheader("🔮 Future Price Prediction")

    st.dataframe(prediction_df, height=300)

    # -----------------------------
    # Clean Prediction Graph
    # -----------------------------

    st.subheader("📈 Predicted Price Trend")

    fig, ax = plt.subplots(figsize=(8,4))

    days = prediction_df["Day"]
    prices = prediction_df["Predicted Price (₹)"]

    ax.plot(days, prices, marker="o", linewidth=2)

    ax.set_title("Predicted Stock Price (Next 30 Days)")
    ax.set_xlabel("Day")
    ax.set_ylabel("Predicted Price (₹)")

    ax.grid(True, linestyle="--", alpha=0.6)

    st.pyplot(fig)