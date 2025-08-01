import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Page config
st.set_page_config(page_title="Household Energy Forecast", layout="wide")

st.title("🔋 Household Energy Consumption Forecasting App")

# Load data
st.subheader("📊 Raw Data")
df = pd.read_csv("household_energy.csv", parse_dates=['timestamp'])
st.write(df.head())

# Feature Engineering
df['hour'] = df['timestamp'].dt.hour
df['day'] = df['timestamp'].dt.day
df['weekday'] = df['timestamp'].dt.weekday
df['date'] = df['timestamp'].dt.date

# Show basic info
st.subheader("ℹ Dataset Info")
st.write("*Shape:*", df.shape)
st.write("*Columns:*", df.columns.tolist())

# Summary statistics
st.subheader("📈 Summary Statistics")
st.write(df.describe())

# Daily Consumption Plot
daily_consumption = df.groupby('date')['energy_consumption'].sum()
st.subheader("📅 Daily Energy Consumption")
fig1, ax1 = plt.subplots(figsize=(12, 4))
ax1.plot(daily_consumption.index, daily_consumption.values, marker='o')
ax1.set_title("Daily Household Energy Consumption")
ax1.set_xlabel("Date")
ax1.set_ylabel("Energy (kWh)")
ax1.grid(True)
st.pyplot(fig1)

# Histogram
st.subheader("📉 Energy Consumption Distribution")
fig2, ax2 = plt.subplots(figsize=(8, 4))
ax2.hist(df['energy_consumption'], bins=30, color='pink', edgecolor='black')
ax2.set_title("Distribution of Household Energy Consumption")
ax2.set_xlabel("Energy Consumption (kWh)")
ax2.set_ylabel("Frequency")
st.pyplot(fig2)

# Model Building
st.subheader("🔧 Model Training")

features = ['temperature', 'outside_temperature', 'device_usage', 'hour', 'weekday']
x = df[features]
y = df['energy_consumption']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(x_train, y_train)
joblib.dump(model, "forcast_model.pkl")

# Predictions and Error
y_pred = model.predict(x_test)
mse = mean_squared_error(y_test, y_pred)

st.success(f"✅ Model trained! Mean Squared Error on test set: {mse:.2f}")

# Input for Forecast
st.subheader("📌 Predict Your Energy Consumption")

with st.form("prediction_form"):
    temperature = st.number_input("Inside Temperature (°C)", value=22.0)
    outside_temperature = st.number_input("Outside Temperature (°C)", value=18.0)
    device_usage = st.number_input("Device Usage (%)", value=50.0)
    hour = st.slider("Hour of Day", 0, 23, 12)
    weekday = st.slider("Weekday (0=Mon, ..., 6=Sun)", 0, 6, 2)
    submitted = st.form_submit_button("Predict")

    if submitted:
        input_data = pd.DataFrame([[temperature, outside_temperature, device_usage, hour, weekday]],
                                  columns=features)
        loaded_model = joblib.load("forcast_model.pkl")
        prediction = loaded_model.predict(input_data)[0]
        st.success(f"⚡ Estimated Energy Consumption: {prediction:.2f} kWh")