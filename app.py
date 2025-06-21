import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import RobustScaler

# Load model and scaler
model = joblib.load("voting_model_for_final.pkl")
df = pd.read_csv("creditcard.csv")
scaler = RobustScaler()
df[['Time', 'Amount']] = scaler.fit_transform(df[['Time', 'Amount']])

# Mappings
device_map = {"Mobile": 0, "Web": 1, "ATM": 2}
location_map = {"Domestic": 0, "International": 1}
tx_type_map = {"POS": 0, "Online Transaction": 1, "Transfer Cash": 2, "Cash Withdrawal": 3}

st.title("💳 Credit Card Fraud Detection")

amount = st.number_input("Transaction Amount")
hour = st.slider("Hour of Transaction", 0, 23)
device = st.selectbox("Device Used", list(device_map.keys()))
location = st.selectbox("Transaction Location", list(location_map.keys()))
tenure = st.slider("Customer Tenure (Years)", 0, 20)
tx_type = st.selectbox("Transaction Type", list(tx_type_map.keys()))

if st.button("Predict Fraud Risk"):
    Time = hour * 3600
    Device = device_map[device]
    Location = location_map[location]
    TxType = tx_type_map[tx_type]

    V1 = np.log1p(amount) * (1 if Location == 0 else -1)
    V2 = np.sqrt(tenure) * (1 if Device == 0 else -1)
    V3 = amount / (tenure + 1)
    V4 = (Device + Location + TxType) * 0.5
    V10 = V1 - V4
    V12 = V2 + TxType
    V14 = V3 - Device

    input_data = pd.DataFrame([{
        "Time": Time,
        "Amount": amount,
        "V1": V1,
        "V2": V2,
        "V3": V3,
        "V4": V4,
        "V10": V10,
        "V12": V12,
        "V14": V14,
        "Hour": hour
    }])
    
    input_data[['Time', 'Amount']] = scaler.transform(input_data[['Time', 'Amount']])
    pred = model.predict(input_data)[0]
    confidence = model.predict_proba(input_data)[0][1] * 100

    if pred == 1:
        st.error(f"Fraud Detected! Confidence: {confidence:.2f}%")
    else:
        st.success(f"Safe Transaction. Confidence: {confidence:.2f}%")
