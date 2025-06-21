from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import RobustScaler
import uvicorn

# Initialize app
app = FastAPI(title="Credit Card Fraud Detection API", description="Predicts fraud risk for credit card transactions.", version="1.0")

# Load model and scaler
model = joblib.load("voting_model_for_final.pkl")
scaler = RobustScaler()
df = pd.read_csv("creditcard.csv")
df[['Time', 'Amount']] = scaler.fit_transform(df[['Time', 'Amount']])

# Mappings
DEVICE_MAP = {"Mobile": 0, "Web": 1, "ATM": 2}
LOCATION_MAP = {"Domestic": 0, "International": 1}
TX_TYPE_MAP = {"POS": 0, "Online Transaction": 1, "Transfer Cash": 2, "Cash Withdrawal": 3}

# Pydantic model for input validation
class TransactionInput(BaseModel):
    amount: float
    hour: int
    device: str
    location: str
    tenure: int
    tx_type: str

@app.get("/")
def root():
    return {"message": "Credit Card Fraud Detection API is running."}

@app.post("/predict")
def predict(input: TransactionInput):
    try:
        # Feature engineering
        Time = input.hour * 3600
        Device = DEVICE_MAP.get(input.device)
        Location = LOCATION_MAP.get(input.location)
        TxType = TX_TYPE_MAP.get(input.tx_type)

        if None in [Device, Location, TxType]:
            raise HTTPException(status_code=400, detail="Invalid categorical input values.")

        V1 = np.log1p(input.amount) * (1 if Location == 0 else -1)
        V2 = np.sqrt(input.tenure) * (1 if Device == 0 else -1)
        V3 = input.amount / (input.tenure + 1)
        V4 = (Device + Location + TxType) * 0.5
        V10 = V1 - V4
        V12 = V2 + TxType
        V14 = V3 - Device

        input_df = pd.DataFrame([{
            "Time": Time,
            "Amount": input.amount,
            "V1": V1,
            "V2": V2,
            "V3": V3,
            "V4": V4,
            "V10": V10,
            "V12": V12,
            "V14": V14,
            "Hour": input.hour
        }])

        input_df[['Time', 'Amount']] = scaler.transform(input_df[['Time', 'Amount']])

        prediction = model.predict(input_df)[0]
        confidence = model.predict_proba(input_df)[0][1] * 100

        return {
            "prediction": "Fraudulent" if prediction == 1 else "Safe",
            "confidence": f"{confidence:.2f}%"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Uncomment this to run locally (only for testing)
# if __name__ == "__main__":
#     uvicorn.run("fastapi_fraud_api:app", host="127.0.0.1", port=8000, reload=True)
