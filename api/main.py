from fastapi import FastAPI
from pydantic import BaseModel
from tensorflow.keras.models import load_model
import pandas as pd
import joblib
import os


# ==========================================
# 1. Create FastAPI application
# ==========================================

app = FastAPI(
    title="AML Fraud Detection API",
    description="Deep Learning based Fraud Detection API",
    version="1.0"
)


# ==========================================
# 2. File paths
# ==========================================

MODEL_PATH = os.path.join(
    "models",
    "fraud_detection_model.keras"
)

SCALER_PATH = os.path.join(
    "models",
    "scaler.pkl"
)

FEATURE_PATH = os.path.join(
    "models",
    "feature_names.pkl"
)


# ==========================================
# 3. Load trained model and preprocessing
# ==========================================

model = load_model(MODEL_PATH)

scaler = joblib.load(
    SCALER_PATH
)

feature_names = joblib.load(
    FEATURE_PATH
)

print("Model loaded successfully!")
print("Scaler loaded successfully!")
print("Feature names loaded successfully!")


# ==========================================
# 4. Transaction input model
# ==========================================

class Transaction(BaseModel):

    step: int

    type: str

    amount: float

    oldbalanceOrg: float

    newbalanceOrig: float

    oldbalanceDest: float

    newbalanceDest: float


# ==========================================
# 5. Home endpoint
# ==========================================

@app.get("/")
def home():

    return {
        "message": "AML Fraud Detection API is running",
        "model": "Deep Learning",
        "status": "ready"
    }


# ==========================================
# 6. Prediction endpoint
# ==========================================

@app.post("/predict")
def predict(transaction: Transaction):

    # Convert transaction to dictionary
    data = transaction.model_dump()

    # Convert dictionary to DataFrame
    df = pd.DataFrame([data])

    # One-hot encode transaction type
    df = pd.get_dummies(
        df,
        columns=["type"],
        dtype=int
    )

    # Make sure all training features exist
    df = df.reindex(
        columns=feature_names,
        fill_value=0
    )

    # Scale the data
    scaled_data = scaler.transform(df)

    # Get fraud probability
    probability = float(
        model.predict(
            scaled_data,
            verbose=0
        )[0][0]
    )

    # Convert probability into prediction
    if probability >= 0.5:
        prediction = "FRAUD"
    else:
        prediction = "NORMAL"

    return {
        "prediction": prediction,
        "fraud_probability": round(probability, 4)
    }