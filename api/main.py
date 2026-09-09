from fastapi import FastAPI
from pydantic import BaseModel
from tensorflow.keras.models import load_model
import pandas as pd
import joblib
import os

from database.save_prediction import save_prediction


# =========================================================
# FASTAPI APPLICATION
# =========================================================

app = FastAPI(
    title="AML Fraud Detection API",
    description="Deep Learning based Fraud Detection API",
    version="1.0"
)


# =========================================================
# MODEL FILE PATHS
# =========================================================

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


# =========================================================
# LOAD MODEL, SCALER AND FEATURE NAMES
# =========================================================

model = load_model(MODEL_PATH)

scaler = joblib.load(SCALER_PATH)

feature_names = joblib.load(FEATURE_PATH)


print("Model loaded successfully!")
print("Scaler loaded successfully!")
print("Feature names loaded successfully!")


# =========================================================
# TRANSACTION INPUT MODEL
# =========================================================

class Transaction(BaseModel):

    step: int

    type: str

    amount: float

    oldbalanceOrg: float

    newbalanceOrig: float

    oldbalanceDest: float

    newbalanceDest: float


# =========================================================
# HOME ENDPOINT
# =========================================================

@app.get("/")
def home():

    return {
        "message": "AML Fraud Detection API is running",
        "model": "Deep Learning",
        "status": "ready"
    }


# =========================================================
# PREDICTION ENDPOINT
# =========================================================

@app.post("/predict")
def predict(transaction: Transaction):

    # -----------------------------------------------------
    # 1. Convert input data into dictionary
    # -----------------------------------------------------

    data = transaction.model_dump()


    # -----------------------------------------------------
    # 2. Convert dictionary into DataFrame
    # -----------------------------------------------------

    df = pd.DataFrame([data])


    # -----------------------------------------------------
    # 3. Convert transaction type into numerical columns
    # -----------------------------------------------------

    df = pd.get_dummies(
        df,
        columns=["type"],
        dtype=int
    )


    # -----------------------------------------------------
    # 4. Match training feature columns
    # -----------------------------------------------------

    df = df.reindex(
        columns=feature_names,
        fill_value=0
    )


    # -----------------------------------------------------
    # 5. Scale input data
    # -----------------------------------------------------

    scaled_data = scaler.transform(df)


    # -----------------------------------------------------
    # 6. Predict using Deep Learning model
    # -----------------------------------------------------

    probability = float(
        model.predict(
            scaled_data,
            verbose=0
        )[0][0]
    )


    # -----------------------------------------------------
    # 7. Convert probability into prediction
    #
    #    0 = NORMAL
    #    1 = FRAUD
    # -----------------------------------------------------

    if probability >= 0.5:

        prediction = "FRAUD"

        # Integer value for PostgreSQL
        prediction_value = 1

    else:

        prediction = "NORMAL"

        # Integer value for PostgreSQL
        prediction_value = 0


    # -----------------------------------------------------
    # 8. Save prediction into PostgreSQL
    # -----------------------------------------------------

    save_prediction(

        step=transaction.step,

        transaction_type=transaction.type,

        amount=transaction.amount,

        oldbalanceOrg=transaction.oldbalanceOrg,

        newbalanceOrig=transaction.newbalanceOrig,

        oldbalanceDest=transaction.oldbalanceDest,

        newbalanceDest=transaction.newbalanceDest,

        prediction=prediction_value,

        fraud_probability=probability
    )


    # -----------------------------------------------------
    # 9. Return response
    # -----------------------------------------------------

    return {

        "prediction": prediction,

        "fraud_probability": round(
            probability,
            4
        ),

        "database_status":
            "Prediction saved successfully"
    }