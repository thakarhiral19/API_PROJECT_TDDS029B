import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import os
import joblib
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import precision_score, recall_score, confusion_matrix


BASE_PATH = r"C:\Users\Hiral\OneDrive\Desktop\AML_DeepLearning_Project"

MODEL_PATH = os.path.join(
    BASE_PATH,
    "models",
    "fraud_detection_model.keras"
)

SCALER_PATH = os.path.join(
    BASE_PATH,
    "models",
    "scaler.pkl"
)

FEATURE_PATH = os.path.join(
    BASE_PATH,
    "models",
    "feature_names.pkl"
)

DATA_PATH = os.path.join(
    BASE_PATH,
    "dataset",
    "final_features.csv"
)


model = load_model(MODEL_PATH)

scaler = joblib.load(SCALER_PATH)

feature_names = joblib.load(FEATURE_PATH)

df = pd.read_csv(DATA_PATH)


y = df["isFraud"]

X = df.drop(columns=["isFraud"])


transaction_type = X["type"].copy()


X = pd.get_dummies(
    X,
    columns=["type"],
    dtype=int
)


X = X.reindex(
    columns=feature_names,
    fill_value=0
)


X_scaled = scaler.transform(X)


probability = model.predict(
    X_scaled,
    verbose=0
).flatten()


y_pred = (
    probability >= 0.5
).astype(int)


results = pd.DataFrame({
    "type": transaction_type,
    "actual": y,
    "predicted": y_pred
})


print()
print("FAIRNESS ANALYSIS")
print("=================")


for transaction_type_name in results["type"].unique():

    group = results[
        results["type"] == transaction_type_name
    ]

    actual = group["actual"]

    predicted = group["predicted"]


    precision = precision_score(
        actual,
        predicted,
        zero_division=0
    )


    recall = recall_score(
        actual,
        predicted,
        zero_division=0
    )


    cm = confusion_matrix(
        actual,
        predicted,
        labels=[0, 1]
    )


    tn, fp, fn, tp = cm.ravel()


    if (fp + tn) > 0:

        false_positive_rate = fp / (fp + tn)

    else:

        false_positive_rate = 0


    print()
    print("Transaction Type:", transaction_type_name)

    print(
        "Number of Transactions:",
        len(group)
    )

    print(
        "Actual Fraud:",
        int(actual.sum())
    )

    print(
        "Precision:",
        round(precision, 4)
    )

    print(
        "Recall:",
        round(recall, 4)
    )

    print(
        "False Positive Rate:",
        round(false_positive_rate, 4)
    )


print()
print("Fairness analysis completed successfully!")