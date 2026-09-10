import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import os
import joblib
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model


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

X = df.drop(columns=["isFraud"])


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


background = X_scaled[:100]


def predict_function(data):

    return model.predict(
        data,
        verbose=0
    ).flatten()


explainer = shap.Explainer(
    predict_function,
    background
)


sample_data = X_scaled[:5]

shap_values = explainer(
    sample_data
)


shap_values.feature_names = feature_names


print("SHAP explanation generated successfully!")


for i in range(len(sample_data)):

    probability = predict_function(
        sample_data[i:i+1]
    )[0]

    prediction = (
        "FRAUD"
        if probability >= 0.5
        else "NORMAL"
    )

    print()
    print("Transaction:", i + 1)
    print("Prediction:", prediction)
    print(
        "Fraud Probability:",
        round(float(probability), 4)
    )


shap.plots.waterfall(
    shap_values[0],
    max_display=10,
    show=False
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        BASE_PATH,
        "shap_waterfall.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()


shap.plots.bar(
    shap_values,
    max_display=10,
    show=False
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        BASE_PATH,
        "shap_feature_importance.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()


print()
print("SHAP plots saved successfully!")
print("shap_waterfall.png")
print("shap_feature_importance.png")