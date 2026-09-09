# AML / Fraud Detection System Using Deep Learning

## 📌 Project Overview

This project is an **Anti-Money Laundering (AML) and Fraud Detection System** that uses **Deep Learning** to identify whether a financial transaction is **Normal or Fraudulent**.

The system takes transaction details as input, preprocesses the data, uses a trained Deep Learning model to predict fraud probability, and stores the prediction results in a **PostgreSQL database**.

The project also includes **FastAPI** for providing a REST API through which users can submit transaction details and receive predictions.

---

## 🎯 Objectives

The main objectives of this project are:

* To understand and preprocess financial transaction data.
* To identify fraudulent transactions using Deep Learning.
* To handle the highly imbalanced nature of fraud data.
* To evaluate the performance of the Deep Learning model.
* To save the trained model and preprocessing objects.
* To provide fraud prediction through a FastAPI REST API.
* To store transaction predictions and fraud probabilities in PostgreSQL.

---

## 🛠️ Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **TensorFlow / Keras**
* **Deep Learning / Artificial Neural Network (ANN)**
* **FastAPI**
* **Pydantic**
* **PostgreSQL**
* **SQLAlchemy**
* **Joblib**
* **Jupyter Notebook**
* **VS Code**
* **Git & GitHub**

---

## 📂 Project Structure

```text
AML_DeepLearning_Project/
│
├── api/
│   └── main.py
│
├── database/
│   ├── database.py
│   └── save_prediction.py
│
├── dataset/
│   └── Dataset files
│
├── models/
│   ├── fraud_detection_model.keras
│   ├── scaler.pkl
│   └── feature_names.pkl
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_data_cleaning.ipynb
│   └── 05_deep_learning.ipynb
│
├── README.md
└── .gitignore
```

> The large dataset and virtual environment are not included in the GitHub repository.

---

## 📊 Dataset

The project uses financial transaction data for AML and fraud detection.

### Important Features

| Feature          | Description                                                                |
| ---------------- | -------------------------------------------------------------------------- |
| `step`           | Represents the time step of the transaction                                |
| `type`           | Type of transaction such as PAYMENT, TRANSFER, CASH_OUT, CASH_IN and DEBIT |
| `amount`         | Transaction amount                                                         |
| `oldbalanceOrg`  | Sender's balance before the transaction                                    |
| `newbalanceOrig` | Sender's balance after the transaction                                     |
| `oldbalanceDest` | Receiver's balance before the transaction                                  |
| `newbalanceDest` | Receiver's balance after the transaction                                   |
| `isFraud`        | Target variable: 0 = Normal, 1 = Fraud                                     |
| `isFlaggedFraud` | Original rule-based fraud flag                                             |

The target variable used for Deep Learning is:

```text
isFraud
```

where:

```text
0 → Normal Transaction
1 → Fraudulent Transaction
```

---

## 🔄 Project Workflow

```text
Raw Dataset
     ↓
Data Understanding
     ↓
Data Cleaning
     ↓
Feature Engineering
     ↓
Train/Test Split
     ↓
Categorical Encoding
     ↓
Feature Scaling
     ↓
Class Imbalance Handling
     ↓
Deep Learning Model
     ↓
Model Evaluation
     ↓
  FastAPI
     ↓
PostgreSQL
```

---

# 📘 1. Data Understanding

The Data Understanding stage was performed to understand the structure and quality of the dataset.

The following were checked:

* Dataset shape
* Column names
* Data types
* Missing values
* Duplicate records
* Target variable distribution
* Fraud and normal transaction counts

Notebook:

```text
notebooks/01_data_understanding.ipynb
```

---

# 🧹 2. Data Cleaning

The dataset was cleaned and prepared for further processing.

The cleaning and preparation process includes:

* Handling duplicate records
* Checking missing values
* Cleaning the transaction data
* Preparing features and target variable
* Preparing data for Deep Learning

Notebook:

```text
notebooks/02_data_cleaning.ipynb
```

---

# 🧠 3. Deep Learning Model

A **Deep Learning Artificial Neural Network (ANN)** was developed using TensorFlow/Keras.

### Model Architecture

```text
Input Layer
     ↓
Dense Layer - 64 neurons - ReLU
     ↓
Dropout - 30%
     ↓
Dense Layer - 32 neurons - ReLU
     ↓
Dropout - 20%
     ↓
Dense Layer - 16 neurons - ReLU
     ↓
Output Layer - 1 neuron - Sigmoid
```

### Model Compilation

The model uses:

* **Optimizer:** Adam
* **Loss Function:** Binary Cross-Entropy
* **Output Activation:** Sigmoid

The model predicts a probability between:

```text
0 and 1
```

A threshold of `0.5` is used:

```text
Probability >= 0.5 → FRAUD
Probability < 0.5  → NORMAL
```

Notebook:

```text
notebooks/05_deep_learning.ipynb
```

---

# ⚖️ 4. Class Imbalance

Fraudulent transactions are much fewer than normal transactions.

Therefore, class imbalance needs to be handled carefully.

The project uses **class weights** during Deep Learning model training so that the model gives more importance to the minority fraud class.

This helps reduce the possibility of the model simply predicting every transaction as normal.

---

# ⏹️ 5. Early Stopping

**Early Stopping** is used during training.

It monitors validation loss and stops training when the model stops improving.

This helps:

* Reduce unnecessary training
* Reduce overfitting
* Restore the best model weights

---

# 📈 6. Model Evaluation

The Deep Learning model is evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix
* ROC-AUC

### Why these metrics?

Accuracy alone is not sufficient for fraud detection because the dataset is highly imbalanced.

Therefore, **Precision, Recall, F1-score and ROC-AUC** are also considered.

#

---

# 🚀 7. FastAPI

FastAPI is used to expose the Deep Learning model through a REST API.

The API accepts transaction information in JSON format.

Example:

```json
{
  "step": 1,
  "type": "PAYMENT",
  "amount": 1000,
  "oldbalanceOrg": 10000,
  "newbalanceOrig": 9000,
  "oldbalanceDest": 5000,
  "newbalanceDest": 6000
}
```

The API then:

```text
JSON Input
    ↓
Validation
    ↓
DataFrame Conversion
    ↓
Transaction Type Encoding
    ↓
Feature Matching
    ↓
Scaling
    ↓
Deep Learning Prediction
    ↓
Fraud Probability
    ↓
FRAUD / NORMAL
```

---

# 🗄️ 8. PostgreSQL Database

PostgreSQL is used to store prediction results generated by the API.

The stored information includes:

* Transaction step
* Transaction type
* Transaction amount
* Sender balance before transaction
* Sender balance after transaction
* Receiver balance before transaction
* Receiver balance after transaction
* Prediction
* Fraud probability

The database table used by the API is:

```text
transactions1
```

This allows transaction prediction history to be stored and retrieved later.

---

# 💾 9. Saved Model Files

The trained Deep Learning model is saved as:

```text
models/fraud_detection_model.keras
```

The feature scaler is saved as:

```text
models/scaler.pkl
```

The training feature names are saved as:

```text
models/feature_names.pkl
```

These files allow the FastAPI application to use the same trained model and preprocessing process during prediction.

---

# 🧪 10. API Testing

The API can be tested using **FastAPI Swagger UI**.

Start the API using:

```bash
uvicorn api.main:app --reload
```

Then open:

```text
http://127.0.0.1:8000/docs
```

The `/predict` endpoint can be tested using the **Try it out** option.

The API returns:

```json
{
  "prediction": "NORMAL",
  "fraud_probability": 0.02,
  "database_status": "Prediction saved successfully"
}
```

The prediction is also stored in PostgreSQL.

---

# 🧩 11. Complete System Architecture

```text
                 Financial Transaction
                         │
                         ▼
                  ┌─────────────┐
                  │  FastAPI    │
                  └──────┬──────┘
                         │
                         ▼
                 Input Validation
                         │
                         ▼
                 Data Preprocessing
                         │
                         ▼
                      Scaler
                         │
                         ▼
              ┌────────────────────┐
              │ Deep Learning ANN   │
              └─────────┬──────────┘
                        │
                        │
                        ▼                  
                Fraud Probability      
                      │            
                      ▼
                 FRAUD / NORMAL
                      │
                      ▼
                 PostgreSQL
              Prediction History
```

---

# 🔐 12. Security

Sensitive information such as database passwords should not be hard-coded or uploaded to GitHub.

Environment variables such as `.env` should be excluded using `.gitignore`.

The `.env` file should **never be committed to GitHub**.

---

# 📁 13. GitHub

This repository contains the source code, notebooks, model-related files and project documentation required to understand and run the project.

Large datasets and the Python virtual environment are excluded from the repository.

---

# ✅ Project Status

* [x] Data Understanding
* [x] Data Cleaning
* [x] Feature Engineering
* [x] Deep Learning Model
* [x] Class Imbalance Handling
* [x] Model Evaluation
* [x] Model Saving
* [x] FastAPI Integration
* [x] PostgreSQL Integration
* [x] API Testing

---

# 🎓 Conclusion

This project demonstrates an end-to-end **AML/Fraud Detection System using Deep Learning**.

The Deep Learning model identifies potentially fraudulent transactions, **SHAP provides explanations for model predictions**, FastAPI provides an interface for real-time prediction, and PostgreSQL stores transaction and prediction history.

The combination of **Deep Learning, Explainable AI, REST API and Database Integration** makes the system suitable as an end-to-end academic AML/Fraud Detection project.

---

## 👩‍💻 Author

**Hiral Thakar**

B.Sc. Data Science
