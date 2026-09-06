from database import engine

def save_prediction(
    step,
    transaction_type,
    amount,
    oldbalanceOrg,
    newbalanceOrig,
    oldbalanceDest,
    newbalanceDest,
    prediction,
    fraud_probability
):
    query = """
    INSERT INTO transactions1 (
        step,
        type,
        amount,
        oldbalanceOrg,
        newbalanceOrig,
        oldbalanceDest,
        newbalanceDest,
        prediction,
        fraud_probability
    )
    VALUES (
        %(step)s,
        %(type)s,
        %(amount)s,
        %(oldbalanceOrg)s,
        %(newbalanceOrig)s,
        %(oldbalanceDest)s,
        %(newbalanceDest)s,
        %(prediction)s,
        %(fraud_probability)s
    )
    """

    with engine.begin() as conn:
        conn.exec_driver_sql(
            query,
            {
                "step": step,
                "type": transaction_type,
                "amount": amount,
                "oldbalanceOrg": oldbalanceOrg,
                "newbalanceOrig": newbalanceOrig,
                "oldbalanceDest": oldbalanceDest,
                "newbalanceDest": newbalanceDest,
                "prediction": prediction,
                "fraud_probability": fraud_probability
            }
        )

    print("Prediction saved to PostgreSQL!")
# Test
save_prediction(
    step=1,
    transaction_type="TRANSFER",
    amount=10000,
    oldbalanceOrg=15000,
    newbalanceOrig=5000,
    oldbalanceDest=2000,
    newbalanceDest=12000,
    prediction=1,
    fraud_probability=0.93
)