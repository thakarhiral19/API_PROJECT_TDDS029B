from sqlalchemy import create_engine

DATABASE_URL = (
    "postgresql+psycopg2://postgres:Hiral%4012345"
    "@localhost:5432/aml_fraud_db"
)

engine = create_engine(DATABASE_URL)

try:
    with engine.connect() as conn:
        print("Database Connected Successfully!")
except Exception as e:
    print("Database connection failed:")
    print(e)