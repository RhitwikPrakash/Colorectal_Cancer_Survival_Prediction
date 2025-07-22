import joblib

scaler = joblib.load("artifacts/processed/scaler.pkl")
joblib.dump(scaler, "artifacts/model/preprocessor.pkl")
