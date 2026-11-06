"""Inference helpers for the trained, five-feature survival classifier."""

from pathlib import Path

import joblib
import pandas as pd


FEATURE_NAMES = (
    "Healthcare_Costs",
    "Tumor_Size_mm",
    "Treatment_Type",
    "Diabetes",
    "Mortality_Rate_per_100K",
)

# Training used sklearn LabelEncoder, which sorts labels alphabetically.
# These labels and their order were checked against artifacts/raw/data.csv.
TREATMENT_CODES = {
    "Chemotherapy": 0,
    "Combination": 1,
    "Radiotherapy": 2,
    "Surgery": 3,
}


def load_artifacts(repo_dir=None):
    """Load trusted model artifacts relative to the repository, not the process CWD."""
    root = Path(repo_dir) if repo_dir is not None else Path(__file__).resolve().parent
    model = joblib.load(root / "artifacts" / "model" / "model.pkl")
    scaler = joblib.load(root / "artifacts" / "model" / "preprocessor.pkl")

    if getattr(model, "n_features_in_", None) != len(FEATURE_NAMES):
        raise ValueError("Model feature count does not match the Streamlit inputs.")
    if tuple(getattr(scaler, "feature_names_in_", ())) != FEATURE_NAMES:
        raise ValueError("Saved scaler feature names/order do not match the Streamlit inputs.")

    return model, scaler


def predict_survival(
    model, scaler, *, healthcare_costs, tumor_size, treatment_type, diabetes, mortality_rate
):
    """Apply the saved scaler before prediction, preserving the training feature order."""
    input_row = pd.DataFrame(
        [[healthcare_costs, tumor_size, treatment_type, diabetes, mortality_rate]],
        columns=list(FEATURE_NAMES),
    )
    transformed = scaler.transform(input_row)
    return str(model.predict(transformed)[0])
