import streamlit as st
import joblib
import numpy as np

# Load model and preprocessor using joblib
model = joblib.load("artifacts/model/model.pkl")
preprocessor = joblib.load("artifacts/model/preprocessor.pkl")

# Streamlit UI
st.title("Colorectal Cancer Survival Prediction")

# Input fields
healthcare_costs = st.number_input("Healthcare Costs", min_value=0.0, format="%.2f")
tumor_size = st.number_input("Tumor Size (mm)", min_value=0.0, format="%.2f")
treatment_type = st.selectbox("Treatment Type", options=[0, 1, 2])
diabetes = st.selectbox("Diabetes", options=[0, 1])
mortality_rate = st.number_input("Mortality Rate per 100K", min_value=0.0, format="%.2f")

# Prediction
if st.button("Predict"):
    input_data = np.array([[healthcare_costs, tumor_size, treatment_type, diabetes, mortality_rate]])
    transformed_data = preprocessor.transform(input_data)
    prediction = model.predict(transformed_data)[0]
    st.success(f"Predicted Outcome: {prediction}")
