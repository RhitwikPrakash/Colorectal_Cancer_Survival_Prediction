import streamlit as st

from predictor import TREATMENT_CODES, load_artifacts, predict_survival


st.set_page_config(page_title="Colorectal Cancer Survival Prediction")
st.title("Colorectal Cancer Survival Prediction")
st.caption("Educational ML demonstration only — not a clinical prognosis or medical advice.")


@st.cache_resource
def cached_artifacts():
    return load_artifacts()


try:
    model, preprocessor = cached_artifacts()
except (OSError, ImportError, ValueError):
    st.error("The prediction model could not be loaded. Please check the deployment logs and dependencies.")
    st.stop()


with st.form("survival_prediction"):
    healthcare_costs = st.number_input("Healthcare costs (USD)", min_value=0.0, value=72000.0, step=1000.0)
    tumor_size = st.number_input("Tumor size (mm)", min_value=0.0, value=42.0, step=1.0)
    treatment = st.selectbox("Treatment type", options=list(TREATMENT_CODES), index=3)
    diabetes = st.selectbox("Diabetes", options=[0, 1], format_func=lambda value: "Yes" if value else "No")
    mortality_rate = st.number_input("Mortality rate per 100K", min_value=0.0, value=17.0, step=1.0)
    submitted = st.form_submit_button("Predict")

if submitted:
    try:
        prediction = predict_survival(
            model,
            preprocessor,
            healthcare_costs=healthcare_costs,
            tumor_size=tumor_size,
            treatment_type=TREATMENT_CODES[treatment],
            diabetes=diabetes,
            mortality_rate=mortality_rate,
        )
        st.success(f"Model prediction (survival): {prediction}")
    except (ValueError, TypeError):
        st.error("The model could not process those values. Please check your inputs.")
