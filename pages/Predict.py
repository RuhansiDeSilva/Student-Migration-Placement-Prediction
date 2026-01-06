import joblib
import pandas as pd
import streamlit as st
from pathlib import Path

# --------------------------------------------------
# Load Model (CORRECT PATH FOR /pages/)
# --------------------------------------------------
@st.cache_resource
def load_model():
    BASE_DIR = Path(__file__).resolve().parent.parent
    model_path = BASE_DIR / "final_random_forest_model.joblib"

    if not model_path.exists():
        st.error("❌ Model file not found. Please check deployment.")
        st.stop()

    return joblib.load(model_path)

model = load_model()

# --------------------------------------------------
# Feature Columns (MUST MATCH TRAINING)
# --------------------------------------------------
FEATURE_COLUMNS = [
    # Placement country
    "placement_country_India",
    "placement_country_United Kingdom",
    "placement_country_Ireland",
    "placement_country_Germany",
    "placement_country_Russia",
    "placement_country_UAE",
    "placement_country_Finland",
    "placement_country_South Africa",
    "placement_country_United States",

    # Placement company
    "placement_company_Microsoft",
    "placement_company_SAP",
    "placement_company_Goldman Sachs",
    "placement_company_Google",
    "placement_company_IBM",
    "placement_company_McKinsey",
    "placement_company_Apple",
    "placement_company_Deloitte",
    "placement_company_Tesla",
    "placement_company_Facebook",

    # Scores
    "gpa_or_score",
    "test_score",
    "study_duration",

    # Visa
    "visa_status_Tier 4",
    "visa_status_Schengen Student Visa"
]

# --------------------------------------------------
# UI
# --------------------------------------------------

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        placement_country = st.selectbox("🌍 Placement Country", [
            "India", "United Kingdom", "Ireland", "Germany",
            "Russia", "UAE", "Finland", "South Africa", "United States"
        ])

        visa_status = st.selectbox("🛂 Visa Status", [
            "Tier 4", "Schengen Student Visa"
        ])

        study_duration = st.number_input("📚 Study Duration (Years)", 1, 6)

    with col2:
        placement_company = st.selectbox("🏢 Placement Company", [
            "Microsoft", "SAP", "Goldman Sachs", "Google", "IBM",
            "McKinsey", "Apple", "Deloitte", "Tesla", "Facebook"
        ])

        gpa = st.number_input("🎓 GPA / Score", 0.0, 10.0, step=0.01)
        test_score = st.number_input("📝 Test Score", 0, 100)


# --------------------------------------------------
if st.button("🔮 Predict Placement", use_container_width=True):

    # Create input dataframe using model features
    input_df = pd.DataFrame(0, index=[0], columns=model.feature_names_in_)

    # Encode categorical values (only if column exists)
    country_col = f"placement_country_{placement_country}"
    if country_col in input_df.columns:
        input_df[country_col] = 1

    company_col = f"placement_company_{placement_company}"
    if company_col in input_df.columns:
        input_df[company_col] = 1

    visa_col = f"visa_status_{visa_status}"
    if visa_col in input_df.columns:
        input_df[visa_col] = 1

    # Numeric values
    input_df["gpa_or_score"] = gpa
    input_df["test_score"] = test_score
    input_df["study_duration"] = study_duration

    # Predict
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("📌 Prediction Result")

    if prediction == 1:
        st.success("✅ **PLACED**")
    else:
        st.error("❌ **NOT PLACED**")

    st.metric("Placement Probability", f"{probability * 100:.2f}%")
