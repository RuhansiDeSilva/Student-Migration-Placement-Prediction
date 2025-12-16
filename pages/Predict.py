import joblib
import pandas as pd
import streamlit as st
from pathlib import Path

# -----------------------------
# Load Model (SAFE + CACHED)
# -----------------------------
@st.cache_resource
def load_model():
    model_path = Path(__file__).parent / "final_random_forest_model.joblib"
    return joblib.load(model_path)

model = load_model()

# -----------------------------
# Feature Template (MUST MATCH TRAINING)
# -----------------------------
FEATURE_COLUMNS = [
    # Placement country (top importance)
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

    # Numeric features
    "gpa_or_score",
    "test_score",
    "study_duration",

    # Visa
    "visa_status_Tier 4",
    "visa_status_Schengen Student Visa"
]

# -----------------------------
# Prediction Function
# -----------------------------
def predict_placement(
    placement_country,
    placement_company,
    visa_status,
    gpa,
    test_score,
    study_duration
):
    """
    Returns:
    - prediction (0 or 1)
    - probability (float)
    """

    # Create empty row
    input_data = pd.DataFrame(0, index=[0], columns=FEATURE_COLUMNS)

    # Encode categorical inputs
    country_col = f"placement_country_{placement_country}"
    company_col = f"placement_company_{placement_company}"
    visa_col = f"visa_status_{visa_status}"

    if country_col in input_data.columns:
        input_data[country_col] = 1

    if company_col in input_data.columns:
        input_data[company_col] = 1

    if visa_col in input_data.columns:
        input_data[visa_col] = 1

    # Numeric values
    input_data["gpa_or_score"] = gpa
    input_data["test_score"] = test_score
    input_data["study_duration"] = study_duration

    # Prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    return int(prediction), float(probability)
