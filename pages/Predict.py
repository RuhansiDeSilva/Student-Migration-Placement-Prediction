import joblib
import pandas as pd
import os

# ---------------- LOAD MODEL ----------------
MODEL_PATH = "final_random_forest_model.joblib"

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model file not found.")

    return joblib.load(MODEL_PATH)


model = load_model()

# ---------------- FEATURE LIST ----------------
FEATURES = [
    # Placement Country
    "placement_country_India",
    "placement_country_United Kingdom",
    "placement_country_Ireland",
    "placement_country_Germany",
    "placement_country_Russia",
    "placement_country_UAE",
    "placement_country_Finland",
    "placement_country_South Africa",
    "placement_country_United States",

    # Placement Company
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

    # Numeric
    "gpa_or_score",
    "test_score",
    "study_duration",

    # Visa
    "visa_status_Tier 4",
    "visa_status_Schengen Student Visa"
]

# ---------------- PREDICTION FUNCTION ----------------
def predict_placement(
    placement_country: str,
    placement_company: str,
    visa_status: str,
    gpa: float,
    test_score: int,
    study_duration: int
):
    """
    Returns:
        prediction (int): 1 = Placed, 0 = Not Placed
        probability (float): probability of placement
    """

    # Initialize all features to 0
    input_data = dict.fromkeys(FEATURES, 0)

    # One-hot encoding
    input_data[f"placement_country_{placement_country}"] = 1
    input_data[f"placement_company_{placement_company}"] = 1
    input_data[f"visa_status_{visa_status}"] = 1

    # Numeric values
    input_data["gpa_or_score"] = gpa
    input_data["test_score"] = test_score
    input_data["study_duration"] = study_duration

    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])

    # Prediction
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    return prediction, probability
