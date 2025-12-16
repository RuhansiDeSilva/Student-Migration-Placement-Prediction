import streamlit as st
prediction, probability = predict_placement(
    placement_country,
    placement_company,
    visa_status,
    gpa,
    test_score,
    study_duration
)

st.set_page_config(
    page_title="Placement Prediction System",
    page_icon="🎓",
    layout="wide"
)

st.sidebar.title("🎓 Placement System")
st.sidebar.info(
    "A Machine Learning based system to predict student placement outcomes."
)

st.title("Welcome 👋")
st.markdown("""
### Student Migration & Placement Prediction System

This application helps predict whether a student is likely to be **placed**
based on academic performance and experience.

👉 Use the **Predict** page to get results
👉 Use **Model Info** to understand how predictions work
""")
