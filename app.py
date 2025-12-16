import streamlit as st
from predict import predict_placement

st.set_page_config(
    page_title="Student Placement Prediction",
    layout="centered"
)

st.title("🎓 Student Placement Prediction System")
st.markdown("Predict whether a student will be **Placed or Not Placed**")

st.divider()

# ---------- INPUTS ----------
placement_country = st.selectbox(
    "Placement Country",
    [
        "India", "United Kingdom", "Ireland", "Germany", "Russia",
        "UAE", "Finland", "South Africa", "United States"
    ]
)

placement_company = st.selectbox(
    "Placement Company",
    [
        "Microsoft", "SAP", "Goldman Sachs", "Google", "IBM",
        "McKinsey", "Apple", "Deloitte", "Tesla", "Facebook"
    ]
)

visa_status = st.selectbox(
    "Visa Status",
    ["Tier 4", "Schengen Student Visa"]
)

gpa = st.number_input("GPA / Score", min_value=0.0, max_value=10.0, step=0.01)
test_score = st.number_input("Test Score", min_value=0, max_value=100)
study_duration = st.number_input("Study Duration (Years)", min_value=1, max_value=6)

st.divider()

# ---------- PREDICT ----------
if st.button("🔮 Predict Placement", use_container_width=True):
    prediction, probability = predict_placement(
        placement_country=placement_country,
        placement_company=placement_company,
        visa_status=visa_status,
        gpa=gpa,
        test_score=test_score,
        study_duration=study_duration
    )

    st.subheader("📌 Prediction Result")

    if prediction == 1:
        st.success("✅ **Placed**")
    else:
        st.error("❌ **Not Placed**")

    st.metric(
        label="Placement Probability",
        value=f"{probability * 100:.2f}%"
    )
