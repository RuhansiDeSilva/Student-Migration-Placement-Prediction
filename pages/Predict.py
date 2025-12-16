import streamlit as st
import pandas as pd
from utils.model_loader import load_model

st.title("🔮 Placement Prediction")

model = load_model()

st.markdown("### Enter Student Details")

with st.form("prediction_form"):
    cgpa = st.slider("CGPA", 0.0, 10.0, 7.0, 0.1)
    internships = st.number_input("Number of Internships", 0, 10, 1)
    projects = st.number_input("Number of Projects", 0, 10, 2)
    certifications = st.number_input("Certifications", 0, 10, 1)

    submitted = st.form_submit_button("Predict Placement")

if submitted:
    input_df = pd.DataFrame(
        [[cgpa, internships, projects, certifications]],
        columns=["cgpa", "internships", "projects", "certifications"]
    )

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.markdown("---")
    if prediction == 1:
        st.success("✅ Student is likely to be PLACED")
    else:
        st.error("❌ Student is NOT likely to be placed")

    st.info(f"📊 Placement Probability: **{probability*100:.2f}%**")
