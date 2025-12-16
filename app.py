import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

st.set_page_config(page_title="Placement Prediction", layout="centered")
st.title("🎓 Placement Prediction App")

@st.cache_resource
def load_model():
    path = "final_random_forest_model.joblib"
    if not os.path.exists(path):
        st.error("❌ Model file not found")
        st.stop()
    return joblib.load(path)

model = load_model()

cgpa = st.number_input("CGPA", 0.0, 10.0, step=0.01)
internships = st.number_input("Internships", 0)
projects = st.number_input("Projects", 0)
certifications = st.number_input("Certifications", 0)

if st.button("Predict"):
    X = pd.DataFrame([[cgpa, internships, projects, certifications]],
                     columns=["cgpa", "internships", "projects", "certifications"])
    pred = model.predict(X)[0]

    if pred == 1:
        st.success("✅ Student is likely to be Placed")
    else:
        st.error("❌ Student is unlikely to be Placed")
