import streamlit as st
import pickle
import numpy as np
import os

st.set_page_config(
    page_title="Placement Prediction",
    layout="centered"
)

st.title("🎓 Placement Prediction App")

# -------- LOAD MODEL (CORRECT) --------
@st.cache_resource
def load_model():
    model_path = os.path.join(os.getcwd(), "final_random_forest_model.pkl")

    if not os.path.exists(model_path):
        st.error("❌ Model file not found: final_random_forest_model.pkl")
        st.stop()

    with open(model_path, "rb") as file:
        return pickle.load(file)

model = load_model()

# -------- INPUTS --------
cgpa = st.number_input("CGPA", 0.0, 10.0, step=0.01)
internships = st.number_input("Internships", 0, step=1)
projects = st.number_input("Projects", 0, step=1)
certifications = st.number_input("Certifications", 0, step=1)

# -------- PREDICTION --------
if st.button("Predict"):
    X = np.array([[cgpa, internships, projects, certifications]])
    prediction = model.predict(X)

    if prediction[0] == 1:
        st.success("✅ Student is likely to be Placed")
    else:
        st.error("❌ Student is unlikely to be Placed")
