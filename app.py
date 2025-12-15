import streamlit as st
import pickle
import numpy as np

# Load trained model
with open("final_random_forest_model", "rb") as file:
    model = pickle.load(file)

st.set_page_config(page_title="Placement Prediction", layout="centered")

st.title("🎓 Placement Prediction App")
st.write("Predict student placement status using ML")

# ---- Input Fields ----
cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, step=0.01)
internships = st.number_input("Number of Internships", min_value=0, step=1)
projects = st.number_input("Number of Projects", min_value=0, step=1)
certifications = st.number_input("Certifications Count", min_value=0, step=1)

# ---- Prediction ----
if st.button("Predict Placement"):
    input_data = np.array([[cgpa, internships, projects, certifications]])
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("✅ Student is likely to be Placed")
    else:
        st.error("❌ Student is unlikely to be Placed")
