import joblib
import os
import streamlit as st

@st.cache_resource
def load_model():
    model_path = "final_random_forest_model.joblib"

    if not os.path.exists(model_path):
        st.error("❌ Model file not found. Please contact administrator.")
        st.stop()

    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error("❌ Failed to load ML model")
        st.exception(e)
        st.stop()
