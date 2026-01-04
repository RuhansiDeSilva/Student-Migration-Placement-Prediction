
import streamlit as st

st.title("📊 Model Information")

st.subheader("🧠 Machine Learning Model")
st.info("Random Forest Classifier")

st.subheader("✅ Why Random Forest?")
st.markdown("""
- Handles **non-linear patterns**
- Reduces **overfitting**
- Excellent for **tabular datasets**
""")

st.subheader("📌 Features Used")
st.markdown("""
- CGPA / Academic score
- Test scores
- Study duration
- Placement country
- Visa type
""")

st.warning("⚠ Predictions are probabilistic and not guarantees.")


