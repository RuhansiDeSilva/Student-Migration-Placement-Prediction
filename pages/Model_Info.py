import streamlit as st

st.title("📊 Model Information")

st.markdown("""
### Model Used
- **Random Forest Classifier**

### Why Random Forest?
- Handles non-linear data
- Reduces overfitting
- Works well with tabular datasets

### Features Used
- CGPA
- Internships
- Projects
- Certifications

### Disclaimer
Predictions are probabilistic and should be used as **decision support**, not guarantees.
""")
