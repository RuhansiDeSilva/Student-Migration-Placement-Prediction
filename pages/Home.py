import streamlit as st

st.title("🏠 Home")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Model Type", "Random Forest")

with col2:
    st.metric("Dataset", "Student Migration")

with col3:
    st.metric("Prediction Type", "Binary")

st.divider()


st.markdown("""
### 📌 About This App
This system predicts **student placement outcomes** based on:
- Academic performance
- Experience
- Migration & visa details

👩‍🎓 **Target Users**
- Students
- Academic institutions
- Placement officers
""")
