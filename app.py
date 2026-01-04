import streamlit as st

st.set_page_config(
    page_title="Placement Prediction System",
    page_icon="🎓",
    layout="wide"
)

st.sidebar.title("🎓 Placement System")
st.sidebar.info(
    "A Machine Learning based system to predict student placement outcomes."
)

st.title("🎓 Placement System")
st.sidebar.info(
    "A Machine Learning based system to predict student placement outcomes."
)
st.markdown(
 """
    <h1 style='text-align:center;'>🎓 Student Migration & Placement Prediction</h1>
    <p style='text-align:center;font-size:18px;'>
    Predict international student placement outcomes using Machine Learning
    </p>
    """,
    unsafe_allow_html=True
    )

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.success("📊 Data-Driven Decisions")

with col2:
    st.info("🧠 Random Forest Model")

with col3:
    st.warning("🌍 Global Opportunities")

st.divider()

st.markdown(
    """
    ### 🔍 How to Use
    - Go to **Predict** → Enter student details
    - View **placement probability**
    - Learn about the model in **Model Info**
    """
)