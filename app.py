import streamlit as st
import pickle
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Student Success Prediction Model",
    page_icon="🎓",
    layout="wide"
)

# Load the model
@st.cache_resource
def load_model():
    with open('final_random_forest_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

# Load the model
try:
    model = load_model()

    # Get feature names from the model
    if hasattr(model, 'feature_names_in_'):
        feature_names = model.feature_names_in_
    else:
        # Extract feature names from the one-hot encoded columns in the pickle file
        feature_names = [
            'scholarship_received', 'gpa_or_score', 'test_score',
            'origin_country_Finland', 'origin_country_Germany', 'origin_country_India',
            'origin_country_Ireland', 'origin_country_Russia', 'origin_country_South Africa',
            'origin_country_Uae', 'origin_country_United Kingdom', 'origin_country_United States',
            'destination_country_Finland', 'destination_country_Germany', 'destination_country_India',
            'destination_country_Ireland', 'destination_country_Russia', 'destination_country_South Africa',
            'destination_country_Uae', 'destination_country_United Kingdom', 'destination_country_United States',
            'destination_city_Abu Dhabi', 'destination_city_Bangalore', 'destination_city_Berlin',
            'destination_city_Birmingham', 'destination_city_Boston', 'destination_city_Cambridge',
            'destination_city_Cape Town', 'destination_city_Chennai', 'destination_city_Chicago',
            'destination_city_Cork', 'destination_city_Delhi', 'destination_city_Dubai',
            'destination_city_Dublin', 'destination_city_Durban', 'destination_city_Edinburgh',
            'destination_city_Edmonton', 'destination_city_Espoo', 'destination_city_Galway',
            'destination_city_Grahamstown', 'destination_city_Heidelberg', 'destination_city_Helsinki',
            'destination_city_Joensuu', 'destination_city_Johannesburg', 'destination_city_Kazan',
            'destination_city_Kingston', 'destination_city_Lappeenranta', 'destination_city_London',
            'destination_city_Los Angeles', 'destination_city_Manchester', 'destination_city_Maynooth',
            'destination_city_Montreal', 'destination_city_Moscow', 'destination_city_Mumbai',
            'destination_city_Munich', 'destination_city_New York', 'destination_city_Novosibirsk',
            'destination_city_Ottawa', 'destination_city_Oxford', 'destination_city_Pilani',
            'destination_city_Pretoria', 'destination_city_Saint Petersburg', 'destination_city_San Francisco',
            'destination_city_Sharjah', 'destination_city_Stellenbosch', 'destination_city_Stuttgart',
            'destination_city_Tampere', 'destination_city_Tiruchirappalli', 'destination_city_Toronto',
            'destination_city_Turku', 'destination_city_Vancouver',
            'university_name_Bauman Moscow State Technical University', 'university_name_Columbia University',
            'university_name_Delhi University', 'university_name_Dublin City University',
            'university_name_Harvard University', 'university_name_Heriot-Watt Dubai',
            'university_name_Higher School of Economics', 'university_name_Humboldt University',
            'university_name_IIT Delhi', 'university_name_Imperial College London', 'university_name_JNU Delhi',
            'university_name_Khalifa University', 'university_name_King\'s College London',
            'university_name_LMU Munich', 'university_name_Lomonosov Moscow State University',
            'university_name_MIT', 'university_name_Manipal Academy Dubai',
            'university_name_Middlesex University Dubai', 'university_name_Moscow Institute of Physics and Technology',
            'university_name_NYU', 'university_name_RWTH Aachen', 'university_name_Simon Fraser University',
            'university_name_Stanford University', 'university_name_TU Berlin',
            'university_name_Technical University of Munich', 'university_name_Technological University Dublin',
            'university_name_Trinity College Dublin', 'university_name_UC Berkeley', 'university_name_UCL',
            'university_name_University College Dublin', 'university_name_University of British Columbia',
            'university_name_University of Dubai', 'university_name_University of Johannesburg',
            'university_name_University of Turku', 'university_name_University of the Witwatersrand',
            'university_name_Western University', 'university_name_Zayed University',
            'university_name_Åbo Akademi University', 'course_name_Biotechnology',
            'course_name_Business Administration', 'course_name_Civil Engineering',
            'course_name_Computer Science', 'course_name_Data Science', 'course_name_Design',
            'course_name_Economics', 'course_name_Electrical Engineering', 'course_name_Finance',
            'course_name_Law', 'course_name_Mechanical Engineering', 'course_name_Medicine',
            'course_name_Political Science', 'course_name_Psychology', 'field_of_study_Business',
            'field_of_study_Computer Science', 'field_of_study_Engineering', 'field_of_study_Law',
            'field_of_study_Medicine', 'field_of_study_Natural Sciences', 'field_of_study_Social Sciences',
            'enrollment_reason_Job Opportunities', 'enrollment_reason_Political Stability',
            'enrollment_reason_Quality of Life', 'enrollment_reason_Scholarship',
            'placement_country_Finland', 'placement_country_Germany', 'placement_country_India',
            'placement_country_Ireland', 'placement_country_Russia', 'placement_country_South Africa',
            'placement_country_Uae', 'placement_country_United Kingdom', 'placement_country_United States',
            'placement_company_Apple', 'placement_company_Deloitte', 'placement_company_Facebook',
            'placement_company_Goldman Sachs', 'placement_company_Google', 'placement_company_IBM',
            'placement_company_Intel', 'placement_company_McKinsey', 'placement_company_Microsoft',
            'placement_company_SAP', 'placement_company_Siemens', 'placement_company_Tesla',
            'language_proficiency_test_IELTS', 'language_proficiency_test_PTE', 'language_proficiency_test_TOEFL',
            'visa_status_J1', 'visa_status_Schengen Student Visa', 'visa_status_Student Visa',
            'visa_status_Study Permit', 'visa_status_Tier 4', 'post_graduation_visa_OPT',
            'post_graduation_visa_PSW', 'post_graduation_visa_Post-Study Visa', 'post_graduation_visa_Work Permit',
            'study_duration'
        ]

except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Title and description
st.title("🎓 Student Success Prediction Model")
st.markdown("""
This application uses a RandomForestClassifier to predict student success outcomes based on various academic and demographic features.
""")

# Create tabs for different functionalities
tab1, tab2, tab3 = st.tabs(["📊 Make Prediction", "📈 Model Info", "🔄 Batch Prediction"])

with tab1:
    st.header("Single Prediction Interface")

    # Create two columns for input
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Academic Information")
        scholarship = st.selectbox("Scholarship Received", ["Yes", "No"])
        gpa_score = st.slider("GPA/Score", 0.0, 4.0, 3.0, 0.1)
        test_score = st.slider("Test Score", 0, 100, 75, 1)

        field_of_study = st.selectbox("Field of Study", [
            "Business", "Computer Science", "Engineering",
            "Law", "Medicine", "Natural Sciences", "Social Sciences"
        ])

        course_name = st.selectbox("Course Name", [
            "Biotechnology", "Business Administration", "Civil Engineering",
            "Computer Science", "Data Science", "Design", "Economics",
            "Electrical Engineering", "Finance", "Law", "Mechanical Engineering",
            "Medicine", "Political Science", "Psychology"
        ])

        study_duration = st.slider("Study Duration (years)", 1, 6, 3)

    with col2:
        st.subheader("Location & University")
        origin_country = st.selectbox("Origin Country", [
            "Finland", "Germany", "India", "Ireland", "Russia",
            "South Africa", "Uae", "United Kingdom", "United States"
        ])

        destination_country = st.selectbox("Destination Country", [
            "Finland", "Germany", "India", "Ireland", "Russia",
            "South Africa", "Uae", "United Kingdom", "United States"
        ])

        destination_city = st.selectbox("Destination City", [
            "Abu Dhabi", "Bangalore", "Berlin", "Birmingham", "Boston",
            "Cambridge", "Cape Town", "Chennai", "Chicago", "Cork",
            "Delhi", "Dubai", "Dublin", "Durban", "Edinburgh",
            "Edmonton", "Espoo", "Galway", "Grahamstown", "Heidelberg",
            "Helsinki", "Joensuu", "Johannesburg", "Kazan", "Kingston",
            "Lappeenranta", "London", "Los Angeles", "Manchester", "Maynooth",
            "Montreal", "Moscow", "Mumbai", "Munich", "New York",
            "Novosibirsk", "Ottawa", "Oxford", "Pilani", "Pretoria",
            "Saint Petersburg", "San Francisco", "Sharjah", "Stellenbosch",
            "Stuttgart", "Tampere", "Tiruchirappalli", "Toronto", "Turku",
            "Vancouver"
        ])

        university_name = st.selectbox("University Name", [
            "Bauman Moscow State Technical University", "Columbia University",
            "Delhi University", "Dublin City University", "Harvard University",
            "Heriot-Watt Dubai", "Higher School of Economics", "Humboldt University",
            "IIT Delhi", "Imperial College London", "JNU Delhi", "Khalifa University",
            "King's College London", "LMU Munich", "Lomonosov Moscow State University",
            "MIT", "Manipal Academy Dubai", "Middlesex University Dubai",
            "Moscow Institute of Physics and Technology", "NYU", "RWTH Aachen",
            "Simon Fraser University", "Stanford University", "TU Berlin",
            "Technical University of Munich", "Technological University Dublin",
            "Trinity College Dublin", "UC Berkeley", "UCL", "University College Dublin",
            "University of British Columbia", "University of Dubai", "University of Johannesburg",
            "University of Turku", "University of the Witwatersrand", "Western University",
            "Zayed University", "Åbo Akademi University"
        ])

        enrollment_reason = st.selectbox("Enrollment Reason", [
            "Job Opportunities", "Political Stability", "Quality of Life", "Scholarship"
        ])

    # Additional columns for placement and visa information
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Placement Information")
        placement_country = st.selectbox("Placement Country", [
            "Finland", "Germany", "India", "Ireland", "Russia",
            "South Africa", "Uae", "United Kingdom", "United States"
        ])

        placement_company = st.selectbox("Placement Company", [
            "Apple", "Deloitte", "Facebook", "Goldman Sachs", "Google",
            "IBM", "Intel", "McKinsey", "Microsoft", "SAP", "Siemens", "Tesla"
        ])

    with col4:
        st.subheader("Visa Information")
        language_test = st.selectbox("Language Proficiency Test", [
            "IELTS", "PTE", "TOEFL"
        ])

        visa_status = st.selectbox("Visa Status", [
            "J1", "Schengen Student Visa", "Student Visa", "Study Permit", "Tier 4"
        ])

        post_grad_visa = st.selectbox("Post-Graduation Visa", [
            "OPT", "PSW", "Post-Study Visa", "Work Permit"
        ])

    # Prediction button
    if st.button("🔮 Predict Success", type="primary"):
        try:
            # Create a dictionary with all features set to 0 initially
            input_data = {feature: 0 for feature in feature_names}

            # Set numeric features
            input_data['scholarship_received'] = 1 if scholarship == "Yes" else 0
            input_data['gpa_or_score'] = gpa_score
            input_data['test_score'] = test_score
            input_data['study_duration'] = study_duration

            # Set one-hot encoded features
            # Origin Country
            origin_country_feature = f"origin_country_{origin_country}"
            if origin_country_feature in input_data:
                input_data[origin_country_feature] = 1

            # Destination Country
            dest_country_feature = f"destination_country_{destination_country}"
            if dest_country_feature in input_data:
                input_data[dest_country_feature] = 1

            # Destination City
            dest_city_feature = f"destination_city_{destination_city}"
            if dest_city_feature in input_data:
                input_data[dest_city_feature] = 1

            # University Name
            university_feature = f"university_name_{university_name}"
            if university_feature in input_data:
                input_data[university_feature] = 1

            # Course Name
            course_feature = f"course_name_{course_name}"
            if course_feature in input_data:
                input_data[course_feature] = 1

            # Field of Study
            field_feature = f"field_of_study_{field_of_study}"
            if field_feature in input_data:
                input_data[field_feature] = 1

            # Enrollment Reason
            enrollment_feature = f"enrollment_reason_{enrollment_reason}"
            if enrollment_feature in input_data:
                input_data[enrollment_feature] = 1

            # Placement Country
            placement_country_feature = f"placement_country_{placement_country}"
            if placement_country_feature in input_data:
                input_data[placement_country_feature] = 1

            # Placement Company
            placement_company_feature = f"placement_company_{placement_company}"
            if placement_company_feature in input_data:
                input_data[placement_company_feature] = 1

            # Language Test
            language_feature = f"language_proficiency_test_{language_test}"
            if language_feature in input_data:
                input_data[language_feature] = 1

            # Visa Status
            visa_feature = f"visa_status_{visa_status}"
            if visa_feature in input_data:
                input_data[visa_feature] = 1

            # Post-Graduation Visa
            post_grad_feature = f"post_graduation_visa_{post_grad_visa}"
            if post_grad_feature in input_data:
                input_data[post_grad_feature] = 1

            # Convert to DataFrame
            input_df = pd.DataFrame([input_data])

            # Ensure all features are in correct order
            input_df = input_df[feature_names]

            # Make prediction
            prediction = model.predict(input_df)
            prediction_proba = model.predict_proba(input_df)

            # Display results
            st.subheader("🎯 Prediction Results")

            result_col1, result_col2, result_col3 = st.columns(3)

            with result_col1:
                st.metric("Predicted Class", f"{prediction[0]}")

            with result_col2:
                st.metric("Probability Class 0", f"{prediction_proba[0][0]:.2%}")

            with result_col3:
                st.metric("Probability Class 1", f"{prediction_proba[0][1]:.2%}")

            # Visual indicator
            st.progress(prediction_proba[0][1])
            st.caption(f"Confidence in positive prediction: {prediction_proba[0][1]:.2%}")

            # Interpretation
            if prediction[0] == 1:
                st.success("✅ High likelihood of student success")
            else:
                st.warning("⚠️ Potential challenges detected - consider intervention strategies")

            # Show feature importance (for top 10 features)
            if hasattr(model, 'feature_importances_'):
                st.subheader("📊 Top 10 Important Features for this Prediction")
                importances = model.feature_importances_
                feature_importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importances
                }).sort_values('Importance', ascending=False).head(10)

                st.dataframe(feature_importance_df, use_container_width=True)

        except Exception as e:
            st.error(f"Prediction error: {e}")

with tab2:
    st.header("Model Information")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Model Details")
        st.write(f"**Model Type:** {type(model).__name__}")
        st.write(f"**Number of Features:** {model.n_features_in_}")
        st.write(f"**Number of Estimators:** {model.n_estimators}")
        st.write(f"**Classes:** {model.classes_.tolist()}")
        st.write(f"**Number of Outputs:** {model.n_outputs_}")

    with col2:
        st.subheader("Model Parameters")
        params = model.get_params()
        param_df = pd.DataFrame(list(params.items()), columns=['Parameter', 'Value'])
        st.dataframe(param_df, use_container_width=True, hide_index=True)

    # Feature importance plot
    st.subheader("📈 Overall Feature Importance")

    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False).head(20)

        st.bar_chart(feature_importance_df.set_index('Feature')['Importance'])

        # Display as table
        with st.expander("View Detailed Feature Importance"):
            st.dataframe(feature_importance_df, use_container_width=True)
    else:
        st.info("Feature importances are not available for this model.")

with tab3:
    st.header("Batch Prediction")

    st.markdown("""
    Upload a CSV file with student data for batch predictions.
    The file should contain the same features as the model expects.
    """)

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            # Read the CSV file
            batch_data = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded data:")
            st.dataframe(batch_data.head(), use_container_width=True)

            # Check if all required features are present
            missing_features = [f for f in feature_names if f not in batch_data.columns]

            if missing_features:
                st.error(f"Missing features in uploaded data: {missing_features[:10]}{'...' if len(missing_features) > 10 else ''}")
            else:
                if st.button("🔮 Run Batch Predictions", type="primary"):
                    with st.spinner("Processing batch predictions..."):
                        # Reorder columns to match model expectations
                        batch_data_processed = batch_data[feature_names]

                        # Make predictions
                        predictions = model.predict(batch_data_processed)
                        predictions_proba = model.predict_proba(batch_data_processed)

                        # Add predictions to the dataframe
                        results_df = batch_data.copy()
                        results_df['Prediction'] = predictions
                        results_df['Probability_Class_0'] = predictions_proba[:, 0]
                        results_df['Probability_Class_1'] = predictions_proba[:, 1]

                        # Display results
                        st.subheader("📋 Batch Prediction Results")
                        st.dataframe(results_df, use_container_width=True)

                        # Summary statistics
                        st.subheader("📊 Summary Statistics")
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            success_rate = (results_df['Prediction'] == 1).mean()
                            st.metric("Success Rate", f"{success_rate:.2%}")

                        with col2:
                            avg_prob_success = results_df['Probability_Class_1'].mean()
                            st.metric("Avg Probability of Success", f"{avg_prob_success:.2%}")

                        with col3:
                            total_count = len(results_df)
                            st.metric("Total Records", total_count)

                        # Download button for results
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Predictions as CSV",
                            data=csv,
                            file_name="batch_predictions.csv",
                            mime="text/csv"
                        )

        except Exception as e:
            st.error(f"Error processing batch file: {e}")

# Footer
st.markdown("---")
st.markdown("""
**Note:** This is a demo application. The model predictions are based on the trained RandomForestClassifier.
Always verify predictions with domain experts before making decisions.
""")