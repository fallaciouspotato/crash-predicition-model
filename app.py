import streamlit as st
import pandas as pd
import joblib
from streamlit_option_menu import option_menu

# -----------------
# Page Configuration
# -----------------
st.set_page_config(
    page_title="Aviation Damage Predictor",
    page_icon="✈️",
    layout="wide",
)

st.text("By Team 2 LPA")
st.markdown("<h1 style='text-align: center; font-weight: bold;'>AVIATION DAMAGE PREDICTION SYSTEM</h1>", unsafe_allow_html=True)

# -----------------
# Load Pre-trained Model and Encoders (very fast)
# -----------------
@st.cache_resource
def load_model_and_encoders():
    model = joblib.load('aviation_model.joblib')
    encoders = joblib.load('aviation_encoders.joblib')
    return model, encoders

model, encoders = load_model_and_encoders()

# -----------------
# Navigation Bar
# -----------------
selected = option_menu(
    menu_title=None,
    options=["Home", "Live Prediction", "Crash Case Studies"],
    icons=["house", "rocket-launch", "journal-text"],
    menu_icon="cast", default_index=0, orientation="horizontal",
)

# -----------------
# Page Content
# -----------------
if selected == "Home":
    st.subheader("A Machine Learning project to enhance aviation safety through data.")
    st.write("Welcome! This application uses a pre-trained machine learning model to predict aircraft damage severity.")
    st.info("Navigate to the **Live Prediction** tab to test the model.")

elif selected == "Live Prediction":
    st.subheader("Enter incident details below to get a real-time prediction.")
    st.markdown("---")

    input_features = {}
    col1, col2, col3 = st.columns(3)

    # Use hardcoded lists for dropdowns to avoid loading CSV in the app
    with col1:
        input_features['Make'] = st.selectbox("Aircraft Make", options=['', 'Boeing', 'Airbus', 'Cessna', 'Piper', 'McDonnell Douglas'])
        input_features['Model'] = st.text_input("Aircraft Model (e.g., 737-800, A320, 172)")
        input_features['Engine_Type'] = st.selectbox("Engine Type", options=['', 'Reciprocating', 'Turbofan', 'Turbojet', 'Turboprop', 'Unknown'])
        input_features['Country'] = st.text_input("Country (e.g., United States, India)")

    with col2:
        input_features['Weather_Condition'] = st.selectbox("Weather Condition", options=['', 'VMC', 'IMC', 'Unknown'])
        input_features['Broad_phase_of_flight'] = st.selectbox("Phase of Flight", options=['', 'Takeoff', 'Cruise', 'Landing', 'Approach', 'Maneuvering'])
        input_features['Purpose_of_flight'] = st.selectbox("Purpose of Flight", options=['', 'Instructional', 'Personal', 'Aerial Application', 'Positioning'])
        input_features['Number_of_Engines'] = st.number_input('Number of Engines', min_value=0, max_value=8, value=2)

    with col3:
        input_features['Year'] = st.number_input('Year of Incident', min_value=1940, max_value=2025, value=2024)
        input_features['Month'] = st.number_input('Month of Incident', min_value=1, max_value=12, value=7)
        input_features['Total_Fatal_Injuries'] = st.number_input('Total Fatal Injuries', min_value=0, value=0)
        input_features['Total_Serious_Injuries'] = st.number_input('Total Serious Injuries', min_value=0, value=0)

    st.markdown("---")

    if st.button("Predict Severity", type="primary"):
        if any(input_features[key] == '' for key in ['Make', 'Model', 'Engine_Type', 'Country', 'Weather_Condition', 'Broad_phase_of_flight', 'Purpose_of_flight']):
            st.error("Please fill in all dropdown and text fields before predicting.")
        else:
            input_df = pd.DataFrame([input_features])
            for col, encoder in encoders.items():
                if col in input_df.columns:
                    val = input_df.iloc[0][col]
                    if val not in encoder.classes_:
                        input_df.loc[0, col] = 'Unknown'
                    input_df[col] = encoder.transform(input_df[col])
            
            input_df = input_df[model.get_booster().feature_names]
            
            prediction_encoded = model.predict(input_df)[0]
            prediction_decoded = encoders['target'].inverse_transform([prediction_encoded])[0]
            
            st.subheader("Prediction Result")
            if prediction_decoded == "Destroyed":
                st.error(f"Predicted Damage Severity: **{prediction_decoded}**")
            elif prediction_decoded == "Substantial":
                st.warning(f"Predicted Damage Severity: **{prediction_decoded}**")
            else:
                st.success(f"Predicted Damage Severity: **{prediction_decoded}**")

elif selected == "Crash Case Studies":
    st.subheader("A review of notable aviation incidents.")
    crashes = {
        "Tenerife Airport Disaster (1977)": {"summary": "**Date:** March 27, 1977 | **Aircraft:** Boeing 747 | **Fatalities:** 583\n\nThe deadliest accident in aviation history. Two Boeing 747s collided on the runway in heavy fog due to miscommunications."},
        "Japan Airlines Flight 123 (1985)": {"summary": "**Date:** August 12, 1985 | **Aircraft:** Boeing 747SR | **Fatalities:** 520\n\nA faulty repair failed, causing an explosive decompression that destroyed all hydraulic controls."},
        "US Airways Flight 1549 (2009)": {"summary": "**Date:** January 15, 2009 | **Aircraft:** Airbus A320 | **Fatalities:** 0\n\nThe 'Miracle on the Hudson.' The aircraft lost all engine power after a bird strike and was ditched on the Hudson River."},
    }
    for crash_name, crash_info in crashes.items():
        with st.expander(f"✈️ **{crash_name}**"):
            st.markdown(crash_info['summary'])
