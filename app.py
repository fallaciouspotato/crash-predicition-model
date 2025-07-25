import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
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

# --- Top-left team credit ---
st.text("By Team 2 LPA")

# ------------------------------------
# Load Pre-trained Model and Encoders
# ------------------------------------
@st.cache_resource
def load_model_and_encoders():
    """Loads the pre-trained model and encoders from disk."""
    model = joblib.load('xgboost_model.joblib')
    encoders = joblib.load('encoders.joblib')
    return model, encoders

model, encoders = load_model_and_encoders()

@st.cache_data
def load_info_data():
    """Loads a small subset of data for populating dropdowns and metrics."""
    data = pd.read_csv('AviationData.csv', encoding='latin-1', usecols=['Make', 'Model', 'Engine_Type', 'Country', 'Weather_Condition', 'Broad_phase_of_flight', 'Purpose_of_flight'])
    return data

info_data = load_info_data()

# --- Main Title ---
st.markdown("<h1 style='text-align: center; font-weight: bold;'>AVIATION DAMAGE PREDICTION SYSTEM</h1>", unsafe_allow_html=True)

# -----------------
# Horizontal Navigation Bar
# -----------------
selected = option_menu(
    menu_title=None,
    options=["Home", "Live Prediction", "Crash Case Studies"],
    icons=["house", "rocket-launch", "journal-text"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
)

# -----------------
# Page Content
# -----------------
if selected == "Home":
    st.subheader("A Machine Learning project to enhance aviation safety through data.")
    st.write("""
    Welcome! This application uses a pre-trained machine learning model to predict the severity of aircraft damage from incident reports.
    By analyzing historical data, we can identify key factors that contribute to severe outcomes and help prevent future accidents.
    """)
    st.info("""
    - **Live Prediction:** Input details of a hypothetical incident to get an instant damage prediction.
    - **Crash Case Studies:** Review summaries of notable historical aviation incidents.
    """)

elif selected == "Live Prediction":
    st.subheader(f"Enter incident details below to get a real-time prediction.")
    st.markdown("---")

    input_features = {}
    col1, col2, col3 = st.columns(3)

    with col1:
        input_features['Make'] = st.selectbox("Aircraft Make", options=[''] + sorted(list(info_data['Make'].unique())))
        input_features['Model'] = st.selectbox("Aircraft Model", options=[''] + sorted(list(info_data['Model'].unique())))
        input_features['Engine_Type'] = st.selectbox("Engine Type", options=[''] + sorted(list(info_data['Engine_Type'].unique())))
        input_features['Country'] = st.selectbox("Country", options=[''] + sorted(list(info_data['Country'].unique())))

    with col2:
        input_features['Weather_Condition'] = st.selectbox("Weather Condition", options=[''] + sorted(list(info_data['Weather_Condition'].unique())))
        input_features['Broad_phase_of_flight'] = st.selectbox("Phase of Flight", options=[''] + sorted(list(info_data['Broad_phase_of_flight'].unique())))
        input_features['Purpose_of_flight'] = st.selectbox("Purpose of Flight", options=[''] + sorted(list(info_data['Purpose_of_flight'].unique())))
        input_features['Number_of_Engines'] = st.number_input('Number of Engines', min_value=0, max_value=8, value=2)

    with col3:
        input_features['Year'] = st.number_input('Year of Incident', min_value=1940, max_value=2025, value=2024)
        input_features['Month'] = st.number_input('Month of Incident', min_value=1, max_value=12, value=7)
        input_features['Total_Fatal_Injuries'] = st.number_input('Total Fatal Injuries', min_value=0, value=0)
        input_features['Total_Serious_Injuries'] = st.number_input('Total Serious Injuries', min_value=0, value=0)

    st.markdown("---")

    if st.button("Predict Severity", type="primary"):
        cat_features = ['Make', 'Model', 'Engine_Type', 'Country', 'Weather_Condition', 'Broad_phase_of_flight', 'Purpose_of_flight']
        if any(input_features[key] == '' for key in cat_features):
            st.error("Please fill in all dropdown details before predicting.")
        else:
            input_df = pd.DataFrame([input_features])
            for col, encoder in encoders.items():
                if col in input_df.columns:
                    # Handle unseen labels by mapping to 'Unknown'
                    input_val = input_df.iloc[0][col]
                    if input_val not in encoder.classes_:
                        input_df[col] = 'Unknown'
                    input_df[col] = encoder.transform(input_df[col])
            
            training_cols = model.get_booster().feature_names
            input_df = input_df[training_cols]
            
            prediction_encoded = model.predict(input_df)[0]
            prediction_proba = model.predict_proba(input_df)
            y_encoder = encoders['target']
            prediction_decoded = y_encoder.inverse_transform([prediction_encoded])[0]
            
            st.subheader("Prediction Result")
            if prediction_decoded == "Destroyed": st.error(f"Predicted Damage Severity: **{prediction_decoded}**")
            elif prediction_decoded == "Substantial": st.warning(f"Predicted Damage Severity: **{prediction_decoded}**")
            else: st.success(f"Predicted Damage Severity: **{prediction_decoded}**")

elif selected == "Crash Case Studies":
    st.subheader("A review of notable aviation incidents to understand contributing factors.")
    
    # ... (rest of the crash case studies expanders remain the same) ...
    crashes = {
        "Tenerife Airport Disaster (1977)": {"summary": "**Date:** March 27, 1977 | **Aircraft:** Boeing 747-100 & 747-200 | **Fatalities:** 583\n\nThe deadliest accident in aviation history. Two Boeing 747s collided on the runway at Los Rodeos Airport in heavy fog."},
        "American Airlines Flight 191 (1979)": {"summary": "**Date:** May 25, 1979 | **Aircraft:** McDonnell Douglas DC-10 | **Fatalities:** 273\n\nShortly after takeoff from Chicago O'Hare, the left engine detached from the wing, severing hydraulic lines and causing catastrophic damage."},
        "Japan Airlines Flight 123 (1985)": {"summary": "**Date:** August 12, 1985 | **Aircraft:** Boeing 747SR | **Fatalities:** 520\n\nA faulty repair of the rear pressure bulkhead failed, causing an explosive decompression that destroyed the vertical stabilizer and all hydraulic systems."},
        "Indian Airlines Flight 113 (1988)": {"summary": "**Date:** October 19, 1988 | **Aircraft:** Boeing 737-200 | **Fatalities:** 133\n\nThe flight crashed on final approach to Ahmedabad, in poor visibility."},
        "US Airways Flight 1549 (2009)": {"summary": "**Date:** January 15, 2009 | **Aircraft:** Airbus A320-214 | **Fatalities:** 0\n\nThe 'Miracle on the Hudson.' The aircraft lost all engine power after a bird strike and was ditched on the Hudson River."},
        "Air France Flight 447 (2009)": {"summary": "**Date:** June 1, 2009 | **Aircraft:** Airbus A330-203 | **Fatalities:** 228\n\nThe aircraft's airspeed sensors became iced over at high altitude, leading to a stall from which the crew did not recover."},
        "Air India Express Flight 812 (2010)": {"summary": "**Date:** May 22, 2010 | **Aircraft:** Boeing 737-800 | **Fatalities:** 158\n\nThe aircraft overshot a 'tabletop' runway upon landing at Mangalore International Airport and fell down a hillside."},
        "Malaysia Airlines Flight 370 (2014)": {"summary": "**Date:** March 8, 2014 | **Aircraft:** Boeing 777-200ER | **Fatalities:** 239 (Presumed)\n\nThe flight disappeared from radar, and despite an extensive search, the aircraft has never been found."},
        "Lion Air Flight 610 (2018)": {"summary": "**Date:** October 29, 2018 | **Aircraft:** Boeing 737 MAX 8 | **Fatalities:** 189\n\nA faulty sensor activated the MCAS flight control system, repeatedly pushing the aircraft's nose down until the pilots lost control."},
        "Ethiopian Airlines Flight 302 (2019)": {"summary": "**Date:** March 10, 2019 | **Aircraft:** Boeing 737 MAX 8 | **Fatalities:** 157\n\nSimilar to the Lion Air crash, a faulty sensor activated the MCAS system, leading to the worldwide grounding of the 737 MAX fleet."},
    }
    
    for crash_name, crash_info in crashes.items():
        with st.expander(f"✈️ **{crash_name}**"):
            st.markdown(crash_info['summary'])
