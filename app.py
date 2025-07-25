import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_option_menu import option_menu
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

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

# -----------------
# Data Loading and Model Training (Cached)
# -----------------
@st.cache_data
def load_data():
    """Loads, cleans, and engineers features from the aviation data."""
    data = pd.read_csv('AviationData.csv', encoding='latin-1')
    data.columns = data.columns.str.replace('.', '_', regex=False)
    
    data['Event_Date'] = pd.to_datetime(data['Event_Date'], errors='coerce')
    data['Year'] = data['Event_Date'].dt.year
    data['Month'] = data['Event_Date'].dt.month
    
    data = data.dropna(subset=['Aircraft_damage'])
    data = data[data['Aircraft_damage'] != 'Unknown']
    
    num_cols_to_impute = [
        'Number_of_Engines', 'Total_Fatal_Injuries', 'Total_Serious_Injuries',
        'Total_Minor_Injuries', 'Total_Uninjured', 'Year', 'Month'
    ]
    for col in num_cols_to_impute:
        if data[col].isnull().any():
            median_val = data[col].median()
            data[col].fillna(median_val, inplace=True)

    for col in data.select_dtypes(include=['object']).columns:
        data[col] = data[col].fillna('Unknown')
    
    data[['Year', 'Month']] = data[['Year', 'Month']].astype(int)
    return data

@st.cache_resource
def train_models(data_df):
    """Trains models and returns all necessary components."""
    features = [
        'Make', 'Model', 'Engine_Type', 'Number_of_Engines', 'Weather_Condition',
        'Broad_phase_of_flight', 'Purpose_of_flight', 'Country', 'Year', 'Month',
        'Total_Fatal_Injuries', 'Total_Serious_Injuries'
    ]
    target = 'Aircraft_damage'
    df = data_df[features + [target]].copy()
    
    encoders = {col: LabelEncoder() for col in df.select_dtypes(include=['object']).columns if col != target}
    for col, encoder in encoders.items():
        df[col] = encoder.fit_transform(df[col])
        
    X = df[features]
    y = df[target]
    
    y_encoder = LabelEncoder()
    y_encoded = y_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    
    xgb_tuned_params = {
        'objective': 'multi:softmax', 'n_estimators': 300, 'max_depth': 8,
        'learning_rate': 0.1, 'subsample': 0.8, 'colsample_bytree': 0.8,
        'use_label_encoder': False, 'eval_metric': 'mlogloss', 'random_state': 42
    }

    models = {
        "Random Forest (Baseline)": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost (Optimized)": XGBClassifier(**xgb_tuned_params)
    }
    
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        performance = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "F1 Score": f1_score(y_test, y_pred, average='weighted'),
            "Confusion Matrix": confusion_matrix(y_test, y_pred),
            "Classes": y_encoder.classes_
        }
        if hasattr(model, 'feature_importances_'):
            performance['Feature Importances'] = model.feature_importances_
            performance['Feature Names'] = X.columns
        trained_models[name] = {"model": model, "performance": performance, "encoders": encoders, "y_encoder": y_encoder}
    return trained_models

# --- Load data and models ---
data = load_data()
models_info = train_models(data)
best_model_name = "XGBoost (Optimized)"
best_model_pack = models_info[best_model_name]

# --- Main Title ---
st.markdown("<h1 style='text-align: center; font-weight: bold;'>AVIATION DAMAGE PREDICTION SYSTEM</h1>", unsafe_allow_html=True)

# -----------------
# Horizontal Navigation Bar
# -----------------
selected = option_menu(
    menu_title=None,
    options=["Home", "Live Prediction", "Crash Case Studies", "Model Performance", "Data Analysis"],
    icons=["house", "rocket-launch", "journal-text", "bar-chart-line", "search"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
)

# -----------------
# Page Content
# -----------------

# --- Home Page ---
if selected == "Home":
    st.subheader("A Machine Learning project to enhance aviation safety through data.")
    st.write("""
    Welcome to the Aircraft Damage Severity Predictor. This application leverages a sophisticated machine learning model 
    to predict the extent of damage to an aircraft following an incident. By analyzing historical data, we can identify key factors 
    that contribute to severe outcomes.
    
    **Our Goal:** To provide a tool that aids in pilot training, risk assessment, and ultimately, the prevention of future incidents.
    """)
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Incidents Analyzed", f"{data.shape[0]:,}")
    col2.metric("Prediction Accuracy", f"{best_model_pack['performance']['Accuracy']:.2%}")
    col3.metric("Key Risk Factor", "Fatal Injuries")

    st.markdown("#### How to use this application:")
    st.info("""
    - **Live Prediction:** Go to this tab to input details of a hypothetical incident and get an instant damage prediction.
    - **Crash Case Studies:** Review summaries of notable historical aviation incidents.
    - **Model Performance:** Explore the accuracy and other metrics of our prediction models.
    - **Data Analysis:** Dive deep into the dataset with interactive charts and graphs.
    """)

# --- Live Prediction Page ---
if selected == "Live Prediction":
    st.subheader(f"Enter incident details below. Our **{best_model_name}** model will predict the outcome.")
    st.markdown("---")

    input_features = {}
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        input_features['Make'] = st.selectbox("Aircraft Make", options=[''] + sorted(list(data['Make'].unique())), key='make')
        input_features['Model'] = st.selectbox("Aircraft Model", options=[''] + sorted(list(data['Model'].unique())), key='model')
        input_features['Engine_Type'] = st.selectbox("Engine Type", options=[''] + sorted(list(data['Engine_Type'].unique())), key='engine')
        input_features['Country'] = st.selectbox("Country", options=[''] + sorted(list(data['Country'].unique())), key='country')

    with col2:
        input_features['Weather_Condition'] = st.selectbox("Weather Condition", options=[''] + sorted(list(data['Weather_Condition'].unique())), key='weather')
        input_features['Broad_phase_of_flight'] = st.selectbox("Phase of Flight", options=[''] + sorted(list(data['Broad_phase_of_flight'].unique())), key='phase')
        input_features['Purpose_of_flight'] = st.selectbox("Purpose of Flight", options=[''] + sorted(list(data['Purpose_of_flight'].unique())), key='purpose')
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
            for col, encoder in best_model_pack['encoders'].items():
                if col in input_df.columns:
                    input_df[col] = encoder.transform(input_df[col])
            
            training_cols = best_model_pack['model'].get_booster().feature_names
            input_df = input_df[training_cols]
            
            prediction_encoded = best_model_pack['model'].predict(input_df)[0]
            prediction_proba = best_model_pack['model'].predict_proba(input_df)
            prediction_decoded = best_model_pack['y_encoder'].inverse_transform([prediction_encoded])[0]
            
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.subheader("Prediction Result")
                if prediction_decoded == "Destroyed":
                    st.error(f"Predicted Damage Severity: **{prediction_decoded}**")
                elif prediction_decoded == "Substantial":
                    st.warning(f"Predicted Damage Severity: **{prediction_decoded}**")
                else:
                    st.success(f"Predicted Damage Severity: **{prediction_decoded}**")

            with res_col2:
                st.subheader("Prediction Probabilities")
                prob_df = pd.DataFrame(prediction_proba, columns=best_model_pack['performance']['Classes'])
                st.dataframe(prob_df)

# --- Crash Case Studies Page ---
if selected == "Crash Case Studies":
    st.subheader("A review of notable aviation incidents. These case studies highlight the complex factors involved in aircraft accidents.")
    
    crashes = {
        "Tenerife Airport Disaster (1977)": {"summary": "**Date:** March 27, 1977 | **Aircraft:** Boeing 747-100 & 747-200 | **Fatalities:** 583\n\nThe deadliest accident in aviation history. Two Boeing 747s collided on the runway at Los Rodeos Airport in heavy fog. A series of miscommunications and procedural errors led to one aircraft attempting to take off while the other was still on the same runway."},
        "American Airlines Flight 191 (1979)": {"summary": "**Date:** May 25, 1979 | **Aircraft:** McDonnell Douglas DC-10 | **Fatalities:** 273\n\nShortly after takeoff from Chicago O'Hare, the left engine detached from the wing, severing hydraulic lines and causing catastrophic damage. The aircraft rolled and crashed less than a minute later. The cause was traced to faulty maintenance procedures."},
        "Japan Airlines Flight 123 (1985)": {"summary": "**Date:** August 12, 1985 | **Aircraft:** Boeing 747SR | **Fatalities:** 520\n\nA faulty repair of the rear pressure bulkhead from a previous incident failed, causing an explosive decompression that destroyed the vertical stabilizer and all hydraulic systems. The crew fought to control the aircraft for 32 minutes before crashing."},
        "Indian Airlines Flight 113 (1988)": {"summary": "**Date:** October 19, 1988 | **Aircraft:** Boeing 737-200 | **Fatalities:** 133\n\nThe flight crashed on final approach to Ahmedabad, in poor visibility. The investigation highlighted issues with crew decision-making in adverse weather and non-adherence to standard approach procedures."},
        "US Airways Flight 1549 (2009)": {"summary": "**Date:** January 15, 2009 | **Aircraft:** Airbus A320-214 | **Fatalities:** 0\n\nThe 'Miracle on the Hudson.' The aircraft lost all engine power after a bird strike. The crew successfully ditched the plane on the Hudson River with no fatalities, showcasing exceptional airmanship."},
        "Air France Flight 447 (2009)": {"summary": "**Date:** June 1, 2009 | **Aircraft:** Airbus A330-203 | **Fatalities:** 228\n\nThe aircraft's airspeed sensors became iced over at high altitude, leading to the autopilot disconnecting. The crew's incorrect response caused the aircraft to enter a stall from which it did not recover."},
        "Air India Express Flight 812 (2010)": {"summary": "**Date:** May 22, 2010 | **Aircraft:** Boeing 737-800 | **Fatalities:** 158\n\nThe aircraft overshot the runway upon landing at Mangalore International Airport, which is a 'tabletop' runway with steep drops at either end. The aircraft fell down a hillside and caught fire. Pilot error was cited as the primary cause."},
        "Malaysia Airlines Flight 370 (2014)": {"summary": "**Date:** March 8, 2014 | **Aircraft:** Boeing 777-200ER | **Fatalities:** 239 (Presumed)\n\nThe flight disappeared from radar while en route from Kuala Lumpur to Beijing. Despite the most extensive search in aviation history, the aircraft has never been found. The cause of the disappearance remains one of the greatest mysteries in modern aviation."},
        "Lion Air Flight 610 (2018)": {"summary": "**Date:** October 29, 2018 | **Aircraft:** Boeing 737 MAX 8 | **Fatalities:** 189\n\nOne of two fatal crashes involving the Boeing 737 MAX. A faulty angle of attack sensor activated the MCAS flight control system, which repeatedly pushed the aircraft's nose down. The pilots were unable to regain control, and the aircraft crashed into the Java Sea shortly after takeoff."},
        "Ethiopian Airlines Flight 302 (2019)": {"summary": "**Date:** March 10, 2019 | **Aircraft:** Boeing 737 MAX 8 | **Fatalities:** 157\n\nSimilar to the Lion Air crash, a faulty sensor activated the MCAS system, causing the aircraft to enter an unrecoverable dive. This second incident led to the worldwide grounding of the entire Boeing 737 MAX fleet."},
    }
    
    for crash_name, crash_info in crashes.items():
        with st.expander(f"✈️ **{crash_name}**"):
            st.markdown(crash_info['summary'])

# --- Model Performance Page ---
if selected == "Model Performance":
    st.subheader("Model Performance & In-Depth Analysis")
    st.markdown("Here we analyze our models to understand their performance and the factors driving their predictions.")
    st.header("📈 Model Comparison")
    
    perf_list = []
    for name, info in models_info.items():
        p = info['performance']
        perf_list.append({'Model': name, 'Metric': 'Accuracy', 'Score': p['Accuracy']})
        perf_list.append({'Model': name, 'Metric': 'F1 Score', 'Score': p['F1 Score']})
    
    perf_df = pd.DataFrame(perf_list)
    fig_comp = px.bar(perf_df, x="Metric", y="Score", color="Model", barmode="group", title="Side-by-Side Model Performance Metrics")
    st.plotly_chart(fig_comp, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.header(f"⚙️ Feature Importance")
        st.markdown("This chart shows which factors our best model (**XGBoost**) considers most important.")
        importances = best_model_pack['performance']['Feature Importances']
        feature_names = best_model_pack['performance']['Feature Names']
        feature_imp_df = pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values('importance', ascending=False).head(10)
        fig_imp = px.bar(feature_imp_df, x='importance', y='feature', orientation='h', title=f"Top 10 Features for {best_model_name}")
        fig_imp.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_imp, use_container_width=True)

    with col2:
        st.header("🚦 Confusion Matrix")
        st.markdown("This matrix shows the detailed breakdown of our model's predictions.")
        cm = best_model_pack['performance']['Confusion Matrix']
        classes = best_model_pack['performance']['Classes']
        fig_cm, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, ax=ax)
        ax.set_xlabel('Predicted Label'); ax.set_ylabel('True Label')
        st.pyplot(fig_cm)

# --- Data Analysis Page ---
if selected == "Data Analysis":
    st.subheader("Exploratory Data Analysis (EDA)")
    st.markdown("A deep dive into the aviation incident dataset to uncover trends and patterns.")

    st.header("Dataset Overview")
    st.dataframe(data.head())
    with st.expander("See Full Dataset Statistics"):
        st.dataframe(data.describe())
    
    st.markdown("---")
    st.header("Visual Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Damage Severity Distribution")
        fig_pie = px.pie(data, names='Aircraft_damage', title='Proportion of Aircraft Damage Severity', hole=0.3)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        st.subheader("Incidents Over Time")
        incidents_by_year = data['Year'].value_counts().sort_index()
        fig_line = px.line(x=incidents_by_year.index, y=incidents_by_year.values, 
                           labels={'x': 'Year', 'y': 'Number of Incidents'}, title='Total Aviation Incidents per Year')
        st.plotly_chart(fig_line, use_container_width=True)

    st.subheader("Damage Severity by Key Factors")
    
    tab1, tab2, tab3 = st.tabs(["Weather", "Phase of Flight", "Engine Type"])
    
    with tab1:
        fig_weather = px.bar(data.groupby(['Weather_Condition', 'Aircraft_damage']).size().reset_index(name='count'),
                               x='Weather_Condition', y='count', color='Aircraft_damage',
                               title='Damage Severity by Weather Condition', barmode='group')
        st.plotly_chart(fig_weather, use_container_width=True)
        
    with tab2:
        fig_phase = px.bar(data.groupby(['Broad_phase_of_flight', 'Aircraft_damage']).size().reset_index(name='count'),
                           x='Broad_phase_of_flight', y='count', color='Aircraft_damage',
                           title='Damage Severity by Phase of Flight', barmode='group')
        st.plotly_chart(fig_phase, use_container_width=True)
        
    with tab3:
        fig_engine = px.bar(data.groupby(['Engine_Type', 'Aircraft_damage']).size().reset_index(name='count'),
                             x='Engine_Type', y='count', color='Aircraft_damage',
                             title='Damage Severity by Engine Type', barmode='group')
        st.plotly_chart(fig_engine, use_container_width=True)

    st.subheader("Geographical Distribution of Incidents")
    country_counts = data['Country'].value_counts().reset_index()
    country_counts.columns = ['Country', 'Incidents']
    fig_map = px.choropleth(country_counts, locations='Country', locationmode='country names',
                            color='Incidents', hover_name='Country', 
                            color_continuous_scale=px.colors.sequential.Plasma,
                            title='Total Incidents by Country')
    st.plotly_chart(fig_map, use_container_width=True)