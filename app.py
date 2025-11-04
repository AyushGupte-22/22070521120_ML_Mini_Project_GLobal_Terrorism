import streamlit as st
import pandas as pd
from catboost import CatBoostClassifier
import joblib
import warnings
from datetime import datetime
import os
import numpy as np
from geopy.geocoders import Nominatim # For location lookup
from geopy.extra.rate_limiter import RateLimiter # For location lookup
import time # For location lookup rate limit

warnings.filterwarnings('ignore')

# --- Configuration ---
# Model Paths
MODEL_GROUP_PATH = 'catboost_top_100_known_groups_model.cbm'
ENCODER_GROUP_PATH = 'top_100_known_groups_label_encoder.pkl'
MODEL_WEAPON_PATH = 'catboost_predict_weapon_model.cbm'
ENCODER_WEAPON_PATH = 'weapon_type_label_encoder.pkl'
MODEL_ATTACK_PATH = 'catboost_predict_attacktype_model.cbm' # New
ENCODER_ATTACK_PATH = 'attack_type_label_encoder.pkl'     # New
MODEL_TARGET_PATH = 'catboost_predict_targettype_model.cbm' # New
ENCODER_TARGET_PATH = 'target_type_label_encoder.pkl'     # New

# Data Path for dropdowns
DATA_PATH = 'Cleaned Dataset/globalterrorismdb_cleaned.csv'
LOG_FILE = 'prediction_log_multi_v2.csv'

# --- Resource Loading ---
@st.cache_resource
def load_all_resources():
    """Loads all models and encoders."""
    resources = {'GroupName': None, 'WeaponType': None, 'AttackType': None, 'TargetType': None}
    load_success = True

    def load_resource(key, model_p, encoder_p):
        nonlocal load_success
        try:
            model = CatBoostClassifier()
            model.load_model(model_p)
            encoder = joblib.load(encoder_p)
            resources[key] = {'model': model, 'encoder': encoder}
            st.sidebar.success(f"{key} model loaded.")
        except Exception as e:
            st.sidebar.error(f"Failed loading {key} model/encoder: {e}")
            load_success = False

    load_resource('GroupName', MODEL_GROUP_PATH, ENCODER_GROUP_PATH)
    load_resource('WeaponType', MODEL_WEAPON_PATH, ENCODER_WEAPON_PATH)
    load_resource('AttackType', MODEL_ATTACK_PATH, ENCODER_ATTACK_PATH)
    load_resource('TargetType', MODEL_TARGET_PATH, ENCODER_TARGET_PATH)
    
    if load_success: st.sidebar.success("All available resources loaded.")
    else: st.sidebar.warning("Some resources failed to load. Check paths/files.")
        
    # Add geolocator separately
    try:
        geolocator = Nominatim(user_agent="gtd_app_geolookup")
        resources['geolocator'] = RateLimiter(geolocator.reverse, min_delay_seconds=1.1, error_wait_seconds=5.0) # Rate limit
        st.sidebar.info("Geocoder initialized.")
    except Exception as e:
        st.sidebar.error(f"Failed to initialize geocoder: {e}")
        resources['geolocator'] = None

    return resources

@st.cache_data
def get_dropdown_options(data_path):
    """Loads unique values for dropdowns and expected columns."""
    try:
        df_options = pd.read_csv(data_path, encoding='ISO-8859-1')
        df_options = standardize_group_names_ui(df_options) # Standardize for group input list

        # Define expected columns for each model based on training scripts
        expected_cols = {
            'GroupName': ['Year', 'Month', 'Country', 'Region', 'Latitude', 'Longitude', 'AttackType', 'TargetType', 'WeaponType', 'suicide'],
            'WeaponType': ['GroupName', 'Region', 'Year', 'TargetType', 'AttackType', 'suicide', 'Latitude', 'Longitude', 'Country'],
            'AttackType': ['GroupName', 'Region', 'Year', 'TargetType', 'WeaponType', 'suicide', 'Latitude', 'Longitude', 'Country'],
            'TargetType': ['GroupName', 'Region', 'Year', 'AttackType', 'WeaponType', 'suicide', 'Latitude', 'Longitude', 'Country']
        }
        numeric_cols_base = ['Year', 'Month', 'Latitude', 'Longitude', 'suicide']

        # Get unique values for dropdowns
        options_dict = {
            'groups': sorted(df_options[df_options['GroupName']!='Unknown']['GroupName'].unique()),
            'countries': sorted(df_options['Country'].unique()),
            'regions': sorted(df_options['Region'].unique()),
            'attack_types': sorted(df_options['AttackType'].unique()),
            'target_types': sorted(df_options['TargetType'].unique()),
            'weapon_types': sorted(df_options['WeaponType'].unique()),
            'numeric_columns': numeric_cols_base
        }
        # Add expected columns for each prediction type
        for key, cols in expected_cols.items():
            options_dict[f'expected_columns_{key.lower()}'] = cols

        return options_dict
    except Exception as e:
        st.error(f"Error loading data for dropdowns from {data_path}: {e}")
        return None

# Simplified standardization for UI dropdowns
def standardize_group_names_ui(df):
     name_map = {r'.*al-shabaab.*': 'Al-Shabaab',r'.*taliban.*': 'Taliban',r'.*isil.*': 'Islamic State of Iraq and the Levant (ISIL)', r'.*boko haram.*': 'Boko Haram',r'.*cpi-maoist.*': 'Communist Party of India - Maoist (CPI-Maoist)',r'^maoists$':'Communist Party of India - Maoist (CPI-Maoist)',r'.*new people\'s army.*': 'New People\'s Army (NPA)', r'.*shining path.*': 'Shining Path (SL)'} # Add more if needed
     df['Standardized_Group'] = df['GroupName'].str.lower().fillna('unknown')
     df['Standardized_Group'].replace(name_map, regex=True, inplace=True)
     known_standard_names = set(name_map.values()); df['Standardized_Group'] = df.apply(lambda row: row['Standardized_Group'] if row['Standardized_Group'] in known_standard_names else row['GroupName'].title(), axis=1); df['Standardized_Group'] = df['Standardized_Group'].fillna('Unknown').apply(lambda x: x.title() if isinstance(x, str) else 'Unknown'); df = df.drop(columns=['GroupName'], errors='ignore'); df = df.rename(columns={'Standardized_Group': 'GroupName'}); return df

# --- Prediction Function ---
def predict_feature(attack_data, prediction_target, resources, options):
    """Selects the correct model and makes a prediction."""
    resource = resources.get(prediction_target)
    if resource is None or resource.get('model') is None: return f"Model for {prediction_target} not loaded"
    model = resource['model']; encoder = resource['encoder']
    expected_cols = options.get(f'expected_columns_{prediction_target.lower()}'); numeric_cols = options['numeric_columns']
    if not expected_cols: return f"Config error: Expected columns missing for {prediction_target}"

    try:
        input_df = pd.DataFrame([attack_data])
        # Add missing, ensure order, handle NaN/dtypes
        for col in expected_cols:
            if col not in input_df.columns:
                input_df[col] = np.nan if col in numeric_cols else 'Unknown'
        input_df = input_df[expected_cols]
        for col in input_df.columns:
            if col not in numeric_cols:
                if pd.isna(input_df[col].iloc[0]): input_df[col] = 'Unknown'
                else: input_df[col] = input_df[col].astype(str)
            elif col in numeric_cols and pd.isna(input_df[col].iloc[0]): # Ensure NaNs are handled if needed
                input_df[col] = np.nan # Use NaN for CatBoost numeric NaNs

        prediction_encoded = model.predict(input_df)[0]
        if isinstance(prediction_encoded, np.ndarray): prediction_encoded = prediction_encoded.item()
        if prediction_encoded >= len(encoder.classes_): return "Prediction Failed (Unknown Label)"
        prediction = encoder.inverse_transform([prediction_encoded])[0]
        return prediction
    except Exception as e:
        st.error(f"Prediction Error ({prediction_target}): {e}."); return "Prediction Failed"

# --- Location Lookup Function ---
def lookup_location(lat, lon, geolocator):
    """Uses geopy to find location from coordinates."""
    if pd.isna(lat) or pd.isna(lon): return "Please provide both Latitude and Longitude."
    if geolocator is None: return "Geocoder not available."
    try:
        location = geolocator((lat, lon), exactly_one=True, language='en', timeout=10)
        time.sleep(1) # Respect rate limit manually just in case
        if location:
            address = location.raw.get('address', {})
            city = address.get('city', address.get('town', address.get('village', address.get('state_district', 'N/A'))))
            state = address.get('state', 'N/A')
            country = address.get('country', 'N/A')
            return f"Predicted Location: {city}, {state}, {country}\n(Full Address: {location.address})"
        else:
            return "Location not found for these coordinates."
    except Exception as e:
        st.error(f"Geocoder Error: {e}")
        return "Location lookup failed."

# --- Logging Function ---
# (Keep the same log_prediction function as before)
def log_prediction(timestamp, input_data, prediction_target, prediction):
    log_entry = {**input_data, 'prediction_timestamp': timestamp, 'prediction_target': prediction_target, 'prediction_result': prediction}
    log_df = pd.DataFrame([log_entry])
    try: header = not os.path.exists(LOG_FILE); log_df.to_csv(LOG_FILE, mode='a', header=header, index=False)
    except Exception as e: st.warning(f"Failed to write log: {e}")

# --- Streamlit App UI ---
st.set_page_config(layout="wide", page_title="Terrorism Attack Predictor")
st.title("🌍 Terrorism Attack Characteristic Prediction")
st.markdown("Use machine learning models and location lookup to predict different aspects of a terrorist attack.")

# Load resources
resources = load_all_resources()
options = get_dropdown_options(DATA_PATH)

# --- Create Tabs for Different Tasks ---
tab_predict, tab_locate = st.tabs(["📊 Predict Attack Feature", "📍 Location Lookup"])

# === Prediction Tab ===
with tab_predict:
    st.header("Predict Attack Feature")
    prediction_target = st.selectbox(
        "Select the feature you want to predict:",
        ('GroupName', 'WeaponType', 'AttackType', 'TargetType'), # Add others as models are loaded
        key='prediction_target_selector'
    )

    # Check if model is available
    if not resources.get(prediction_target):
        st.warning(f"Model to predict '{prediction_target}' is not available or failed to load.")
    elif options:
        st.markdown(f"Enter the known details below to predict the **{prediction_target}**.")
        form = st.form(key='prediction_form')
        input_data = {}
        col1, col2 = form.columns(2)

        # Dynamically create input fields
        with col1:
            form.subheader("Temporal & Geographic")
            input_data['Year'] = form.number_input("Year", 1970, 2030, datetime.now().year)
            input_data['Month'] = form.number_input("Month", 1, 12, datetime.now().month)
            if prediction_target != 'Country':
                 default_c_idx = options['countries'].index('Iraq') if 'Iraq' in options['countries'] else 0
                 input_data['Country'] = form.selectbox("Country", options['countries'], index=default_c_idx)
            if prediction_target != 'Region':
                 default_r_idx = options['regions'].index('Middle East & North Africa') if 'Middle East & North Africa' in options['regions'] else 0
                 input_data['Region'] = form.selectbox("Region", options['regions'], index=default_r_idx)
            input_data['Latitude'] = form.number_input("Latitude (Optional)", value=None, placeholder="e.g., 33.3152", format="%.4f")
            input_data['Longitude'] = form.number_input("Longitude (Optional)", value=None, placeholder="e.g., 44.3661", format="%.4f")

        with col2:
            form.subheader("Attack Specifics")
            if prediction_target != 'GroupName':
                 try:
                    # Use encoder for the *model being used as input*, e.g., WeaponType model needs Top 100 groups
                    group_encoder_path = ENCODER_GROUP_PATH # Assuming GroupName input is based on the GroupName model's scope
                    group_encoder_input = joblib.load(group_encoder_path)
                    available_groups = sorted(list(group_encoder_input.classes_))
                    default_g_idx = available_groups.index('Taliban') if 'Taliban' in available_groups else 0
                    input_data['GroupName'] = form.selectbox(f"Known Group Name (Top {len(available_groups)})", available_groups, index=default_g_idx)
                 except Exception as e:
                     st.warning(f"Could not load group list: {e}. Using text input.")
                     input_data['GroupName'] = form.text_input("Known Group Name", value="Taliban")

            if prediction_target != 'AttackType':
                 default_a_idx = options['attack_types'].index('Bombing/Explosion') if 'Bombing/Explosion' in options['attack_types'] else 0
                 input_data['AttackType'] = form.selectbox("Attack Type", options['attack_types'], index=default_a_idx)
            if prediction_target != 'TargetType':
                 default_t_idx = options['target_types'].index('Military') if 'Military' in options['target_types'] else 0
                 input_data['TargetType'] = form.selectbox("Target Type", options['target_types'], index=default_t_idx)
            if prediction_target != 'WeaponType':
                 default_w_idx = options['weapon_types'].index('Explosives') if 'Explosives' in options['weapon_types'] else 0
                 input_data['WeaponType'] = form.selectbox("Weapon Type", options['weapon_types'], index=default_w_idx)
            input_data['suicide'] = form.selectbox("Suicide Attack?", (0, 1), index=0, format_func=lambda x: "Yes" if x == 1 else "No")

        # Submit button for the form
        submit_button = form.form_submit_button(label=f"Predict {prediction_target}")

        if submit_button:
            with st.spinner(f"Predicting {prediction_target}..."):
                lat_input = input_data.get('Latitude', None); lon_input = input_data.get('Longitude', None)
                input_data['Latitude'] = lat_input if lat_input is not None else np.nan
                input_data['Longitude'] = lon_input if lon_input is not None else np.nan

                prediction_result = predict_feature(input_data, prediction_target, resources, options)

                if prediction_result not in ["Model/Encoder not loaded", "Prediction Failed", "Prediction Failed (Unknown Label)"]:
                    st.success(f"**Predicted {prediction_target}:** {prediction_result}")
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    log_input = input_data.copy(); log_input['Latitude'] = lat_input; log_input['Longitude'] = lon_input # Log original None if applicable
                    log_prediction(timestamp, log_input, prediction_target, prediction_result)
                # Errors handled by predict_feature

    else:
        st.error("Could not load data options. Please check data file path.")

# === Location Lookup Tab ===
with tab_locate:
    st.header("Find Location from Coordinates")
    st.markdown("Enter the Latitude and Longitude to look up the approximate location (City, State, Country).")

    loc_col1, loc_col2 = st.columns(2)
    with loc_col1:
        lat_lookup = st.number_input("Enter Latitude", value=None, placeholder="e.g., 21.1458", format="%.4f", key='lat_lookup')
    with loc_col2:
        lon_lookup = st.number_input("Enter Longitude", value=None, placeholder="e.g., 79.0882", format="%.4f", key='lon_lookup')

    if st.button("Look Up Location", use_container_width=True):
        if lat_lookup is not None and lon_lookup is not None:
             with st.spinner("Looking up location... (May take a second due to rate limit)"):
                 geolocator = resources.get('geolocator')
                 location_result = lookup_location(lat_lookup, lon_lookup, geolocator)
                 st.info(location_result)
        else:
             st.warning("Please enter both Latitude and Longitude.")

# --- Sidebar Info ---
st.sidebar.header("About This App")
st.sidebar.markdown(f"""
Uses CatBoost models trained on GTD to predict attack features or uses Geopy for location lookup.
- **GroupName:** Predicts from Top 100 known groups.
- **WeaponType:** Predicts weapon using Top 100 groups as input.
- **AttackType:** Predicts attack type using Top 100 groups as input.
- **TargetType:** Predicts target type using Top 100 groups as input.
- **Location Lookup:** Uses Nominatim (OpenStreetMap) via Geopy.
""")
st.sidebar.header("How to Use")
st.sidebar.markdown("""
1.  **Run Locally:** `streamlit run app.py`
2.  **Deploy Free:** Use Streamlit Community Cloud + GitHub. Include all `.cbm` and `.pkl` files, and `requirements.txt` (add `geopy`).
""")