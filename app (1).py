import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Define a function to load all saved artifacts
@st.cache_resource
def load_artifacts():
    try:
        best_model = joblib.load('best_model.joblib')
        feature_specs = joblib.load('feature_specs.joblib')
        le_y = joblib.load('le_y.joblib')
        fitted_iso_forest_models = joblib.load('fitted_iso_forest_models.joblib')
        capping_bounds = joblib.load('capping_bounds.joblib')
        X_train_capped_columns = joblib.load('X_train_capped_columns.joblib')

        # Identify numerical columns for outlier detection based on feature_specs that have capping bounds
        numerical_cols_for_outliers = [col for col, spec in feature_specs.items() if 'min_val' in spec and 'max_val' in spec]

        return best_model, feature_specs, le_y, fitted_iso_forest_models, capping_bounds, X_train_capped_columns, numerical_cols_for_outliers
    except FileNotFoundError as e:
        st.error(f"Error loading artifact: {e}. Make sure all .joblib files are in the same directory.")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred while loading artifacts: {e}")
        st.stop()

# Load artifacts
best_model, feature_specs, le_y, fitted_iso_forest_models, capping_bounds, X_train_capped_columns, numerical_cols_for_outliers = load_artifacts()

# 2. Set up the Streamlit application title and a brief description
st.title('KTAS_RN Level Prediction')
st.write('Enter patient details to predict the KTAS_RN (Korean Triage and Acuity Scale - Registered Nurse) level.')

# 3. Define a function to collect user input for each feature
def collect_user_input(feature_specs):
    user_inputs = {}
    st.sidebar.header('Patient Input Features')

    for feature_name, spec in feature_specs.items():
        dtype = spec['dtype']
        min_val = spec.get('min_val')
        max_val = spec.get('max_val')
        options = spec.get('options')

        if options:
            # For categorical/ordinal features with a limited set of options
            user_inputs[feature_name] = st.sidebar.selectbox(f"Select {feature_name}", options, format_func=lambda x: str(x))
        else:
            # For continuous numerical features
            if dtype == int:
                user_inputs[feature_name] = st.sidebar.number_input(f"Enter {feature_name}", min_value=int(min_val), max_value=int(max_val), value=int(np.mean([min_val, max_val]))) # Added a default value
            elif dtype == float:
                user_inputs[feature_name] = st.sidebar.number_input(f"Enter {feature_name}", min_value=float(min_val), max_value=float(max_val), value=float(np.mean([min_val, max_val]))) # Added a default value
    return pd.DataFrame([user_inputs])

# Collect user input
input_df = collect_user_input(feature_specs)

# 4. Define a function to preprocess the user input
def preprocess_input(input_df, numerical_cols_for_outliers, capping_bounds, fitted_iso_forest_models, X_train_capped_columns):
    input_df_processed = input_df.copy()

    # Apply capping
    for col in numerical_cols_for_outliers:
        if col in capping_bounds:
            lower_bound = capping_bounds[col]['lower']
            upper_bound = capping_bounds[col]['upper']
            input_df_processed[col] = np.clip(input_df_processed[col], lower_bound, upper_bound)

    # Apply Isolation Forest transformations (anomaly scores and outlier flags)
    for col in numerical_cols_for_outliers:
        if col in fitted_iso_forest_models:
            iso_forest_model = fitted_iso_forest_models[col]
            data_for_if_predict = input_df_processed[[col]].values

            input_df_processed[f'{col}_anomaly_score'] = iso_forest_model.decision_function(data_for_if_predict)
            input_df_processed[f'{col}_is_outlier'] = (iso_forest_model.predict(data_for_if_predict) == -1).astype(bool)

    # Drop columns that were excluded from X_train_capped (Chief_complain, Diagnosis in ED)
    # and any other columns not in X_train_capped_columns
    cols_to_drop_from_input = [col for col in input_df_processed.columns if col not in X_train_capped_columns]
    input_df_processed = input_df_processed.drop(columns=cols_to_drop_from_input)

    # Ensure the processed DataFrame has the exact same columns and order as X_train_capped_columns
    # Fill missing columns (if any) with 0, though given the previous steps, this should mainly handle order.
    input_df_processed = input_df_processed.reindex(columns=X_train_capped_columns, fill_value=0)

    return input_df_processed

# Preprocess user input
processed_input = preprocess_input(input_df, numerical_cols_for_outliers, capping_bounds, fitted_iso_forest_models, X_train_capped_columns)

# Display raw and processed inputs for debugging/transparency
st.subheader('Raw User Input')
st.write(input_df)

st.subheader('Processed Input for Model')
st.write(processed_input)

# 5. Make predictions and display the result
if st.button('Predict KTAS_RN Level'):
    prediction_encoded = best_model.predict(processed_input)
    prediction_ktas_rn = le_y.inverse_transform(prediction_encoded)

    st.subheader('Prediction Result')
    st.success(f"The predicted KTAS_RN level is: {prediction_ktas_rn[0]}")

    st.write("KTAS_RN Levels:")
    st.write("1: Resuscitation")
    st.write("2: Emergency")
    st.write("3: Urgent")
    st.write("4: Less Urgent")
    st.write("5: Non-Urgent")
