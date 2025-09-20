
import streamlit as st
import joblib
import pandas as pd
from PIL import Image
import base64
from io import BytesIO
import numpy as np
import io

# Utility functions
def img_to_base64(img):
    """Converts a PIL Image to a base64 string."""
    buffered = BytesIO()
    img.save(buffered, format='PNG')  # Using PNG for better quality/transparency support
    img_bytes = buffered.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode()
    return img_base64

def convert_df_to_csv_bytes(df):
    """Converts a DataFrame to CSV bytes for downloading."""
    return df.to_csv(index=False).encode('utf-8')

# Functions to create sample telecom dummy datasets
def create_dummy_data_2017():
    data = []
    for i in range(20):
        data.append({
            'account_length': 100 + i,
            'voice_mail_plan': i % 2,
            'voice_mail_messages': i * 2,
            'day_mins': 200 + i*3,
            'evening_mins': 150 + i*2,
            'night_mins': 100 + i,
            'international_mins': 10 + i//2,
            'customer_service_calls': i % 4,
            'international_plan': 0,
            'day_calls': 80 + i,
            'day_charge': 30.0 + i*1.2,
            'evening_calls': 70 + i,
            'evening_charge': 10.0 + i*0.5,
            'night_calls': 50 + i,
            'night_charge': 5.0 + i*0.3,
            'international_calls': 2 + i//4,
            'international_charge': 2.0 + i*0.1,
            'total_charge': 47.0 + i*2,
            'total_mins': 460.0 + i*6,
            'total_calls': 202 + i*3,
            'mins_per_call': 2.3 + (i*0.05)
        })
    return pd.DataFrame(data)

def create_dummy_data_2018():
    data = []
    for i in range(20):
        data.append({
            'account_length': 120 + i,
            'voice_mail_plan': (i+1) % 2,
            'voice_mail_messages': i,
            'day_mins': 180 + i*2.5,
            'evening_mins': 160 + i*1.5,
            'night_mins': 110 + i*1.1,
            'international_mins': 1 + i//3,
            'customer_service_calls': (i+2) % 3,
            'international_plan': 1 if i % 5 == 0 else 0,
            'day_calls': 75 + i,
            'day_charge': 27.0 + i*1.0,
            'evening_calls': 80 + i,
            'evening_charge': 11.0 + i*0.6,
            'night_calls': 55 + i,
            'night_charge': 6.0 + i*0.2,
            'international_calls': 1 + i//5,
            'international_charge': 1.0 + i*0.2,
            'total_charge': 45.0 + i*1.8,
            'total_mins': 451.0 + i*5,
            'total_calls': 211 + i*4,
            'mins_per_call': 2.14 + (i*0.04)
        })
    return pd.DataFrame(data)

def create_dummy_data_telco_x():
    data = []
    for i in range(20):
        data.append({
            'account_length': 110 + i,
            'voice_mail_plan': int(i % 3 == 0),
            'voice_mail_messages': i + 4,
            'day_mins': 210 + i*2,
            'evening_mins': 140 + i*3,
            'night_mins': 100 + i*2,
            'international_mins': 5 + i//3,
            'customer_service_calls': i % 5,
            'international_plan': 0 if i % 4 != 0 else 1,
            'day_calls': 70 + i,
            'day_charge': 25.0 + i*1.4,
            'evening_calls': 60 + i,
            'evening_charge': 9.0 + i*0.6,
            'night_calls': 40 + i,
            'night_charge': 4.0 + i*0.25,
            'international_calls': 3 + i//4,
            'international_charge': 1.5 + i*0.2,
            'total_charge': 42.0 + i*2.1,
            'total_mins': 430.0 + i*7,
            'total_calls': 190 + i*5,
            'mins_per_call': 2.2 + (i * 0.03)
        })
    return pd.DataFrame(data)

def create_dummy_data_telco_y():
    data = []
    for i in range(20):
        data.append({
            'account_length': 105 + i,
            'voice_mail_plan': int((i+1) % 2),
            'voice_mail_messages': (i + 3) * 2,
            'day_mins': 205 + i*3.5,
            'evening_mins': 145 + i*1.8,
            'night_mins': 95 + i*1.2,
            'international_mins': 8 + i//4,
            'customer_service_calls': (i+1) % 3,
            'international_plan': 0,
            'day_calls': 85 + i,
            'day_charge': 28.0 + i*1.1,
            'evening_calls': 75 + i,
            'evening_charge': 12.0 + i*0.4,
            'night_calls': 45 + i,
            'night_charge': 4.5 + i*0.35,
            'international_calls': 2 + i//5,
            'international_charge': 2.0 + i*0.15,
            'total_charge': 46.0 + i*1.9,
            'total_mins': 455.0 + i*5.5,
            'total_calls': 200 + i*3,
            'mins_per_call': 2.25 + (i * 0.02)
        })
    return pd.DataFrame(data)

# App configuration
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Banner and Images ---
banner_image_path = "bunny4.png"
sidebar_image_path = "bunny.png"
banner_base64 = None
try:
    banner_img = Image.open(banner_image_path)
    banner_base64 = img_to_base64(banner_img)
    sidebar_img = Image.open(sidebar_image_path)
except FileNotFoundError:
    st.error("Image files (bunny4.png, bunny.png) not found. Please check the file paths.")
    sidebar_img = None

# Corrected Banner Implementation
if banner_base64:
    st.markdown(f"""
    <style>
    .banner {{
        background-image: url("data:image/png;base64,{banner_base64}");
        background-size: cover;
        background-position: center;
        padding: 50px 0px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 20px;
    }}
    .banner-text {{
        font-size: 2.5em;
        font-weight: bold;
        color: white;
        text-shadow: 2px 2px 4px #000000;
    }}
    </style>
    """, unsafe_allow_html=True)
    st.markdown('<div class="banner"><div class="banner-text">Customer Churn Predictor 🔮</div></div>', unsafe_allow_html=True)
else:
    # Fallback title if image fails to load
    st.title("Customer Churn Predictor 🔮")

# --- Initial Disclaimer and Sample Data Download ---
if 'popup_shown' not in st.session_state:
    st.markdown("### Disclaimer")
    st.markdown("""
        ** 1. Welcome to the Customer Churn Predictor!** This is a ML app. Predictions are from a historical data trained model - GB45.
        2. Please select one sample dataset below and download it if you want to try uploading sample data for prediction.
        
        3. Select dataset and click download to get sample telecom customer records.
        """)
    
    sample_datasets = {
        "Telecom Data 2017": create_dummy_data_2017(),
        "Telecom Data 2018": create_dummy_data_2018(),
        "TelcoX Sample": create_dummy_data_telco_x(),
        "TelcoY Sample": create_dummy_data_telco_y(),
    }
    selected_ds_name = st.selectbox("Select a Sample Dataset to Download", options=list(sample_datasets.keys()))
    
    csv_bytes = convert_df_to_csv_bytes(sample_datasets[selected_ds_name])
    st.download_button(
        label=f"Download {selected_ds_name} CSV",
        data=csv_bytes,
        file_name=f"{selected_ds_name.replace(' ', '_').lower()}.csv",
        mime="text/csv"
    )
    if st.button("I Understand and Agree"):
        st.session_state.popup_shown = True
        st.rerun()

# --- Main Application Logic ---
if st.session_state.get('popup_shown', False):
    @st.cache_resource
    def load_model():
        """Loads the trained model, caching it for performance."""
        try:
            with open('churn_prediction_model.pkl', 'rb') as f:
                model = joblib.load(f)
            return model
        except FileNotFoundError:
            st.error("Model file 'churn_prediction_model.pkl' not found.")
            return None

    best_model = load_model()

    if best_model:
        st.sidebar.header('👤 Customer Details')
        if sidebar_img:
            st.sidebar.image(sidebar_img, caption="Connected Customers")  # Removed use_container_width
        
        st.sidebar.markdown("---")
        
        features = ['account_length', 'voice_mail_plan', 'voice_mail_messages', 'day_mins', 
                    'evening_mins', 'night_mins', 'international_mins', 'customer_service_calls',
                    'international_plan', 'day_calls', 'day_charge', 'evening_calls', 'evening_charge',
                    'night_calls', 'night_charge', 'international_calls', 'international_charge',
                    'total_charge', 'total_mins', 'total_calls', 'mins_per_call']
        categorical_features = ['voice_mail_plan', 'international_plan']

        def zero_out_mins_and_charges(df):
            """Recalculates totals and zeros out mins/charges if calls are zero."""
            call_mappings = [
                ('day_calls', 'day_mins', 'day_charge'),
                ('evening_calls', 'evening_mins', 'evening_charge'),
                ('night_calls', 'night_mins', 'night_charge'),
                ('international_calls', 'international_mins', 'international_charge')
            ]
            for calls_col, mins_col, charge_col in call_mappings:
                if calls_col in df.columns:
                    zero_mask = df[calls_col] == 0
                    if mins_col in df.columns: df.loc[zero_mask, mins_col] = 0
                    if charge_col in df.columns: df.loc[zero_mask, charge_col] = 0
            
            # Recalculate totals
            df['total_calls'] = df[[c for c,_,_ in call_mappings if c in df.columns]].sum(axis=1)
            df['total_mins'] = df[[m for _,m,_ in call_mappings if m in df.columns]].sum(axis=1)
            df['mins_per_call'] = (df['total_mins'] / df['total_calls']).fillna(0)
            return df

        # --- File Upload Section ---
        uploaded_file = st.sidebar.file_uploader("Upload a CSV file with customer data", type=['csv'])
        if uploaded_file is not None:
            df_uploaded = pd.read_csv(uploaded_file)
            df_uploaded.columns = df_uploaded.columns.str.strip().str.lower().str.replace(' ', '_')
            for cat_feat in categorical_features:
                if cat_feat in df_uploaded.columns:
                    df_uploaded[cat_feat] = df_uploaded[cat_feat].map({'Yes': 1, 'No': 0, 'yes': 1, 'no': 0, 1: 1, 0: 0}).fillna(df_uploaded[cat_feat])
            
            df_processed = zero_out_mins_and_charges(df_uploaded.copy())
            df_model_ready = df_processed.reindex(columns=features).fillna(0)
            preds = best_model.predict(df_model_ready)
            proba = best_model.predict_proba(df_model_ready)[:, 1] * 100
            df_uploaded['churn_prediction'] = ['Churn' if x == 1 else 'Stay' for x in preds]
            df_uploaded['churn_probability (%)'] = proba.round(2)

            st.markdown("### Uploaded Data with Predictions")
            st.dataframe(df_uploaded)
            csv_data = convert_df_to_csv_bytes(df_uploaded)
            st.download_button(
                label="Download CSV with Predictions",
                data=csv_data,
                file_name="predictions.csv",
                mime="text/csv"
            )

        st.markdown("---")
        st.markdown("## Enter Customer Details Manually")

        # --- Manual Input Form ---
        with st.form(key='manual_input_form'):
            manual_inputs = {}
            cols = st.columns(3)
            field_layout = {
                'account_length': 0, 'customer_service_calls': 0, 'voice_mail_plan': 0, 'voice_mail_messages': 0, 'international_plan': 0,
                'day_calls': 1, 'day_mins': 1, 'day_charge': 1, 'evening_calls': 1, 'evening_mins': 1, 'evening_charge': 1,
                'night_calls': 2, 'night_mins': 2, 'night_charge': 2, 'international_calls': 2, 'international_mins': 2, 'international_charge': 2
            }
            
            # Define which features should be integers
            integer_features = [
                'account_length', 'voice_mail_messages', 'day_calls', 'night_calls',
                'customer_service_calls', 'evening_calls', 'international_calls'
            ]
            
            for feature in features:
                if feature not in field_layout:
                    continue
                col_index = field_layout[feature]
                
                with cols[col_index]:
                    if feature in categorical_features:
                        val = st.radio(f"{feature.replace('_',' ').title()}", ("No", "Yes"), horizontal=True)
                        manual_inputs[feature] = 1 if val == "Yes" else 0
                    elif feature in integer_features:
                        manual_inputs[feature] = st.number_input(
                            f"{feature.replace('_',' ').title()}",
                            min_value=0,
                            value=0,
                            step=1,
                            format="%d"
                        )
                    else:
                        manual_inputs[feature] = st.number_input(
                            f"{feature.replace('_',' ').title()}",
                            min_value=0.0,
                            value=0.0,
                            format="%.2f"
                        )
            submit_manual = st.form_submit_button("Predict Churn")

        if submit_manual:
            df_manual = pd.DataFrame([manual_inputs])
            df_processed_manual = zero_out_mins_and_charges(df_manual.copy())
            df_model_ready_manual = df_processed_manual.reindex(columns=features).fillna(0)
            preds_manual = best_model.predict(df_model_ready_manual)
            proba_manual = best_model.predict_proba(df_model_ready_manual)[:, 1] * 100
            prediction = 'Churn' if preds_manual[0] == 1 else 'Stay'
            probability = proba_manual[0]
            st.markdown("### Manual Entry Prediction Result")
            if prediction == "Churn":
                st.error(f"Prediction: **{prediction}** with **{probability:.2f}%** probability.")
            else:
                st.success(f"Prediction: **{prediction}** with **{100-probability:.2f}%** probability.")
            
            st.write("Based on the details provided:")
            st.dataframe(df_processed_manual)
