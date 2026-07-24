import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Page Configuration
st.set_page_config(page_title="Heart Disease Risk Predictor", page_icon="🫀", layout="centered")

st.title("🫀 Heart Disease Risk Prediction App")
st.write("Enter clinical parameters to check heart disease risk.")

# Load Assets
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('heart_disease_rf_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        return None, None

model, scaler = load_assets()

# Form Inputs
st.subheader("📋 Enter Patient Details:")
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age (Years)", min_value=20, max_value=100, value=50)
    sex_label = st.selectbox("Sex", options=["Female", "Male"])
    sex = 1 if sex_label == "Male" else 0

with col2:
    bp = st.number_input("Blood Pressure (BP in mmHg)", min_value=80, max_value=220, value=120)
    cholesterol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)

# Predict Button
if st.button("🔍 Predict Risk Status", type="primary", use_container_width=True):
    if model is not None and scaler is not None:
        raw_features = np.array([[age, sex, bp, cholesterol]])
        scaled_features = scaler.transform(raw_features)

        prediction = model.predict(scaled_features)[0]
        prob = model.predict_proba(scaled_features)[0]

        if prediction == 1:
            st.error(f"⚠️ High Risk Detected! ({prob[1]*100:.1f}% Risk)")
        else:
            st.success(f"✅ Low Risk / Healthy! ({prob[0]*100:.1f}% Confidence)")
    else:
        st.error("Model files missing!")
