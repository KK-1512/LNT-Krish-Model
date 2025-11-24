import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

st.set_page_config(page_title="Heat Treatment Hardness Predictor", layout="centered")

# =========================
# 1. Load Model Safely
# =========================
def load_model():
    base = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base, "hardness_model.pkl")

    if not os.path.exists(model_path):
        st.error("❌ Model file 'hardness_model.pkl' not found in repo.")
        return None

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    return model

model = load_model()

st.title("🔥 Heat Treatment Hardness Prediction")
st.write("Predict final **hardness (HRC)** based on alloy chemistry and heat-treatment parameters.")

if model is None:
    st.stop()

# =========================
# 2. User Inputs
# =========================
st.header("Enter Parameters")

col1, col2 = st.columns(2)

with col1:
    C = st.number_input("Carbon (C %)", 0.1, 1.0, 0.3)
    Mn = st.number_input("Manganese (Mn %)", 0.1, 2.0, 0.8)
    Si = st.number_input("Silicon (Si %)", 0.0, 1.5, 0.3)
    Cr = st.number_input("Chromium (Cr %)", 0.0, 3.0, 0.4)
    Ni = st.number_input("Nickel (Ni %)", 0.0, 2.0, 0.4)
    Mo = st.number_input("Molybdenum (Mo %)", 0.0, 1.0, 0.1)

with col2:
    AustenitizeTemp = st.number_input("Austenitizing Temperature (°C)", 700, 1200, 850)
    AustenitizeTime = st.number_input("Austenitizing Time (min)", 10, 200, 60)
    QuenchMedium = st.selectbox("Quench Medium", ["Water", "Oil", "Polymer"])
    TemperingTemp = st.number_input("Tempering Temperature (°C)", 100, 700, 300)
    TemperingTime = st.number_input("Tempering Time (min)", 10, 200, 60)

# =========================
# 3. Predict
# =========================
if st.button("Predict Hardness"):
    input_df = pd.DataFrame([[
        C, Mn, Si, Cr, Ni, Mo,
        AustenitizeTemp, AustenitizeTime,
        QuenchMedium,
        TemperingTemp, TemperingTime
    ]],
    columns=[
        "C","Mn","Si","Cr","Ni","Mo",
        "AustenitizeTemp","AustenitizeTime",
        "QuenchMedium",
        "TemperingTemp","TemperingTime"
    ])

    hardness = model.predict(input_df)[0]

    st.success(f"### 🔧 Predicted Hardness: **{hardness:.2f} HRC**")
