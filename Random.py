import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load Model
import os
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "hardness_model.pkl")

model = pickle.load(open(model_path, "rb"))




st.title("Heat Treatment Hardness Prediction (ML Model)")

st.write("Predict final hardness based on alloy chemistry and heat-treatment parameters.")

# Inputs
C = st.number_input("Carbon (C %)", 0.1, 1.0, 0.3)
Mn = st.number_input("Manganese (Mn %)", 0.1, 2.0, 0.8)
Si = st.number_input("Silicon (Si %)", 0.0, 1.5, 0.25)
Cr = st.number_input("Chromium (Cr %)", 0.0, 3.0, 0.5)
Ni = st.number_input("Nickel (Ni %)", 0.0, 2.0, 0.4)
Mo = st.number_input("Molybdenum (Mo %)", 0.0, 1.0, 0.1)

AustenitizeTemp = st.number_input("Austenitizing Temperature (°C)", 700, 1100, 850)
AustenitizeTime = st.number_input("Austenitizing Time (min)", 10, 180, 45)

QuenchMedium = st.selectbox("Quench Medium", ["Water", "Oil", "Polymer"])

TemperingTemp = st.number_input("Tempering Temperature (°C)", 100, 700, 300)
TemperingTime = st.number_input("Tempering Time (min)", 10, 240, 60)

if st.button("Predict Hardness"):
    input_data = pd.DataFrame([[
        C, Mn, Si, Cr, Ni, Mo,
        AustenitizeTemp, AustenitizeTime,
        QuenchMedium,
        TemperingTemp, TemperingTime
    ]], columns=[
        "C","Mn","Si","Cr","Ni","Mo",
        "AustenitizeTemp","AustenitizeTime",
        "QuenchMedium",
        "TemperingTemp","TemperingTime"
    ])

    hardness = model.predict(input_data)[0]
    st.success(f"Predicted Hardness: **{hardness:.2f} HRC**")
