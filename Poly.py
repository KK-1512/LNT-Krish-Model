import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

st.title("Polynomial Regression – Heat Treatment Hardness Prediction")

# Load model safely
def load_model():
    base = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base, "poly_model.pkl")
    poly_path = os.path.join(base, "poly_features.pkl")

    model = pickle.load(open(model_path, "rb"))
    poly = pickle.load(open(poly_path, "rb"))

    return model, poly

model, poly = load_model()

# -----------------------------
# UI Inputs
# -----------------------------
st.header("Input Parameters")

C = st.number_input("Carbon (C%)", 0.10, 1.0, 0.3)
Mn = st.number_input("Manganese (Mn%)", 0.10, 2.0, 0.8)
Si = st.number_input("Silicon (Si%)", 0.00, 1.5, 0.25)
Cr = st.number_input("Chromium (Cr%)", 0.00, 3.0, 0.5)
Ni = st.number_input("Nickel (Ni%)", 0.00, 2.0, 0.4)
Mo = st.number_input("Molybdenum (Mo%)", 0.00, 1.0, 0.1)

AustenitizeTemp = st.number_input("Austenitize Temperature", 700, 1100, 850)
AustenitizeTime = st.number_input("Austenitize Time", 10, 180, 60)

QuenchMedium = st.selectbox("Quench Medium", ["Water", "Oil", "Polymer"])

TemperingTemp = st.number_input("Tempering Temperature", 100, 700, 300)
TemperingTime = st.number_input("Tempering Time", 10, 240, 90)

# -----------------------------
# Manual One-hot encoding for compatibility
# -----------------------------
qm_water = 1 if QuenchMedium == "Water" else 0
qm_oil = 1 if QuenchMedium == "Oil" else 0
# "Polymer" is baseline (0,0)

# -----------------------------
# Create input array
# -----------------------------
input_row = np.array([[
    C, Mn, Si, Cr, Ni, Mo,
    AustenitizeTemp, AustenitizeTime,
    qm_oil, qm_water,   # dummies
    TemperingTemp, TemperingTime
]])

# -----------------------------
# Predict
# -----------------------------
if st.button("Predict Hardness"):
    input_poly = poly.transform(input_row)
    pred = model.predict(input_poly)[0]
    st.success(f"Predicted Hardness: **{pred:.2f} HRC**")
