import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os

def load_all():
    base = os.path.dirname(os.path.abspath(__file__))

    model = pickle.load(open(os.path.join(base, "poly_model.pkl"), "rb"))
    poly = pickle.load(open(os.path.join(base, "poly_features.pkl"), "rb"))
    scaler = pickle.load(open(os.path.join(base, "scaler.pkl"), "rb"))

    return model, poly, scaler

model, poly, scaler = load_all()

st.title("Polynomial Regression – Stable Hardness Predictor")

# Inputs
C = st.number_input("C", 0.1, 1.0, 0.3)
Mn = st.number_input("Mn", 0.1, 2.0, 0.8)
Si = st.number_input("Si", 0.0, 1.5, 0.25)
Cr = st.number_input("Cr", 0.0, 3.0, 0.5)
Ni = st.number_input("Ni", 0.0, 2.0, 0.4)
Mo = st.number_input("Mo", 0.0, 1.0, 0.1)

AustenitizeTemp = st.number_input("Austenitize Temp", 700, 1100, 850)
AustenitizeTime = st.number_input("Austenitize Time", 10, 180, 60)

Q = st.selectbox("Quench Medium", ["Water", "Oil", "Polymer"])

TemperingTemp = st.number_input("Tempering Temp", 100, 700, 300)
TemperingTime = st.number_input("Tempering Time", 10, 240, 90)

q_water = 1 if Q == "Water" else 0
q_oil = 1 if Q == "Oil" else 0

# Make row
row = np.array([[
    C, Mn, Si, Cr, Ni, Mo,
    AustenitizeTemp, AustenitizeTime,
    q_oil, q_water,
    TemperingTemp, TemperingTime
]])

# Predict
if st.button("Predict Hardness"):
    row_scaled = scaler.transform(row)
    row_poly = poly.transform(row_scaled)
    pred = model.predict(row_poly)[0]
    st.success(f"Predicted Hardness: {pred:.2f} HRC")
