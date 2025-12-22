import streamlit as st
import numpy as np
import pickle
import os

st.title("Heat Treatment Hardness Predictor")
st.markdown("### _Done by **Krishnakumar**_")

# Optional image banner
st.image("LNT.jpg", use_column_width=True)

# Load model
def load_model():
    base = os.path.dirname(os.path.abspath(__file__))
    model = pickle.load(open(os.path.join(base, "tree_model.pkl"), "rb"))
    return model

model = load_model()

# Features (all now start from 0)
C = st.number_input("Carbon (C%)", min_value=0.0, max_value=2.0, value=0.30, step=0.001)
Mn = st.number_input("Manganese (Mn%)", min_value=0.0, max_value=3.0, value=0.80, step=0.001)
Si = st.number_input("Silicon (Si%)", min_value=0.0, max_value=2.0, value=0.25, step=0.001)
Cr = st.number_input("Chromium (Cr%)", min_value=0.0, max_value=5.0, value=0.50, step=0.001)
Ni = st.number_input("Nickel (Ni%)", min_value=0.0, max_value=5.0, value=0.40, step=0.001)
Mo = st.number_input("Molybdenum (Mo%)", min_value=0.0, max_value=2.0, value=0.10, step=0.001)

AustenitizeTemp = st.number_input("Austenitizing Temperature (°C)", min_value=0, max_value=1500, value=850)
AustenitizeTime = st.number_input("Austenitizing Time (min)", min_value=0, max_value=500, value=60)

Q = st.selectbox("Quench Medium", ["Water", "Oil", "Polymer"])

TemperingTemp = st.number_input("Tempering Temperature (°C)", min_value=0, max_value=800, value=300)
TemperingTime = st.number_input("Tempering Time (min)", min_value=0, max_value=500, value=90)

# Manual one-hot encoding
q_water = 1 if Q == "Water" else 0
q_oil   = 1 if Q == "Oil" else 0
# Polymer = (0,0)

row = np.array([[
    C, Mn, Si, Cr, Ni, Mo,
    AustenitizeTemp, AustenitizeTime,
    q_oil, q_water,
    TemperingTemp, TemperingTime
]])

if st.button("Predict Hardness"):
    pred = model.predict(row)[0]
    st.success(f"Predicted Hardness: {pred:.2f} HRC")
