import streamlit as st
import pickle
import numpy as np
import pandas as pd


model = pickle.load(open("health_risk_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("Health Risk Level Prediction System")
st.write("Enter the patient's vitals to predict the risk level.")



num_features = [
    "Respiratory_Rate",
    "Oxygen_Saturation",
    "O2_Scale",
    "Systolic_BP",
    "Heart_Rate",
    "Temperature",
    "On_Oxygen"
]

cat_features = ["Consciousness"]


consciousness_map = {
    "A": 0,
    "P": 1,
    "V": 2,
    "U": 3
}


risk_map = {
    0: "Low",
    1: "Medium",
    2: "High"
}


st.header("Enter Patient Data")

user_data = {}

for feature in num_features:
    user_data[feature] = st.number_input(feature, min_value=0.0, step=0.1)


user_data["Consciousness"] = st.selectbox("Consciousness (A, P, V, U)", list(consciousness_map.keys()))


input_df = pd.DataFrame([user_data])

input_df["Consciousness"] = input_df["Consciousness"].map(consciousness_map)

input_df[num_features] = scaler.transform(input_df[num_features])


if st.button("Predict Risk Level"):

    pred_numeric = int(model.predict(input_df)[0])
    pred_label = risk_map[pred_numeric]

    st.subheader(f"Predicted Risk Level: **{pred_label}**")
