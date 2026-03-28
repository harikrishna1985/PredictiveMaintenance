import streamlit as st
import pandas as pd
from src.predict import predict_input
import yaml


def load_title():
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config["app"]["title"], config["app"]["threshold_note"]


title, note = load_title()

st.set_page_config(page_title=title, layout="centered")
st.title(title)
st.write("Enter the engine sensor values below to predict engine condition.")
st.info(note)

engine_rpm = st.number_input("Engine RPM", min_value=0.0, value=1500.0)
coolant_temperature = st.number_input("Coolant Temperature", min_value=0.0, value=85.0)
oil_pressure = st.number_input("Oil Pressure", min_value=0.0, value=45.0)
fuel_pressure = st.number_input("Fuel Pressure", min_value=0.0, value=55.0)
intake_temp = st.number_input("Intake Temperature", min_value=0.0, value=30.0)
battery_voltage = st.number_input("Battery Voltage", min_value=0.0, value=12.5)

if st.button("Predict"):
    input_data = {
        "engine_rpm": [engine_rpm],
        "coolant_temperature": [coolant_temperature],
        "oil_pressure": [oil_pressure],
        "fuel_pressure": [fuel_pressure],
        "intake_temp": [intake_temp],
        "battery_voltage": [battery_voltage],
    }

    input_df = pd.DataFrame(input_data)

    st.subheader("Input DataFrame")
    st.dataframe(input_df)

    result = predict_input(input_df)

    st.subheader("Prediction Result")
    st.write(f"Predicted Engine Condition: **{result['prediction']}**")

    if "probabilities" in result:
        st.subheader("Prediction Probabilities")
        st.write(result["probabilities"])
