import streamlit as st
import pandas as pd
import yaml

from src.predict import predict_input


def load_config():
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


config = load_config()

TITLE = config["app"]["title"]
SUBTITLE = config["app"]["subtitle"]
NOTE = config["app"]["threshold_note"]

INPUT_COLUMNS = config["features"]["input_columns"]

st.set_page_config(page_title=TITLE, layout="centered")

st.title(TITLE)
st.subheader(SUBTITLE)
st.info(NOTE)

st.markdown("### Enter Sensor Values")

inputs = {}

default_values = {
    "engine_rpm": 1500.0,
    "lub_oil_pressure": 45.0,
    "fuel_pressure": 55.0,
    "coolant_pressure": 35.0,
    "lub_oil_temp": 80.0,
    "coolant_temp": 85.0,
}

for col in INPUT_COLUMNS:
    label = col.replace("_", " ").title()

    inputs[col] = st.number_input(
        label,
        min_value=0.0,
        value=float(default_values.get(col, 0.0))
    )

if st.button("Predict"):
    try:
        input_df = pd.DataFrame([inputs])

        st.markdown("### Input DataFrame")
        st.dataframe(input_df)

        result = predict_input(input_df)

        st.markdown("### Prediction Result")
        prediction = result["prediction"]

        if str(prediction).lower() in ["0", "healthy", "normal"]:
            st.success(f"Engine Condition: {prediction}")
        else:
            st.error(f"Engine Condition: {prediction}")

        if "probabilities" in result:
            st.markdown("### Prediction Probabilities")
            prob_df = pd.DataFrame(
                [result["probabilities"]],
                columns=[f"Class {i}" for i in range(len(result["probabilities"]))]
            )
            st.dataframe(prob_df)

        st.markdown("### Model-Ready Features")
        st.json(result["processed_input"])

    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")

st.markdown("---")
st.markdown("This tool is for decision support only. Always validate predictions with expert inspection.")
