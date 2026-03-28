import streamlit as st
import pandas as pd
import yaml

from src.predict import predict_input


# =========================
# LOAD CONFIG
# =========================
def load_config():
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


config = load_config()

TITLE = config["app"]["title"]
SUBTITLE = config["app"]["subtitle"]
NOTE = config["app"]["threshold_note"]

INPUT_COLUMNS = config["features"]["input_columns"]


# =========================
# STREAMLIT UI CONFIG
# =========================
st.set_page_config(
    page_title=TITLE,
    layout="centered"
)

st.title(TITLE)
st.subheader(SUBTITLE)
st.info(NOTE)


# =========================
# INPUT SECTION
# =========================
st.markdown("### Enter Sensor Values")

inputs = {}

for col in INPUT_COLUMNS:
    label = col.replace("_", " ").title()

    default_value = 0.0

    if col == "engine_rpm":
        default_value = 1500.0
    elif col == "coolant_temperature":
        default_value = 85.0
    elif col == "oil_pressure":
        default_value = 45.0
    elif col == "fuel_pressure":
        default_value = 55.0
    elif col == "intake_temp":
        default_value = 30.0
    elif col == "battery_voltage":
        default_value = 12.5

    inputs[col] = st.number_input(
        label,
        min_value=0.0,
        value=float(default_value)
    )


# =========================
# PREDICTION BUTTON
# =========================
if st.button("Predict Engine Condition"):

    try:
        input_df = pd.DataFrame([inputs])

        st.markdown("### Input Data")
        st.dataframe(input_df)

        result = predict_input(input_df)

        # =========================
        # OUTPUT
        # =========================
        st.markdown("### Prediction Result")

        prediction = result["prediction"]

        if str(prediction).lower() in ["0", "healthy", "normal"]:
            st.success(f"Engine Condition: {prediction}")
        else:
            st.error(f"Engine Condition: {prediction}")

        # =========================
        # PROBABILITIES
        # =========================
        if "probabilities" in result:
            st.markdown("### Prediction Probabilities")
            prob_df = pd.DataFrame(
                [result["probabilities"]],
                columns=[f"Class {i}" for i in range(len(result["probabilities"]))]
            )
            st.dataframe(prob_df)

        # =========================
        # MODEL-READY INPUT
        # =========================
        st.markdown("### Model-Ready Features")
        st.json(result["processed_input"])

    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")


# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown(
    "This tool is for decision support only. Always validate predictions with expert inspection."
)
