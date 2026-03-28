from huggingface_hub import hf_hub_download
from src.preprocess import preprocess_input
import joblib
import pandas as pd
import yaml


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_model():
    config = load_config()
    repo_id = config["model"]["repo_id"]
    filename = config["model"]["filename"]

    model_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename
    )
    model = joblib.load(model_path)
    return model



def predict_input(input_df):
    model = load_model()

    processed_df = preprocess_input(input_df)

    prediction = model.predict(processed_df)

    result = {"prediction": prediction[0]}

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(processed_df)
        result["probabilities"] = proba[0].tolist()

    return result
