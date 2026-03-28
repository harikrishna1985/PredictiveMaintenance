from huggingface_hub import hf_hub_download
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


def predict_input(input_df: pd.DataFrame):
    model = load_model()
    prediction = model.predict(input_df)

    result = {"prediction": prediction[0]}

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_df)
        result["probabilities"] = proba[0].tolist()

    return result
