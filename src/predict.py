import json
import joblib
import pandas as pd
import yaml

from huggingface_hub import hf_hub_download
from src.preprocess import preprocess_input


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_model_and_info():
    config = load_config()

    repo_id = config["model"]["repo_id"]
    model_filename = config["model"]["filename"]
    info_filename = config["model"]["info_filename"]

    model_path = hf_hub_download(
        repo_id=repo_id,
        filename=model_filename,
        repo_type="model",
    )

    info_path = hf_hub_download(
        repo_id=repo_id,
        filename=info_filename,
        repo_type="model",
    )

    model = joblib.load(model_path)

    with open(info_path, "r", encoding="utf-8") as f:
        model_info = json.load(f)

    return model, model_info


def align_features_for_inference(input_df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    df = input_df.copy()

    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

    # apply one-hot encoding in case categoricals are introduced later
    df = pd.get_dummies(df, drop_first=False)

    # align to exact training feature set
    df = df.reindex(columns=feature_columns, fill_value=0)

    return df


def predict_input(input_df: pd.DataFrame) -> dict:
    model, model_info = load_model_and_info()

    processed_df = preprocess_input(input_df)

    feature_columns = model_info["feature_columns"]
    aligned_df = align_features_for_inference(processed_df, feature_columns)

    prediction = model.predict(aligned_df)

    result = {
        "prediction": prediction[0],
        "processed_input": aligned_df.to_dict(orient="records")[0],
    }

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(aligned_df)
        result["probabilities"] = proba[0].tolist()

    return result
