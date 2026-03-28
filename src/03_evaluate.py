import json
from pathlib import Path

import joblib
import pandas as pd
from huggingface_hub import hf_hub_download
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


# =========================
# CONFIG
# =========================
DATASET_REPO_ID = "harikrishna1985/Engine_data"
MODEL_REPO_ID = "harikrishna1985/predictive-maintenance-model"

TEST_FILENAME = "processed/test.csv"
MODEL_FILENAME = "best_model.pkl"
MODEL_INFO_FILENAME = "best_model_info.json"

TARGET_COLUMN = "engine_condition"

LOCAL_EVAL_DIR = Path("artifacts")
LOCAL_EVAL_DIR.mkdir(parents=True, exist_ok=True)

EVAL_SUMMARY_FILE = LOCAL_EVAL_DIR / "evaluation_summary.json"
CLASSIFICATION_REPORT_FILE = LOCAL_EVAL_DIR / "classification_report.csv"
CONFUSION_MATRIX_FILE = LOCAL_EVAL_DIR / "confusion_matrix.csv"


def load_test_data():
    test_path = hf_hub_download(
        repo_id=DATASET_REPO_ID,
        filename=TEST_FILENAME,
        repo_type="dataset",
    )
    test_df = pd.read_csv(test_path)
    test_df.columns = [c.strip().lower().replace(" ", "_") for c in test_df.columns]
    return test_df


def load_model_and_info():
    model_path = hf_hub_download(
        repo_id=MODEL_REPO_ID,
        filename=MODEL_FILENAME,
        repo_type="model",
    )
    info_path = hf_hub_download(
        repo_id=MODEL_REPO_ID,
        filename=MODEL_INFO_FILENAME,
        repo_type="model",
    )

    model = joblib.load(model_path)
    with open(info_path, "r", encoding="utf-8") as f:
        model_info = json.load(f)

    return model, model_info


def prepare_test_features(test_df: pd.DataFrame, feature_columns: list[str]):
    target_col_clean = TARGET_COLUMN.strip().lower().replace(" ", "_")

    if target_col_clean not in test_df.columns:
        raise ValueError(f"Target column '{target_col_clean}' missing in test data.")

    X_test = test_df.drop(columns=[target_col_clean])
    y_test = test_df[target_col_clean]

    X_test = pd.get_dummies(X_test, drop_first=False)

    # align to training features
    X_test = X_test.reindex(columns=feature_columns, fill_value=0)

    return X_test, y_test


def evaluate():
    test_df = load_test_data()
    model, model_info = load_model_and_info()

    feature_columns = model_info["feature_columns"]

    X_test, y_test = prepare_test_features(test_df, feature_columns)

    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="weighted")

    report = classification_report(y_test, preds, output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    labels = sorted(y_test.astype(str).unique().tolist())
    cm = confusion_matrix(y_test.astype(str), pd.Series(preds).astype(str), labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    summary = {
        "model_name": model_info.get("model_name"),
        "params": model_info.get("params"),
        "accuracy": acc,
        "f1_weighted": f1,
    }

    with open(EVAL_SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    report_df.to_csv(CLASSIFICATION_REPORT_FILE, index=True)
    cm_df.to_csv(CONFUSION_MATRIX_FILE, index=True)

    print("Evaluation completed.")
    print(json.dumps(summary, indent=2))
    print(f"Saved: {EVAL_SUMMARY_FILE}")
    print(f"Saved: {CLASSIFICATION_REPORT_FILE}")
    print(f"Saved: {CONFUSION_MATRIX_FILE}")


if __name__ == "__main__":
    evaluate()
