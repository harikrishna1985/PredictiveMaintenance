import os
import json
from pathlib import Path

import joblib
import pandas as pd
from huggingface_hub import hf_hub_download, HfApi

from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import ParameterGrid

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False


# =========================
# CONFIG
# =========================
DATASET_REPO_ID = "harikrishna1985/Engine_data"
MODEL_REPO_ID = "harikrishna1985/predictive-maintenance-model"

TRAIN_FILENAME = "processed/train.csv"
TEST_FILENAME = "processed/test.csv"

TARGET_COLUMN = "engine_condition"

LOCAL_ARTIFACTS_DIR = Path("artifacts")
LOCAL_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_FILE = LOCAL_ARTIFACTS_DIR / "best_model.pkl"
RESULTS_FILE = LOCAL_ARTIFACTS_DIR / "tuning_results.csv"
BEST_MODEL_INFO_FILE = LOCAL_ARTIFACTS_DIR / "best_model_info.json"


# =========================
# HELPERS
# =========================
def get_hf_api() -> HfApi:
    token = os.getenv("HF_TOKEN")
    return HfApi(token=token)


def download_train_test() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path = hf_hub_download(
        repo_id=DATASET_REPO_ID,
        filename=TRAIN_FILENAME,
        repo_type="dataset",
    )
    test_path = hf_hub_download(
        repo_id=DATASET_REPO_ID,
        filename=TEST_FILENAME,
        repo_type="dataset",
    )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    return train_df, test_df


def prepare_features(train_df: pd.DataFrame, test_df: pd.DataFrame):
    target_col_clean = TARGET_COLUMN.strip().lower().replace(" ", "_")

    train_df.columns = [c.strip().lower().replace(" ", "_") for c in train_df.columns]
    test_df.columns = [c.strip().lower().replace(" ", "_") for c in test_df.columns]

    if target_col_clean not in train_df.columns or target_col_clean not in test_df.columns:
        raise ValueError(f"Target column '{target_col_clean}' not found in train/test data.")

    X_train = train_df.drop(columns=[target_col_clean])
    y_train = train_df[target_col_clean]

    X_test = test_df.drop(columns=[target_col_clean])
    y_test = test_df[target_col_clean]

    # keep common columns only, same order
    common_cols = [c for c in X_train.columns if c in X_test.columns]
    X_train = X_train[common_cols]
    X_test = X_test[common_cols]

    # one-hot encode categoricals if any
    X_train = pd.get_dummies(X_train, drop_first=False)
    X_test = pd.get_dummies(X_test, drop_first=False)

    X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

    return X_train, X_test, y_train, y_test


def build_model_candidates():
    candidates = {
        "decision_tree": {
            "model_class": DecisionTreeClassifier,
            "param_grid": {
                "max_depth": [3, 5, 10, None],
                "min_samples_split": [2, 5],
                "random_state": [42],
            },
        },
        "random_forest": {
            "model_class": RandomForestClassifier,
            "param_grid": {
                "n_estimators": [100, 200],
                "max_depth": [5, 10, None],
                "min_samples_split": [2, 5],
                "random_state": [42],
                "n_jobs": [-1],
            },
        },
        "adaboost": {
            "model_class": AdaBoostClassifier,
            "param_grid": {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.5, 1.0],
                "random_state": [42],
            },
        },
        "gradient_boosting": {
            "model_class": GradientBoostingClassifier,
            "param_grid": {
                "n_estimators": [100, 200],
                "learning_rate": [0.05, 0.1],
                "max_depth": [3, 5],
                "random_state": [42],
            },
        },
        "bagging": {
            "model_class": BaggingClassifier,
            "param_grid": {
                "n_estimators": [50, 100],
                "random_state": [42],
            },
        },
    }

    if XGBOOST_AVAILABLE:
        candidates["xgboost"] = {
            "model_class": XGBClassifier,
            "param_grid": {
                "n_estimators": [100, 200],
                "max_depth": [3, 5],
                "learning_rate": [0.05, 0.1],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0],
                "random_state": [42],
                "eval_metric": ["mlogloss"],
            },
        }

    return candidates


def train_and_tune(X_train, y_train, X_test, y_test):
    candidates = build_model_candidates()

    all_results = []
    best_model = None
    best_score = -1
    best_info = None

    for model_name, model_spec in candidates.items():
        model_class = model_spec["model_class"]
        grid = list(ParameterGrid(model_spec["param_grid"]))

        print(f"\nTraining model: {model_name}")
        print(f"Parameter combinations: {len(grid)}")

        for params in grid:
            try:
                model = model_class(**params)
                model.fit(X_train, y_train)

                preds = model.predict(X_test)

                acc = accuracy_score(y_test, preds)
                f1 = f1_score(y_test, preds, average="weighted")

                row = {
                    "model_name": model_name,
                    "params": json.dumps(params),
                    "accuracy": acc,
                    "f1_weighted": f1,
                }
                all_results.append(row)

                if f1 > best_score:
                    best_score = f1
                    best_model = model
                    best_info = {
                        "model_name": model_name,
                        "params": params,
                        "accuracy": acc,
                        "f1_weighted": f1,
                        "feature_columns": X_train.columns.tolist(),
                        "target_column": TARGET_COLUMN.strip().lower().replace(" ", "_"),
                    }

                print(f"{model_name} | params={params} | acc={acc:.4f} | f1={f1:.4f}")

            except Exception as e:
                print(f"Skipping params due to error: {params} | error={e}")

    if best_model is None or best_info is None:
        raise RuntimeError("No model was trained successfully.")

    results_df = pd.DataFrame(all_results).sort_values(by="f1_weighted", ascending=False)
    results_df.to_csv(RESULTS_FILE, index=False)

    joblib.dump(best_model, BEST_MODEL_FILE)
    with open(BEST_MODEL_INFO_FILE, "w", encoding="utf-8") as f:
        json.dump(best_info, f, indent=2)

    print(f"\nBest model saved to: {BEST_MODEL_FILE}")
    print(f"Tuning results saved to: {RESULTS_FILE}")
    print(f"Best model info saved to: {BEST_MODEL_INFO_FILE}")
    print(f"Best model: {best_info['model_name']} | f1={best_info['f1_weighted']:.4f}")

    return best_model, best_info


def upload_model_artifacts():
    api = get_hf_api()

    files_to_upload = [
        (str(BEST_MODEL_FILE), "best_model.pkl"),
        (str(RESULTS_FILE), "tuning_results.csv"),
        (str(BEST_MODEL_INFO_FILE), "best_model_info.json"),
    ]

    for local_file, path_in_repo in files_to_upload:
        print(f"Uploading {local_file} -> {path_in_repo}")
        api.upload_file(
            path_or_fileobj=local_file,
            path_in_repo=path_in_repo,
            repo_id=MODEL_REPO_ID,
            repo_type="model",
        )

    print("Best model and tuning artifacts uploaded successfully to HF model repo.")


def main():
    train_df, test_df = download_train_test()
    X_train, X_test, y_train, y_test = prepare_features(train_df, test_df)
    train_and_tune(X_train, y_train, X_test, y_test)
    upload_model_artifacts()
    print("Training completed successfully.")


if __name__ == "__main__":
    main()
