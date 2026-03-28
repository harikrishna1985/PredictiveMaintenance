import os
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from huggingface_hub import hf_hub_download, HfApi


# =========================
# CONFIG
# =========================
DATASET_REPO_ID = "harikrishna1985/Engine_data"
RAW_FILENAME = "raw/engine_data.csv"   # change if needed

TARGET_COLUMN = "engine_condition"

# columns to drop if unnecessary
DROP_COLUMNS = [
    # "unnamed: 0",
    # "id",
]

TEST_SIZE = 0.2
RANDOM_STATE = 42

LOCAL_DATA_DIR = Path("data")
LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = LOCAL_DATA_DIR / "train.csv"
TEST_FILE = LOCAL_DATA_DIR / "test.csv"
CLEAN_FILE = LOCAL_DATA_DIR / "cleaned_data.csv"
METADATA_FILE = LOCAL_DATA_DIR / "prep_metadata.json"


# =========================
# HELPERS
# =========================
def get_hf_api() -> HfApi:
    token = os.getenv("HF_TOKEN")
    return HfApi(token=token)


def load_raw_data_from_hf() -> pd.DataFrame:
    print(f"Downloading raw dataset from HF dataset repo: {DATASET_REPO_ID}")
    local_path = hf_hub_download(
        repo_id=DATASET_REPO_ID,
        filename=RAW_FILENAME,
        repo_type="dataset",
    )
    df = pd.read_csv(local_path)
    print(f"Raw data shape: {df.shape}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # standardize column names
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

    # align target/drop names with cleaned columns
    drop_cols_clean = [c.strip().lower().replace(" ", "_") for c in DROP_COLUMNS]
    target_col_clean = TARGET_COLUMN.strip().lower().replace(" ", "_")

    # drop unwanted columns if present
    cols_to_drop = [c for c in drop_cols_clean if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    # remove duplicates
    df = df.drop_duplicates()

    # remove rows with missing target
    if target_col_clean not in df.columns:
        raise ValueError(f"Target column '{target_col_clean}' not found in dataset columns: {list(df.columns)}")

    df = df.dropna(subset=[target_col_clean])

    # fill numeric missing values with median
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if numeric_cols:
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # fill non-numeric missing values with mode if possible
    non_numeric_cols = [c for c in df.columns if c not in numeric_cols]
    for col in non_numeric_cols:
        if df[col].isna().sum() > 0:
            mode_vals = df[col].mode()
            fill_value = mode_vals.iloc[0] if not mode_vals.empty else "unknown"
            df[col] = df[col].fillna(fill_value)

    print(f"Cleaned data shape: {df.shape}")
    return df


def split_and_save(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    target_col_clean = TARGET_COLUMN.strip().lower().replace(" ", "_")

    # stratify if target is classification-friendly
    stratify_arg = df[target_col_clean] if df[target_col_clean].nunique() <= 20 else None

    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify_arg,
    )

    df.to_csv(CLEAN_FILE, index=False)
    train_df.to_csv(TRAIN_FILE, index=False)
    test_df.to_csv(TEST_FILE, index=False)

    metadata = {
        "dataset_repo_id": DATASET_REPO_ID,
        "raw_filename": RAW_FILENAME,
        "target_column": target_col_clean,
        "drop_columns": DROP_COLUMNS,
        "cleaned_shape": list(df.shape),
        "train_shape": list(train_df.shape),
        "test_shape": list(test_df.shape),
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
    }

    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved cleaned data to: {CLEAN_FILE}")
    print(f"Saved train data to: {TRAIN_FILE}")
    print(f"Saved test data to: {TEST_FILE}")

    return train_df, test_df


def upload_prepared_files_to_hf() -> None:
    api = get_hf_api()

    files_to_upload = [
        (str(CLEAN_FILE), "processed/cleaned_data.csv"),
        (str(TRAIN_FILE), "processed/train.csv"),
        (str(TEST_FILE), "processed/test.csv"),
        (str(METADATA_FILE), "processed/prep_metadata.json"),
    ]

    for local_file, path_in_repo in files_to_upload:
        print(f"Uploading {local_file} -> {path_in_repo}")
        api.upload_file(
            path_or_fileobj=local_file,
            path_in_repo=path_in_repo,
            repo_id=DATASET_REPO_ID,
            repo_type="dataset",
        )

    print("Prepared dataset files uploaded successfully to HF dataset repo.")


def main():
    df = load_raw_data_from_hf()
    df = clean_data(df)
    split_and_save(df)
    upload_prepared_files_to_hf()
    print("Data preparation completed successfully.")


if __name__ == "__main__":
    main()
