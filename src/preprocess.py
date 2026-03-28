import pandas as pd


REQUIRED_COLUMNS = [
    "engine_rpm",
    "lub_oil_pressure",
    "fuel_pressure",
    "coolant_pressure",
    "lub_oil_temp",
    "coolant_temp",
]


def validate_input(df: pd.DataFrame) -> None:
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    return df.fillna(df.median(numeric_only=True))


def ensure_column_order(df: pd.DataFrame) -> pd.DataFrame:
    return df[REQUIRED_COLUMNS]


def preprocess_input(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

    validate_input(df)
    df = handle_missing_values(df)
    df = ensure_column_order(df)

    return df
