import pandas as pd


REQUIRED_COLUMNS = [
    "engine_rpm",
    "coolant_temperature",
    "oil_pressure",
    "fuel_pressure",
    "intake_temp",
    "battery_voltage",
]


def validate_input(df: pd.DataFrame) -> None:
    """Check if all required columns are present"""
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Simple missing value handling"""
    return df.fillna(df.median(numeric_only=True))


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived features (if used in training)"""

    
    df["temp_pressure_ratio"] = df["coolant_temperature"] / (df["oil_pressure"] + 1)
    df["rpm_pressure_interaction"] = df["engine_rpm"] * df["oil_pressure"]

    return df


def ensure_column_order(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure correct feature order for model"""
    return df[sorted(df.columns)]


def preprocess_input(df: pd.DataFrame) -> pd.DataFrame:
    """Full preprocessing pipeline"""

    validate_input(df)

    df = handle_missing_values(df)

    df = feature_engineering(df)

    df = ensure_column_order(df)

    return df
