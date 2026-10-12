"""Data preprocessing and feature pipeline for MindBloom AI Stress Prediction.

Handles dataset loading, cleaning, feature transformation (numerical and categorical),
and train-test splitting without data leakage.
"""
from pathlib import Path
import re
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

# Target column candidates
TARGET_CANDIDATES = [
    "Overall, how would you rate your current stress level?",
    "Stress Level",
    "Stress level",
    "stress_level",
    "Predicted Stress Level",
    "Level of Stress",
]

# Canonical 8 survey features
FEATURE_COLUMNS = [
    "Age",
    "Year of Study",
    "How often do you feel stressed?",
    "What is the main cause of your stress?",
    "How many hours do you sleep on average?",
    "How often do you exercise or do physical activity?",
    "Have you experienced any of the following due to stress?",
    "How do you usually cope with stress?",
]

# Explicit types
NUMERIC_FEATURE_NAMES = [
    "Age",
    "Year of Study",
    "How many hours do you sleep on average?",
]

CATEGORICAL_FEATURE_NAMES = [
    "How often do you feel stressed?",
    "What is the main cause of your stress?",
    "How often do you exercise or do physical activity?",
    "Have you experienced any of the following due to stress?",
    "How do you usually cope with stress?",
]

STRESS_LABELS = {
    1: "Very Low",
    2: "Low",
    3: "Moderate",
    4: "High",
    5: "Very High",
}


def find_dataset_file():
    """Locate the dataset file across standard project locations."""
    candidate_paths = [
        BASE_DIR / "data" / "processed" / "cleaned_student_stress_data.xlsx",
        BASE_DIR / "data" / "cleaned_student_stress_data.xlsx",
        BASE_DIR / "cleaned_student_stress_data.xlsx",
    ]
    # Check for any xlsx/csv in data directory
    for path in candidate_paths:
        if path.is_file():
            return path

    data_dir = BASE_DIR / "data"
    if data_dir.is_dir():
        for ext in ("*.xlsx", "*.xls", "*.csv"):
            for match in data_dir.rglob(ext):
                if not match.name.startswith("~") and not match.name.startswith("."):
                    return match

    # Search in root
    for ext in ("*.xlsx", "*.xls", "*.csv"):
        for match in BASE_DIR.glob(ext):
            if "model" not in match.name.lower() and not match.name.startswith("~"):
                return match

    raise FileNotFoundError("Could not find student stress dataset in project folder.")


def clean_column_names(df):
    """Strip whitespace and newlines from column names."""
    df = df.copy()
    df.columns = [str(c).strip().replace("\n", "").replace("\r", "") for c in df.columns]
    # Remove unnamed spreadsheet index columns
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed:\s*\d+$", case=False)]
    return df


def parse_numeric_age(value):
    """Convert survey age range or raw number into continuous numerical age."""
    if pd.isna(value):
        return np.nan
    s = str(value).strip().lower()
    if "below 18" in s or "<18" in s:
        return 17.0
    if "18" in s and "20" in s:
        return 19.0
    if "21" in s and "22" in s:
        return 21.5
    if "above 23" in s or ">23" in s:
        return 24.0
    try:
        val = float(s)
        return val if 10.0 <= val <= 100.0 else np.nan
    except (ValueError, TypeError):
        return np.nan


def parse_numeric_sleep(value):
    """Convert survey sleep range or raw hours into continuous numerical sleep hours."""
    if pd.isna(value):
        return np.nan
    s = str(value).strip().lower()
    if "less than 5" in s or "<5" in s or "< 5" in s:
        return 4.5
    if "5" in s and "6" in s:
        return 5.5
    if "6" in s and "8" in s:
        return 7.0
    if "more than 8" in s or ">8" in s or "> 8" in s:
        return 8.5
    try:
        val = float(s)
        return val if 0.0 <= val <= 24.0 else np.nan
    except (ValueError, TypeError):
        return np.nan


def parse_numeric_year(value):
    """Convert year of study string or number into continuous/ordinal number (1-5)."""
    if pd.isna(value):
        return np.nan
    s = str(value).strip().lower()
    for year_idx in range(1, 6):
        if str(year_idx) in s:
            return float(year_idx)
    try:
        val = float(s)
        return val if 1.0 <= val <= 6.0 else np.nan
    except (ValueError, TypeError):
        return np.nan


def normalize_target_values(values):
    """Convert target column entries into integers 1-5."""
    mapping = {
        "1": 1, "very low": 1, "very low stress": 1, "level 1": 1,
        "2": 2, "low": 2, "low stress": 2, "level 2": 2,
        "3": 3, "moderate": 3, "moderate stress": 3, "medium": 3, "medium stress": 3, "level 3": 3,
        "4": 4, "high": 4, "high stress": 4, "level 4": 4,
        "5": 5, "very high": 5, "very high stress": 5, "level 5": 5,
    }

    def convert_single(v):
        if pd.isna(v):
            return np.nan
        try:
            num = float(v)
            if num.is_integer() and int(num) in (1, 2, 3, 4, 5):
                return int(num)
        except (ValueError, TypeError):
            pass
        return mapping.get(str(v).strip().casefold(), np.nan)

    res = pd.Series(values).map(convert_single)
    return res


def transform_features(df):
    """Transform raw student DataFrame or dict into preprocessed features.

    Numerical columns: continuous float values.
    Categorical columns: clean standardized string values.
    """
    if isinstance(df, dict):
        df = pd.DataFrame([df])
    else:
        df = df.copy()

    df = clean_column_names(df)

    # Ensure required columns are present (fill with NaN if missing)
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    out = pd.DataFrame(index=df.index)

    # 1. Age (Continuous numerical: Gaussian likelihood)
    out["Age"] = df["Age"].apply(parse_numeric_age)

    # 2. Year of Study (Continuous/ordinal numerical: Gaussian likelihood)
    out["Year of Study"] = df["Year of Study"].apply(parse_numeric_year)

    # 3. How often do you feel stressed? (Categorical: Laplace likelihood)
    out["How often do you feel stressed?"] = (
        df["How often do you feel stressed?"]
        .map(lambda v: str(v).strip() if pd.notna(v) and str(v).strip() != "" else "Unknown")
    )

    # 4. What is the main cause of your stress? (Categorical: Laplace likelihood)
    out["What is the main cause of your stress?"] = (
        df["What is the main cause of your stress?"]
        .map(lambda v: str(v).strip() if pd.notna(v) and str(v).strip() != "" else "Unknown")
    )

    # 5. How many hours do you sleep on average? (Continuous numerical: Gaussian likelihood)
    out["How many hours do you sleep on average?"] = (
        df["How many hours do you sleep on average?"].apply(parse_numeric_sleep)
    )

    # 6. How often do you exercise or do physical activity? (Categorical: Laplace likelihood)
    out["How often do you exercise or do physical activity?"] = (
        df["How often do you exercise or do physical activity?"]
        .map(lambda v: str(v).strip() if pd.notna(v) and str(v).strip() != "" else "Unknown")
    )

    # 7. Have you experienced any of the following due to stress? (Categorical: Laplace likelihood)
    out["Have you experienced any of the following due to stress?"] = (
        df["Have you experienced any of the following due to stress?"]
        .map(lambda v: str(v).strip() if pd.notna(v) and str(v).strip() != "" else "None")
    )

    # 8. How do you usually cope with stress? (Categorical: Laplace likelihood)
    out["How do you usually cope with stress?"] = (
        df["How do you usually cope with stress?"]
        .map(lambda v: str(v).strip() if pd.notna(v) and str(v).strip() != "" else "None")
    )

    return out[FEATURE_COLUMNS]


def load_dataset(file_path=None):
    """Load, clean, and extract features and target from the dataset."""
    if file_path is None:
        file_path = find_dataset_file()
    else:
        file_path = Path(file_path)

    if file_path.suffix.lower() == ".csv":
        df = pd.read_csv(file_path, encoding="utf-8-sig")
    else:
        df = pd.read_excel(file_path)

    df = clean_column_names(df)

    # Find target column
    target_col = None
    for cand in TARGET_CANDIDATES:
        cand_clean = cand.strip().replace("\n", "").replace("\r", "")
        for col in df.columns:
            if col.strip().casefold() == cand_clean.casefold():
                target_col = col
                break
        if target_col:
            break

    if target_col is None:
        raise ValueError(f"Could not identify stress level target column in {df.columns.tolist()}")

    # Normalize target
    y_raw = df[target_col]
    y = normalize_target_values(y_raw)

    valid_mask = y.notna()
    df_valid = df.loc[valid_mask].copy()
    y_valid = y.loc[valid_mask].astype(int)

    X_valid = transform_features(df_valid)

    return X_valid, y_valid, df_valid, str(file_path.name)
