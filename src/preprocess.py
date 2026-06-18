# src/preprocess.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import src.config as config


def load_and_clean_data():
    """Load the dataset and fix dates and missing values."""
    df = pd.read_csv(config.DATA_PATH)
    df['date'] = pd.to_datetime(df['date'], dayfirst=True)
    df = df.ffill().bfill()
    return df


def add_new_features(df):
    """Create new columns for our model to learn from."""
    df = df.copy()

    # 1. Get the hour and day name
    df['Hour'] = df['date'].dt.hour
    df['DayOfWeek'] = df['date'].dt.day_name()

    # 2. Create cyclical features (using 24 hours to represent a full day cycle)
    df['hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24.0)

    # 3. Create a weekend indicator
    df['is_weekend'] = df['DayOfWeek'].isin(['Saturday', 'Sunday']).astype(int)

    # 4. Convert text categories to numbers
    encoder = LabelEncoder()
    for col in config.CATEGORICAL_COLUMNS:
        df[f"{col}_enc"] = encoder.fit_transform(df[col])

    return df
