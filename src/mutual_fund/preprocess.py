"""
Mutual fund data preprocessing — cleans Parquet data and builds features using shared utilities.
"""
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.shared.base_preprocess import add_rolling_features


def preprocess_mf_data(file_path):
    """
    Preprocess mutual fund NAV history from a Parquet file.

    Args:
        file_path (str): Path to the mutual fund Parquet file

    Returns:
        pd.DataFrame: Cleaned mutual fund dataframe
    """
    df = pd.read_parquet(file_path)

    # Reset index if 'Date' is in the index
    if 'Date' not in df.columns:
        df = df.reset_index()

    # Parse and sort by date
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')

    # Forward-fill missing NAV values
    df['NAV'] = df['NAV'].ffill()

    # Drop any remaining NaN rows
    df = df.dropna()

    print(f"Mutual fund data loaded: {len(df)} rows")
    print(f"Columns: {df.columns.tolist()}")

    return df


def create_features(df):
    """
    Create features from cleaned mutual fund data using shared rolling feature builder.

    Args:
        df (pd.DataFrame): Cleaned mutual fund dataframe

    Returns:
        pd.DataFrame: Dataframe with engineered features
    """
    df = add_rolling_features(df, price_col='NAV', window=20)

    # MF-specific: NAV daily change (as decimal)
    df['nav_change'] = df['NAV'].pct_change()

    # Remove outliers — clean data is already handled; just drop any extreme outliers
    df = df[df['returns'].abs() < 0.1]

    df = df.dropna()

    print(f"Features created: {len(df)} rows")
    print(f"Feature columns: {['returns', 'volatility', 'nav_change', 'nav_ratio', 'momentum_5d', 'nav_range_ratio']}")

    return df