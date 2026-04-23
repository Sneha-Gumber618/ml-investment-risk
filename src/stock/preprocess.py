"""
Stock data preprocessing — cleans CSV and builds features using shared utilities.
"""
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.shared.base_preprocess import add_rolling_features


def preprocess_stock_data(file_path):
    """
    Preprocess stock data from a CSV file.

    Args:
        file_path (str): Path to the stock CSV file (e.g. data/RELIANCE.csv)

    Returns:
        pd.DataFrame: Cleaned stock dataframe
    """
    df = pd.read_csv(file_path)

    # Parse and sort by date
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')

    # Forward-fill missing values for OHLCV columns
    numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    df[numeric_columns] = df[numeric_columns].ffill()

    # Drop any remaining NaN rows
    df = df.dropna()

    print(f"Stock data loaded: {len(df)} rows")
    print(f"Columns: {df.columns.tolist()}")

    return df


def create_features(df):
    """
    Create features from cleaned stock data using the shared rolling feature builder.

    Args:
        df (pd.DataFrame): Cleaned stock dataframe

    Returns:
        pd.DataFrame: Dataframe with engineered features
    """
    df = add_rolling_features(df, price_col='Close', window=20)

    # Stock-specific extra features
    df['price_change'] = (df['Close'] - df['Open']) / df['Open']

    if 'Volume' in df.columns:
        volume_ma = df['Volume'].rolling(20).mean()
        df['volume_ratio'] = df['Volume'] / volume_ma

    df['spread'] = (df['High'] - df['Low']) / df['Close']

    df = df.dropna()

    print(f"Features created: {len(df)} rows")
    print(f"Feature columns: {['returns', 'volatility', 'nav_change', 'price_change', 'volume_ratio', 'momentum_5d', 'spread']}")

    return df