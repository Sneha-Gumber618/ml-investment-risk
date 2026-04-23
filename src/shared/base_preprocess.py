"""
Shared feature engineering — used by both stock and mutual_fund preprocess modules.
Both had identical rolling-window calculations. Now in one place.
"""
import pandas as pd
import numpy as np


def add_rolling_features(df, price_col, window=20):
    """
    Add standard rolling-window features used by both stock and mutual fund data.

    Args:
        df (pd.DataFrame): Input dataframe — must have a 'Date' column and price_col
        price_col (str): Column name for the price ('Close' for stock, 'NAV' for mutual fund)
        window (int): Rolling window size in days (default 20)

    Returns:
        pd.DataFrame: Dataframe with engineered feature columns added
    """
    df = df.sort_values('Date')

    # Daily returns (as decimal, not percentage)
    df['returns'] = df[price_col].pct_change()

    # Rolling volatility — 20-day std of returns. Will be small values.
    df['volatility'] = df['returns'].rolling(window).std()

    # Rolling mean and ratio
    df['nav_ma'] = df[price_col].rolling(window).mean()
    df['nav_ratio'] = df[price_col] / df['nav_ma']

    # Momentum: 5-day return
    df['momentum_5d'] = df[price_col].pct_change(5)

    # Price range features
    if 'High' in df.columns and 'Low' in df.columns:
        # Stock data: use actual High/Low columns
        df['nav_range'] = (
            df['High'].rolling(window).max() - df['Low'].rolling(window).min()
        )
    else:
        # Mutual fund data: approximate range from NAV
        df['nav_range'] = (
            df[price_col].rolling(window).max() - df[price_col].rolling(window).min()
        )

    df['nav_range_ratio'] = df['nav_range'] / df['nav_ma']

    # Drop NaN rows created by rolling calculations
    df = df.dropna()

    return df