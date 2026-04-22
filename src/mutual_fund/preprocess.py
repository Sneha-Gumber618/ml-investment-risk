# Mutual Fund Data Preprocessing
# This module will handle data cleaning and preprocessing for mutual fund data

import pandas as pd
import numpy as np

def preprocess_mf_data(file_path):
    """
    Preprocess mutual fund data from Parquet file (FIXED)
    
    Args:
        file_path (str): Path to the mutual fund data Parquet file
        
    Returns:
        pd.DataFrame: Preprocessed mutual fund data
    """
    # Load data from parquet file
    df = pd.read_parquet(file_path)
    
    # Check if we have Scheme_Code in index or columns
    if 'Scheme_Code' not in df.columns:
        # If Scheme_Code is in index, reset it
        df = df.reset_index()
    
    # Convert Date column to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Sort by Date only (remove Scheme_Code dependency)
    df = df.sort_values('Date')
    
    # Handle missing values - simple forward fill
    df['NAV'] = df['NAV'].ffill()
    
    # Remove any remaining NaN rows
    df = df.dropna()
    
    print(f"Mutual fund data loaded: {len(df)} rows")
    print(f"Columns: {df.columns.tolist()}")
    
    return df

def create_features(df):
    """
    Create features from mutual fund data (FIXED - no Scheme_Code dependency)
    
    Args:
        df (pd.DataFrame): Raw mutual fund data
        
    Returns:
        pd.DataFrame: Data with engineered features
    """
    # Sort by date
    df = df.sort_values('Date')
    
    # Calculate daily returns (as decimal, not percentage)
    df['returns'] = df['NAV'].pct_change()
    
    # Clean data is already realistic, no need for heavy filtering
    # Just remove any remaining extreme outliers
    df = df[(df['returns'] > -0.1) & (df['returns'] < 0.1)]
    print(f"Data after cleaning: {len(df)} rows")
    
    # Calculate NAV change percentage (as decimal)
    df['nav_change'] = df['returns']
    
    # Calculate rolling volatility (20-day window) - should be small values
    df['volatility'] = df['returns'].rolling(window=20).std()
    
    # Calculate rolling mean NAV
    df['nav_ma'] = df['NAV'].rolling(window=20).mean()
    df['nav_ratio'] = df['NAV'] / df['nav_ma']
    
    # Calculate momentum (5-day return)
    df['momentum_5d'] = df['NAV'].pct_change(5)
    
    # Calculate NAV range
    df['nav_range'] = df['NAV'].rolling(window=20).max() - df['NAV'].rolling(window=20).min()
    df['nav_range_ratio'] = df['nav_range'] / df['nav_ma']
    
    # Remove NaN values created by rolling calculations
    df = df.dropna()
    
    print(f"Features created: {len(df)} rows")
    print(f"Feature columns: {['returns', 'volatility', 'nav_change', 'nav_ratio', 'momentum_5d', 'nav_range_ratio']}")
    
    return df
