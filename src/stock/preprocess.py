# Stock Data Preprocessing
# This module will handle data cleaning and preprocessing for stock data

import pandas as pd
import numpy as np

def preprocess_stock_data(file_path):
    """
    Preprocess stock data from CSV file
    
    Args:
        file_path (str): Path to the stock data CSV file
        
    Returns:
        pd.DataFrame: Preprocessed stock data
    """
    # Load data
    df = pd.read_csv(file_path)
    
    # Convert Date column to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Sort by date
    df = df.sort_values('Date')
    
    # Handle missing values - forward fill
    numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    df[numeric_columns] = df[numeric_columns].ffill()
    
    # Remove any remaining NaN rows
    df = df.dropna()
    
    return df

def create_features(df):
    """
    Create features from stock data
    
    Args:
        df (pd.DataFrame): Raw stock data
        
    Returns:
        pd.DataFrame: Data with engineered features
    """
    # Calculate daily returns (as decimal, not percentage)
    df['returns'] = df['Close'].pct_change()
    
    # Calculate price change percentage (as decimal)
    df['price_change'] = (df['Close'] - df['Open']) / df['Open']
    
    # Calculate rolling volatility (20-day window) - should be small values
    df['volatility'] = df['returns'].rolling(window=20).std()
    
    # Calculate volume features
    df['volume_ma'] = df['Volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['Volume'] / df['volume_ma']
    
    # Calculate price momentum (5-day return)
    df['momentum_5d'] = df['Close'].pct_change(5)
    
    # Calculate high-low spread (as decimal)
    df['spread'] = (df['High'] - df['Low']) / df['Low']
    
    # Remove NaN values created by rolling calculations
    df = df.dropna()
    
    return df
