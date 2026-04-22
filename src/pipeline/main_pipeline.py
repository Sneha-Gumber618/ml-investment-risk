# Main Pipeline
# This module orchestrates the entire investment analysis process

import sys
import os
from pathlib import Path
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from stock.train import train_stock_model
from stock.predict import load_model as load_stock_model, predict_stock_risk, calculate_expected_return as calculate_stock_return
from mutual_fund.train import train_mf_model
from mutual_fund.predict import load_model as load_mf_model, predict_mf_risk, calculate_expected_return as calculate_mf_return
from fd.fd_calculator import analyze_fd_investment
from stock.preprocess import preprocess_stock_data, create_features as create_stock_features
from mutual_fund.preprocess import preprocess_mf_data, create_features as create_mf_features

def train_all_models():
    """
    Train all models (stock and mutual fund)
    
    Returns:
        dict: Training results for all models
    """
    results = {}
    
    print("Training Stock Model...")
    try:
        stock_model, stock_accuracy, stock_feature_importance = train_stock_model('../../data/RELIANCE.csv')
        results['stock'] = {
            'model': stock_model,
            'accuracy': stock_accuracy,
            'feature_importance': stock_feature_importance
        }
        print("Stock model trained successfully!")
    except Exception as e:
        print(f"Error training stock model: {e}")
        results['stock'] = None
    
    print("\nTraining Mutual Fund Model...")
    try:
        mf_model, mf_accuracy, mf_feature_importance = train_mf_model('../../data/mutual_fund_nav_history.parquet')
        results['mutual_fund'] = {
            'model': mf_model,
            'accuracy': mf_accuracy,
            'feature_importance': mf_feature_importance
        }
        print("Mutual fund model trained successfully!")
    except Exception as e:
        print(f"Error training mutual fund model: {e}")
        results['mutual_fund'] = None
    
    return results

def analyze_investment(investment_type, amount, time_period):
    """
    Analyze investment based on type and parameters
    
    Args:
        investment_type (str): 'stock', 'mutual_fund', or 'fd'
        amount (float): Investment amount
        time_period (int): Time period in years
        
    Returns:
        dict: Analysis results with risk, return, and explanation
    """
    if investment_type == 'stock':
        return analyze_stock_investment(amount, time_period)
    elif investment_type == 'mutual_fund':
        return analyze_mutual_fund_investment(amount, time_period)
    elif investment_type == 'fd':
        return analyze_fd_investment(amount, time_period)
    else:
        return {'error': 'Invalid investment type'}

def analyze_stock_investment(amount, time_period):
    """
    Analyze stock investment
    
    Args:
        amount (float): Investment amount
        time_period (int): Time period in years
        
    Returns:
        dict: Analysis results
    """
    try:
        # Load trained model
        model = load_stock_model('models/stock_model.pkl')
        
        # Get recent stock data for features
        df = preprocess_stock_data('data/RELIANCE.csv')
        df = create_stock_features(df)
        
        # Use most recent data point for prediction
        recent_data = df.iloc[-1]
        
        # Prepare features (REMOVED volatility to prevent data leakage)
        features = {
            'returns': recent_data['returns'],
            'price_change': recent_data['price_change'],
            'volume_ratio': recent_data['volume_ratio'],
            'momentum_5d': recent_data['momentum_5d'],
            'spread': recent_data['spread']
        }
        
        # Make prediction
        prediction = predict_stock_risk(model, features)
        
        # Calculate expected return
        expected_return = calculate_stock_return(prediction['risk_level'], amount, time_period)
        
        return {
            'risk_level': prediction['risk_level'],
            'expected_return': expected_return,
            'explanation': prediction['explanation'],
            'confidence': prediction['confidence']
        }
        
    except Exception as e:
        return {'error': f'Error analyzing stock investment: {e}'}

def analyze_mutual_fund_investment(amount, time_period):
    """
    Analyze mutual fund investment
    
    Args:
        amount (float): Investment amount
        time_period (int): Time period in years
        
    Returns:
        dict: Analysis results
    """
    try:
        # Load trained model
        model = load_mf_model('models/mf_model.pkl')
        
        # Get recent mutual fund data for features
        df = preprocess_mf_data('data/mutual_fund_optimal.parquet')
        df = create_mf_features(df)
        
        # Use most recent data point for prediction
        recent_data = df.iloc[-1]
        
        # Prepare features (REMOVED volatility to prevent data leakage)
        features = {
            'returns': recent_data['returns'],
            'nav_change': recent_data['nav_change'],
            'nav_ratio': recent_data['nav_ratio'],
            'momentum_5d': recent_data['momentum_5d'],
            'nav_range_ratio': recent_data['nav_range_ratio']
        }
        
        # Make prediction
        prediction = predict_mf_risk(model, features)
        
        # Calculate expected return
        expected_return = calculate_mf_return(prediction['risk_level'], amount, time_period)
        
        return {
            'risk_level': prediction['risk_level'],
            'expected_return': expected_return,
            'explanation': prediction['explanation'],
            'confidence': prediction['confidence']
        }
        
    except Exception as e:
        return {'error': f'Error analyzing mutual fund investment: {e}'}

def main():
    """
    Main function to run the pipeline
    """
    print("Investment Analysis Pipeline")
    print("=" * 40)
    
    # Train models
    print("\n1. Training models...")
    results = train_all_models()
    
    # Test analysis
    print("\n2. Testing analysis...")
    test_result = analyze_investment('stock', 10000, 5)
    print(f"Stock analysis result: {test_result}")
    
    print("\nPipeline completed successfully!")

if __name__ == "__main__":
    main()
