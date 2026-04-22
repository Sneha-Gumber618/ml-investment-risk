# Mutual Fund Model Prediction
# This module will make predictions using trained mutual fund model

import pandas as pd
import numpy as np
import pickle
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.mutual_fund.preprocess import create_features

def load_model(model_path):
    """
    Load trained model from file
    
    Args:
        model_path (str): Path to saved model
        
    Returns:
        model: Loaded model
    """
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def predict_mf_risk(model, features):
    """
    Predict risk level for mutual fund
    
    Args:
        model: Trained model
        features (dict): Input features
        
    Returns:
        dict: Prediction results with risk level and explanation
    """
    # Prepare feature array for prediction (REMOVED volatility)
    feature_columns = ['returns', 'nav_change', 'nav_ratio', 'momentum_5d', 'nav_range_ratio']
    feature_array = np.array([[features[col] for col in feature_columns]])
    
    # Make prediction
    risk_level = model.predict(feature_array)[0]
    
    # Get prediction probabilities
    probabilities = model.predict_proba(feature_array)[0]
    confidence = max(probabilities)
    
    # Generate explanation
    explanation = generate_explanation(risk_level, features)
    
    return {
        'risk_level': risk_level,
        'confidence': confidence,
        'explanation': explanation
    }

def calculate_expected_return(risk_level, amount, time_period):
    """
    Calculate expected return based on risk level
    
    Args:
        risk_level (str): Risk level (Low, Medium, High)
        amount (float): Investment amount
        time_period (int): Time period in years
        
    Returns:
        float: Expected return amount
    """
    # Define annual return rates based on risk level (slightly lower than stocks)
    if risk_level == 'Low':
        annual_return_rate = 0.06  # 6% annual return
    elif risk_level == 'Medium':
        annual_return_rate = 0.12  # 12% annual return
    else:  # High
        annual_return_rate = 0.20  # 20% annual return
    
    # Calculate compound interest
    expected_return = amount * ((1 + annual_return_rate) ** time_period - 1)
    
    return expected_return

def generate_explanation(risk_level, features):
    """
    Generate simple explanation for risk level
    
    Args:
        risk_level (str): Risk level
        features (dict): Key features used
        
    Returns:
        str: Simple explanation
    """
    volatility = features.get('volatility', 0)
    returns = features.get('returns', 0)
    
    if risk_level == 'Low':
        return "Low risk due to stable NAV movement and moderate volatility"
    elif risk_level == 'Medium':
        return "Medium risk due to moderate NAV fluctuations and volatility"
    else:  # High
        return "High risk due to high NAV volatility and significant swings"
