"""
Mutual fund prediction — only MF-specific logic here.
All shared utilities (load_model, generate_explanation, calculate_expected_return)
come from src/shared/base_predict.py.
"""
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.shared.base_predict import load_model, calculate_expected_return, generate_explanation

# Mutual fund rates — slightly lower than stocks (less volatile)
MF_RATES = {'Low': 0.06, 'Medium': 0.12, 'High': 0.20}

# Features the mutual fund model was trained on (FIXED — no volatility leakage)
FEATURE_COLUMNS = ['returns', 'nav_change', 'nav_ratio', 'momentum_5d', 'nav_range_ratio']


def predict_mf_risk(model, features):
    """
    Predict risk level for a mutual fund investment.

    Args:
        model: Trained sklearn model (loaded from models/mf_model.pkl)
        features (dict): Input feature values

    Returns:
        dict: risk_level, confidence, explanation
    """
    feature_array = np.array([[features[col] for col in FEATURE_COLUMNS]])

    risk_level = model.predict(feature_array)[0]
    probabilities = model.predict_proba(feature_array)[0]
    confidence = max(probabilities)
    explanation = generate_explanation(risk_level, features)

    return {
        'risk_level': risk_level,
        'confidence': confidence,
        'explanation': explanation,
    }


def calculate_mf_return(risk_level, amount, time_period):
    """
    Calculate expected return for a mutual fund investment using MF-specific rates.

    Args:
        risk_level (str): 'Low', 'Medium', or 'High'
        amount (float): Investment amount
        time_period (int): Time period in years

    Returns:
        float: Expected return amount
    """
    return calculate_expected_return(risk_level, amount, time_period, rates=MF_RATES)