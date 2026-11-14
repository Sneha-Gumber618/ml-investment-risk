"""
Stock prediction — only stock-specific logic here.
All shared utilities (load_model, generate_explanation, calculate_expected_return)
come from src/shared/base_predict.py.
"""
import numpy as np
import pandas as pd
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))
from src.shared.base_predict import load_model, calculate_expected_return, generate_explanation

# Stock-specific annual return rates by risk level
STOCK_RATES = {'Low': 0.08, 'Medium': 0.15, 'High': 0.25}

# Features the stock model was trained on (no volatility — prevents leakage)
FEATURE_COLUMNS = ['returns', 'price_change', 'volume_ratio', 'momentum_5d', 'spread']


def predict_stock_risk(model, features):
    """
    Predict risk level for a stock investment.

    Args:
        model: Trained sklearn model (loaded from models/stock_model.pkl)
        features (dict): Input feature values

    Returns:
        dict: risk_level, confidence, explanation
    """
    feature_array = pd.DataFrame([[features[col] for col in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)

    risk_level = model.predict(feature_array)[0]
    probabilities = model.predict_proba(feature_array)[0]
    confidence = max(probabilities)
    explanation = generate_explanation(risk_level, features)

    return {
        'risk_level': risk_level,
        'confidence': confidence,
        'explanation': explanation,
    }


def calculate_expected_return(risk_level, amount, time_period):
    """
    Calculate expected return for a stock investment using stock-specific rates.

    Args:
        risk_level (str): 'Low', 'Medium', or 'High'
        amount (float): Investment amount
        time_period (int): Time period in years

    Returns:
        float: Expected return amount
    """
    from src.shared.base_predict import calculate_expected_return as _calc
    return _calc(risk_level, amount, time_period, rates=STOCK_RATES)
