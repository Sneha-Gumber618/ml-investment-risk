"""
Shared prediction utilities — used by both stock and mutual_fund modules.
Removes all duplicated load_model / generate_explanation / calculate_expected_return code.
"""
import pickle


def load_model(model_path):
    """
    Load a trained model from a .pkl file.

    Args:
        model_path (str): Path to the saved model file

    Returns:
        model: Loaded sklearn model
    """
    with open(model_path, 'rb') as f:
        return pickle.load(f)


def calculate_expected_return(risk_level, amount, time_period, rates=None):
    """
    Calculate compound expected return based on risk level.

    Args:
        risk_level (str): 'Low', 'Medium', or 'High'
        amount (float): Investment amount
        time_period (int): Time period in years
        rates (dict): Optional rate override.
                      Defaults to stock rates if not provided.
                      Example: {'Low': 0.06, 'Medium': 0.12, 'High': 0.20}

    Returns:
        float: Expected return amount
    """
    default_rates = {'Low': 0.08, 'Medium': 0.15, 'High': 0.25}
    r = (rates or default_rates).get(risk_level, 0.08)
    return amount * ((1 + r) ** time_period - 1)


def generate_explanation(risk_level, features):
    """
    Generate a simple human-readable explanation for the predicted risk level.

    Args:
        risk_level (str): 'Low', 'Medium', or 'High'
        features (dict): Feature values (expects 'volatility' key)

    Returns:
        str: Explanation string
    """
    volatility = features.get('volatility', 0)

    if risk_level == 'Low':
        return "Low risk due to stable price movement and moderate volatility"
    elif risk_level == 'Medium':
        return "Medium risk due to moderate price fluctuations and volatility"
    else:
        return "High risk due to high volatility and significant price swings"