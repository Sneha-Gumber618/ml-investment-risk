# Main Pipeline
# Orchestrates the entire investment analysis process

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from src.stock.train import train_stock_model
from src.stock.predict import load_model as load_stock_model, predict_stock_risk
from src.stock.predict import calculate_expected_return as calculate_stock_return
from src.mutual_fund.train import train_mf_model
from src.mutual_fund.predict import load_model as load_mf_model, predict_mf_risk
from src.mutual_fund.predict import calculate_expected_return as calculate_mf_return
from src.fd_calculator import analyze_fd_investment
from src.stock.preprocess import preprocess_stock_data, create_features as create_stock_features
from src.mutual_fund.preprocess import preprocess_mf_data, create_features as create_mf_features

DATA_DIR = PROJECT_ROOT / 'data'
MODELS_DIR = PROJECT_ROOT / 'models'


def train_all_models():
    """
    Train all models (stock and mutual fund).

    Returns:
        dict: Training results for all models
    """
    results = {}
    MODELS_DIR.mkdir(exist_ok=True)

    print("Training Stock Model...")
    try:
        stock_model, stock_accuracy, stock_feature_importance = train_stock_model(
            str(DATA_DIR / 'RELIANCE.csv')
        )
        results['stock'] = {
            'model': stock_model,
            'accuracy': stock_accuracy,
            'feature_importance': stock_feature_importance,
        }
        print("Stock model trained successfully!")
    except Exception as e:
        print(f"Error training stock model: {e}")
        results['stock'] = None

    print("\nTraining Mutual Fund Model...")
    try:
        mf_model, mf_accuracy, mf_feature_importance = train_mf_model(
            str(DATA_DIR / 'mutual_fund_optimal.parquet')
        )
        results['mutual_fund'] = {
            'model': mf_model,
            'accuracy': mf_accuracy,
            'feature_importance': mf_feature_importance,
        }
        print("Mutual fund model trained successfully!")
    except Exception as e:
        print(f"Error training mutual fund model: {e}")
        results['mutual_fund'] = None

    return results


def analyze_investment(investment_type, amount, time_period):
    """
    Analyze investment based on type and parameters.

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
    Analyze stock investment.

    Args:
        amount (float): Investment amount
        time_period (int): Time period in years

    Returns:
        dict: Analysis results
    """
    try:
        model_path = str(MODELS_DIR / 'stock_model.pkl')
        model = load_stock_model(model_path)

        df = preprocess_stock_data(str(DATA_DIR / 'RELIANCE.csv'))
        df = create_stock_features(df)

        recent_data = df.iloc[-1]

        features = {
            'returns': recent_data['returns'],
            'price_change': recent_data['price_change'],
            'volume_ratio': recent_data['volume_ratio'],
            'momentum_5d': recent_data['momentum_5d'],
            'spread': recent_data['spread'],
        }

        prediction = predict_stock_risk(model, features)
        expected_return = calculate_stock_return(prediction['risk_level'], amount, time_period)

        return {
            'risk_level': prediction['risk_level'],
            'expected_return': expected_return,
            'explanation': prediction['explanation'],
            'confidence': prediction['confidence'],
        }

    except FileNotFoundError:
        return {'error': 'Stock model not found. Please train models first using the sidebar button.'}
    except Exception as e:
        return {'error': f'Error analyzing stock investment: {e}'}


def analyze_mutual_fund_investment(amount, time_period):
    """
    Analyze mutual fund investment.

    Args:
        amount (float): Investment amount
        time_period (int): Time period in years

    Returns:
        dict: Analysis results
    """
    try:
        model_path = str(MODELS_DIR / 'mf_model.pkl')
        model = load_mf_model(model_path)

        df = preprocess_mf_data(str(DATA_DIR / 'mutual_fund_optimal.parquet'))
        df = create_mf_features(df)

        recent_data = df.iloc[-1]

        features = {
            'returns': recent_data['returns'],
            'nav_change': recent_data['nav_change'],
            'nav_ratio': recent_data['nav_ratio'],
            'momentum_5d': recent_data['momentum_5d'],
            'nav_range_ratio': recent_data['nav_range_ratio'],
        }

        prediction = predict_mf_risk(model, features)
        expected_return = calculate_mf_return(prediction['risk_level'], amount, time_period)

        return {
            'risk_level': prediction['risk_level'],
            'expected_return': expected_return,
            'explanation': prediction['explanation'],
            'confidence': prediction['confidence'],
        }

    except FileNotFoundError:
        return {'error': 'Mutual fund model not found. Please train models first using the sidebar button.'}
    except Exception as e:
        return {'error': f'Error analyzing mutual fund investment: {e}'}


def main():
    """
    Main function to run the pipeline.
    """
    print("Investment Analysis Pipeline")
    print("=" * 40)

    print("\n1. Training models...")
    results = train_all_models()

    print("\n2. Testing analysis...")
    test_result = analyze_investment('stock', 10000, 5)
    print(f"Stock analysis result: {test_result}")

    print("\nPipeline completed successfully!")


if __name__ == "__main__":
    main()
