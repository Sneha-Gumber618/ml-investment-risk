"""
Stock Model Training — trains a Random Forest classifier for stock risk prediction.
"""
import pandas as pd
import numpy as np
import pickle
import sys
import os
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))
from src.stock.preprocess import preprocess_stock_data, create_features


def assign_stock_risk(vol):
    """
    Assign risk level based on volatility (FIXED thresholds for better balance).

    Args:
        vol (float): Volatility value

    Returns:
        str: 'Low', 'Medium', or 'High'
    """
    if vol < 0.012:
        return 'Low'
    elif vol < 0.020:
        return 'Medium'
    else:
        return 'High'


def create_stock_labels(df):
    """
    Create risk labels from stock features (CORRECTED — no data leakage).

    Args:
        df (pd.DataFrame): Stock data with features

    Returns:
        pd.Series: Risk labels ('Low', 'Medium', 'High')
    """
    volatility = df['volatility'].copy()
    volatility = volatility.replace([np.inf, -np.inf], np.nan).fillna(0)
    risk_labels = volatility.apply(assign_stock_risk)

    print("Label Distribution:")
    print(risk_labels.value_counts())
    label_counts = risk_labels.value_counts()

    if len(label_counts) < 3:
        print("WARNING: Not all risk levels present!")

    return risk_labels


def save_model(model, model_path):
    """
    Save a trained model to a .pkl file.
    Creates parent directory if it doesn't exist.

    Args:
        model: Trained sklearn model
        model_path (str): Path to save the model
    """
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {model_path}")


def train_stock_model(data_path):
    """
    Full stock model training pipeline.

    Args:
        data_path (str): Path to the stock CSV file

    Returns:
        tuple: (model, accuracy, feature_importance dict)
    """
    # Load and preprocess data
    df = preprocess_stock_data(data_path)
    df = create_features(df)

    # Create labels
    risk_labels = create_stock_labels(df)

    # Select features (no volatility — it's used for labelling, would cause leakage)
    feature_columns = ['returns', 'price_change', 'volume_ratio', 'momentum_5d', 'spread']
    features = df[feature_columns]

    # Align labels
    y = risk_labels[features.index]
    X = features.copy()

    # Remove any remaining NaN/Inf rows
    X = X.replace([np.inf, -np.inf], np.nan)
    mask = X.notna().all(axis=1)
    X = X[mask]
    y = y[mask]

    print(f"Training data after cleaning: {len(X)} samples")

    # Check we have at least 2 classes
    label_counts = y.value_counts()
    if len(label_counts) < 2:
        raise ValueError(f"ERROR: Need at least 2 risk levels. Got: {label_counts}")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train Random Forest — reduced complexity to avoid overfitting
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=42,
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Stock Model Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred))

    # Feature importance
    feature_importance = dict(zip(feature_columns, model.feature_importances_))
    print("Feature Importances:")
    for feat, imp in sorted(feature_importance.items(), key=lambda x: -x[1]):
        print(f"  {feat}: {imp:.4f}")

    # Save model using absolute path relative to project root
    model_path = str(PROJECT_ROOT / 'models' / 'stock_model.pkl')
    save_model(model, model_path)

    return model, accuracy, feature_importance
