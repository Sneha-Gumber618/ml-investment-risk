"""
Mutual Fund Model Training — trains a Random Forest classifier for MF risk prediction.
"""
import pandas as pd
import numpy as np
import pickle
import sys
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.mutual_fund.preprocess import preprocess_mf_data, create_features


def assign_mf_risk(vol):
    """
    Assign risk level based on volatility (RELAXED thresholds for mutual fund data).

    Args:
        vol (float): Volatility value

    Returns:
        str: 'Low', 'Medium', or 'High'
    """
    if vol < 0.02:
        return 'Low'
    elif vol < 0.05:
        return 'Medium'
    else:
        return 'High'


def create_risk_labels(df):
    """
    Create risk labels from mutual fund features (CORRECTED — no data leakage).

    Args:
        df (pd.DataFrame): Mutual fund data with features

    Returns:
        pd.Series: Risk labels ('Low', 'Medium', 'High')
    """
    # Use volatility as labelling signal only — clean no-leakage approach
    volatility = df['volatility'].copy()

    # Handle any NaN or Inf values
    volatility = volatility.replace([np.inf, -np.inf], np.nan).fillna(0)

    risk_labels = volatility.apply(assign_mf_risk)

    # Check distribution
    print("Label Distribution:")
    print(risk_labels.value_counts())
    label_counts = risk_labels.value_counts()

    if len(label_counts) < 3:
        print("WARNING: Not all risk levels present!")

    return risk_labels


def save_model(model, model_path):
    """
    Save a trained model to a .pkl file.

    Args:
        model: Trained sklearn model
        model_path (str): Path to save the model
    """
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {model_path}")


def train_mf_model(data_path):
    """
    Full mutual fund model training pipeline.

    Args:
        data_path (str): Path to the mutual fund Parquet file

    Returns:
        tuple: (model, accuracy, feature_importance dict)
    """
    # Load and preprocess data
    df = preprocess_mf_data(data_path)
    df = create_features(df)

    # Create labels
    risk_labels = create_risk_labels(df)

    # Select features (no volatility — used for labelling, would cause leakage)
    feature_columns = ['returns', 'nav_change', 'nav_ratio', 'momentum_5d', 'nav_range_ratio']
    features = df[feature_columns]

    # Align labels and features
    y = risk_labels[features.index]
    X = features.copy()

    # Remove any remaining NaN/Inf rows
    X = X.replace([np.inf, -np.inf], np.nan)
    mask = X.notna().all(axis=1)
    X = X[mask]
    y = y[mask]

    print(f"Training data after cleaning: {len(X)} samples")

    # Need at least 2 classes to train
    label_counts = y.value_counts()
    if len(label_counts) < 2:
        raise ValueError(f"ERROR: Need at least 2 risk levels. Got: {label_counts}")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train Random Forest
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        min_samples_leaf=5,
        random_state=42,
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Mutual Fund Model Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred))

    # Feature importance
    feature_importance = dict(zip(feature_columns, model.feature_importances_))
    print("Feature Importances:")
    for feat, imp in sorted(feature_importance.items(), key=lambda x: -x[1]):
        print(f"  {feat}: {imp:.4f}")

    # Save model
    save_model(model, 'models/mf_model.pkl')

    return model, accuracy, feature_importance