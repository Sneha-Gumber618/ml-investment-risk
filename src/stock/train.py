# Stock Model Training
# This module will train the Random Forest model for stock analysis

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.stock.preprocess import preprocess_stock_data, create_features

def train_stock_model(data_path):
    """
    Train Random Forest model for stock risk prediction
    
    Args:
        data_path (str): Path to processed stock data
        
    Returns:
        tuple: (model, accuracy, feature_importance)
    """
    # Load and preprocess data
    df = preprocess_stock_data(data_path)
    df = create_features(df)
    
    # Create risk labels
    df['risk_level'] = create_risk_labels(df)
    
    # Prepare features for training (REMOVED volatility to prevent data leakage)
    feature_columns = ['returns', 'price_change', 'volume_ratio', 'momentum_5d', 'spread']
    X = df[feature_columns]
    y = df['risk_level']
    
    # Remove any remaining NaN values
    X = X.dropna()
    y = y[X.index]
    
    print(f"\nData after cleaning: {len(X)} samples")
    
    # Check feature statistics (IMPORTANT)
    print("\nFeature Statistics:")
    print(X.describe())
    
    # Check if features have variation
    for col in feature_columns:
        if X[col].std() == 0:
            print(f"WARNING: {col} has zero variance!")
    
    # Check risk distribution
    print("\nFinal Risk Distribution:")
    print(y.value_counts())
    
    # Stop if we don't have balanced classes
    if len(y.value_counts()) < 2:
        print("ERROR: Need at least 2 different risk levels!")
        return None, None, None
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train Random Forest (reduced complexity to avoid overfitting)
    model = RandomForestClassifier(
        n_estimators=50,    # Reduced from 100
        max_depth=8,         # Reduced from 10
        random_state=42,
        min_samples_split=10,  # Add regularization
        min_samples_leaf=5     # Add regularization
    )
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Stock Model Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Feature importance
    feature_importance = dict(zip(feature_columns, model.feature_importances_))
    print("\nFeature Importance:")
    for feature, importance in feature_importance.items():
        print(f"{feature}: {importance:.4f}")
    
    # Save model
    save_model(model, 'models/stock_model.pkl')
    
    return model, accuracy, feature_importance

def assign_risk(vol):
    """
    Assign risk level based on volatility (FIXED for small values)
    
    Args:
        vol (float): Volatility value (should be 0.01-0.05 range)
        
    Returns:
        str: Risk level (Low, Medium, High)
    """
    if vol < 0.015:  # Low volatility
        return "Low"
    elif vol < 0.025:  # Medium volatility
        return "Medium"
    else:  # High volatility
        return "High"

def create_risk_labels(df):
    """
    Create risk labels based on volatility (CORRECTED)
    
    Args:
        df (pd.DataFrame): Stock data with features
        
    Returns:
        pd.Series: Risk labels (Low, Medium, High)
    """
    # Clean volatility data
    volatility = df['volatility'].fillna(0)
    
    # Remove any invalid volatility values
    volatility = volatility.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Assign risk based on volatility
    risk_labels = volatility.apply(assign_risk)
    
    # Check distribution
    print("Risk Label Distribution:")
    print(risk_labels.value_counts())
    print(f"Total samples: {len(risk_labels)}")
    
    # Check if we have balanced classes
    label_counts = risk_labels.value_counts()
    if len(label_counts) < 3:
        print("WARNING: Not all risk levels present!")
    
    return risk_labels

def save_model(model, model_path):
    """
    Save trained model to file
    
    Args:
        model: Trained model
        model_path (str): Path to save model
    """
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {model_path}")
