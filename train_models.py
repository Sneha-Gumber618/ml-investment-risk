#!/usr/bin/env python3
"""
Simple script to train the investment analysis models
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__) / 'src'))

from src.stock.train import train_stock_model
from src.mutual_fund.train import train_mf_model

def main():
    print("Investment Analysis Model Training")
    print("=" * 50)
    
    # Train stock model
    print("\n1. Training Stock Model...")
    try:
        stock_model, stock_accuracy, stock_feature_importance = train_stock_model('data/RELIANCE.csv')
        print(f"Stock model trained successfully! Accuracy: {stock_accuracy:.4f}")
    except Exception as e:
        print(f"Error training stock model: {e}")
    
    # Train mutual fund model
    print("\n2. Training Mutual Fund Model...")
    try:
        mf_model, mf_accuracy, mf_feature_importance = train_mf_model('data/mutual_fund_optimal.parquet')
        print(f"Mutual fund model trained successfully! Accuracy: {mf_accuracy:.4f}")
    except Exception as e:
        print(f"Error training mutual fund model: {e}")
    
    print("\nTraining completed!")

if __name__ == "__main__":
    main()
