#!/usr/bin/env python3
"""
Entry-point script to train both investment analysis models.
Run this once from the project root before launching the app:
    python train_models.py
"""

import sys
import os
from pathlib import Path

# Ensure project root is on the path so 'src.*' imports resolve
PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

# Create models/ directory if it doesn't exist
models_dir = PROJECT_ROOT / 'models'
models_dir.mkdir(exist_ok=True)

from src.stock.train import train_stock_model
from src.mutual_fund.train import train_mf_model


def main():
    print("Investment Analysis Model Training")
    print("=" * 50)

    data_dir = PROJECT_ROOT / 'data'

    # Train stock model
    print("\n1. Training Stock Model...")
    try:
        stock_model, stock_accuracy, stock_feature_importance = train_stock_model(
            str(data_dir / 'RELIANCE.csv')
        )
        print(f"[OK] Stock model trained successfully! Accuracy: {stock_accuracy:.4f}")
    except Exception as e:
        print(f"[ERROR] Error training stock model: {e}")

    # Train mutual fund model
    print("\n2. Training Mutual Fund Model...")
    try:
        mf_model, mf_accuracy, mf_feature_importance = train_mf_model(
            str(data_dir / 'mutual_fund_optimal.parquet')
        )
        print(f"[OK] Mutual fund model trained successfully! Accuracy: {mf_accuracy:.4f}")
    except Exception as e:
        print(f"[ERROR] Error training mutual fund model: {e}")

    print("\nTraining completed!")


if __name__ == "__main__":
    main()
