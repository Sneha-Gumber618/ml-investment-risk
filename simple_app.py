#!/usr/bin/env python3
"""
Simple web interface for investment analysis (no Streamlit)
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__) / 'src'))

def analyze_investment():
    """Simple command-line investment analysis"""
    
    print("=" * 60)
    print("INVESTMENT ANALYSIS APP")
    print("=" * 60)
    
    while True:
        print("\nSelect Investment Type:")
        print("1. Stock Analysis (ML)")
        print("2. Mutual Fund Analysis (ML)")
        print("3. Fixed Deposit Calculator")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '4':
            print("Goodbye!")
            break
            
        if choice not in ['1', '2', '3']:
            print("Invalid choice! Please try again.")
            continue
        
        try:
            # Get investment amount
            amount = float(input("Enter investment amount (Rs): "))
            time_years = float(input("Enter time period (years): "))
            
            if choice == '1':
                analyze_stock(amount, time_years)
            elif choice == '2':
                analyze_mutual_fund(amount, time_years)
            elif choice == '3':
                analyze_fixed_deposit(amount, time_years)
                
        except ValueError:
            print("Invalid input! Please enter numbers only.")
        except Exception as e:
            print(f"Error: {e}")

def analyze_stock(amount, time_years):
    """Analyze stock investment"""
    try:
        from src.pipeline.main_pipeline import analyze_stock_investment
        result = analyze_stock_investment(amount, time_years)
        
        print("\n" + "=" * 40)
        print("STOCK ANALYSIS RESULTS")
        print("=" * 40)
        print(f"Investment Amount: Rs {amount:,.2f}")
        print(f"Time Period: {time_years} years")
        print(f"Risk Level: {result['risk_level']}")
        if 'expected_return' in result:
            print(f"Expected Return: {result['expected_return']:.2f}%")
        print(f"Explanation: {result['explanation']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print("=" * 40)
        
    except Exception as e:
        print(f"Error in stock analysis: {e}")

def analyze_mutual_fund(amount, time_years):
    """Analyze mutual fund investment"""
    try:
        from src.pipeline.main_pipeline import analyze_mutual_fund_investment
        result = analyze_mutual_fund_investment(amount, time_years)
        
        print("\n" + "=" * 40)
        print("MUTUAL FUND ANALYSIS RESULTS")
        print("=" * 40)
        print(f"Investment Amount: Rs {amount:,.2f}")
        print(f"Time Period: {time_years} years")
        print(f"Risk Level: {result['risk_level']}")
        if 'expected_return' in result:
            print(f"Expected Return: {result['expected_return']:.2f}%")
        print(f"Explanation: {result['explanation']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print("=" * 40)
        
    except Exception as e:
        print(f"Error in mutual fund analysis: {e}")

def analyze_fixed_deposit(amount, time_years):
    """Analyze fixed deposit investment"""
    try:
        from src.pipeline.main_pipeline import analyze_fixed_deposit_investment
        result = analyze_fixed_deposit_investment(amount, time_years)
        
        print("\n" + "=" * 40)
        print("FIXED DEPOSIT RESULTS")
        print("=" * 40)
        print(f"Investment Amount: Rs {amount:,.2f}")
        print(f"Time Period: {time_years} years")
        print(f"Risk Level: {result['risk_level']}")
        print(f"Expected Return: {result['expected_return']:.2f}%")
        print(f"Explanation: {result['explanation']}")
        print(f"Final Amount: Rs {result['final_amount']:,.2f}")
        print("=" * 40)
        
    except Exception as e:
        print(f"Error in fixed deposit analysis: {e}")

if __name__ == "__main__":
    print("Starting Investment Analysis App...")
    print("Models should be trained before running this app.")
    print("If models are not trained, run: python train_models.py")
    print()
    
    analyze_investment()
