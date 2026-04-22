# Streamlit Investment Analysis App
# Simple web interface for investment analysis

import streamlit as st
import sys
import os
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from src.pipeline.main_pipeline import analyze_investment, train_all_models

def main():
    """
    Main Streamlit app function
    """
    # Page configuration
    st.set_page_config(
        page_title="Investment Analysis App",
        page_icon="📊",
        layout="centered"
    )
    
    # Title and description
    st.title("📈 Investment Analysis App")
    st.write("Analyze your investment options using Machine Learning")
    
    # Sidebar for model training
    with st.sidebar:
        st.header("Model Management")
        if st.button("Train Models", help="Train ML models with your data"):
            with st.spinner("Training models... This may take a few minutes."):
                try:
                    results = train_all_models()
                    if results['stock'] and results['mutual_fund']:
                        st.success("Models trained successfully!")
                        st.write(f"Stock Model Accuracy: {results['stock']['accuracy']:.4f}")
                        st.write(f"Mutual Fund Model Accuracy: {results['mutual_fund']['accuracy']:.4f}")
                    else:
                        st.error("Model training failed. Please check your data files.")
                except Exception as e:
                    st.error(f"Error during training: {e}")
    
    # Main content
    st.header("Investment Analysis")
    
    # Investment type selection
    investment_type = st.selectbox(
        "Select Investment Type:",
        ["Stock Analysis (ML)", "Mutual Fund Analysis (ML)", "Fixed Deposit (FD)"],
        help="Choose the type of investment you want to analyze"
    )
    
    # Create columns for input fields
    col1, col2 = st.columns(2)
    
    with col1:
        amount = st.number_input(
            "Investment Amount ($)", 
            min_value=1000, 
            max_value=1000000, 
            value=10000,
            step=1000,
            help="Enter the amount you want to invest"
        )
    
    with col2:
        time_period = st.number_input(
            "Time Period (years)", 
            min_value=1, 
            max_value=30, 
            value=5,
            step=1,
            help="Enter the investment duration in years"
        )
    
    # Analyze button
    if st.button("🚀 Analyze Investment", type="primary"):
        # Map selection to internal type
        type_mapping = {
            "Stock Analysis (ML)": "stock",
            "Mutual Fund Analysis (ML)": "mutual_fund",
            "Fixed Deposit (FD)": "fd"
        }
        
        internal_type = type_mapping[investment_type]
        
        # Show loading spinner
        with st.spinner("Analyzing investment..."):
            result = analyze_investment(internal_type, amount, time_period)
        
        # Display results
        if 'error' in result:
            st.error(result['error'])
        else:
            # Success message
            st.success("Analysis completed successfully!")
            
            # Results section
            st.header("📊 Analysis Results")
            
            # Create metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Risk Level",
                    result['risk_level'],
                    help="Predicted risk level for this investment"
                )
            
            with col2:
                # Calculate simple expected return based on risk
                if result['risk_level'] == 'Low':
                    expected_return = amount * 1.08  # 8% return
                elif result['risk_level'] == 'Medium':
                    expected_return = amount * 1.12  # 12% return
                else:  # High
                    expected_return = amount * 1.18  # 18% return
                
                st.metric(
                    "Expected Return",
                    f"${expected_return:,.2f}",
                    help="Expected return over the investment period"
                )
            
            with col3:
                if 'confidence' in result:
                    st.metric(
                        "Confidence",
                        f"{result['confidence']:.2%}",
                        help="Model confidence in the prediction"
                    )
                else:
                    st.metric(
                        "Total Amount",
                        f"${result.get('total_amount', amount):,.2f}",
                        help="Total amount after investment period"
                    )
            
            # Explanation
            st.subheader("💡 Explanation")
            st.info(result['explanation'])
            
            # Additional details
            with st.expander("📈 Detailed Analysis"):
                if internal_type == "fd":
                    st.write(f"**Interest Rate:** {result.get('rate', 0):.2f}%")
                    st.write(f"**Interest Earned:** ${result.get('interest_earned', 0):,.2f}")
                else:
                    st.write(f"**Investment Type:** {investment_type}")
                    st.write(f"**Principal Amount:** ${amount:,.2f}")
                    st.write(f"**Time Period:** {time_period} years")
                    if 'confidence' in result:
                        st.write(f"**Model Confidence:** {result['confidence']:.2%}")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center'>"
        "<p>Made with ❤️ using Machine Learning | Random Forest Classifier</p>"
        "<p>Data: Stock (CSV) | Mutual Fund (Parquet) | FD (Formula)</p>"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
