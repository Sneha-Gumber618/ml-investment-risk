# Streamlit Investment Analysis App
# Web interface for AI-powered investment risk analysis

import streamlit as st
import sys
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.main_pipeline import analyze_investment, train_all_models


def risk_color(risk_level):
    """Return a colour emoji for each risk level."""
    return {'Low': '🟢', 'Medium': '🟡', 'High': '🔴'}.get(risk_level, '⚪')


def main():
    """Main Streamlit app function."""

    st.set_page_config(
        page_title="Investment Risk Analyzer",
        page_icon="📊",
        layout="centered",
    )

    # ── Header ──────────────────────────────────────────────────────────────
    st.title("📈 Investment Risk Analyzer")
    st.markdown(
        "Analyze risk and expected returns for **Stocks**, **Mutual Funds**, and **Fixed Deposits** "
        "using Machine Learning (Random Forest Classifier)."
    )
    st.divider()

    # ── Sidebar — model management ──────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Model Management")
        st.info(
            "Models must be trained before you can analyze Stocks or Mutual Funds. "
            "Click **Train Models** to get started."
        )
        if st.button("🚀 Train Models", help="Train ML models with the bundled data"):
            with st.spinner("Training models… this may take a minute."):
                try:
                    results = train_all_models()
                    stock_ok = results.get('stock') is not None
                    mf_ok = results.get('mutual_fund') is not None

                    if stock_ok:
                        st.success(f"✅ Stock Model — Accuracy: {results['stock']['accuracy']:.2%}")
                    else:
                        st.error("❌ Stock model training failed.")

                    if mf_ok:
                        st.success(f"✅ Mutual Fund Model — Accuracy: {results['mutual_fund']['accuracy']:.2%}")
                    else:
                        st.error("❌ Mutual fund model training failed.")

                except Exception as e:
                    st.error(f"Unexpected error during training: {e}")

        st.divider()
        st.markdown("**Data sources**")
        st.markdown("- 📄 Stock: `RELIANCE.csv` (NSE)")
        st.markdown("- 📄 Mutual Fund: `mutual_fund_optimal.parquet`")
        st.markdown("- 🧮 Fixed Deposit: Compound interest formula")

    # ── Main — inputs ───────────────────────────────────────────────────────
    st.subheader("🔍 Analyze an Investment")

    investment_type = st.selectbox(
        "Investment Type",
        ["Stock (ML)", "Mutual Fund (ML)", "Fixed Deposit (Formula)"],
        help="Choose the investment vehicle to analyze",
    )

    col1, col2 = st.columns(2)

    with col1:
        amount = st.number_input(
            "Investment Amount (₹)",
            min_value=1_000,
            max_value=10_000_000,
            value=100_000,
            step=10_000,
            format="%d",
            help="Principal amount in Indian Rupees",
        )

    with col2:
        time_period = st.slider(
            "Time Period (years)",
            min_value=1,
            max_value=30,
            value=5,
            help="How long you plan to stay invested",
        )

    # ── Analyze ─────────────────────────────────────────────────────────────
    if st.button("📊 Analyze Investment", type="primary", use_container_width=True):
        type_map = {
            "Stock (ML)": "stock",
            "Mutual Fund (ML)": "mutual_fund",
            "Fixed Deposit (Formula)": "fd",
        }
        internal_type = type_map[investment_type]

        with st.spinner("Analyzing…"):
            result = analyze_investment(internal_type, amount, time_period)

        st.divider()

        if 'error' in result:
            st.error(f"⚠️ {result['error']}")
            if 'train' in result['error'].lower():
                st.info("👈 Use the **Train Models** button in the sidebar to train the models first.")
        else:
            st.success("Analysis complete!")

            # ── Metrics row ─────────────────────────────────────────────────
            risk_level = result['risk_level']
            st.subheader("📊 Results")

            if internal_type == 'fd':
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Risk Level", f"{risk_color(risk_level)} {risk_level}")
                with col2:
                    st.metric(
                        "Total Amount",
                        f"₹{result['total_amount']:,.0f}",
                        delta=f"+₹{result['interest_earned']:,.0f} earned",
                    )
                with col3:
                    st.metric("Interest Rate", f"{result['rate']:.1f}% p.a.")
            else:
                expected_return = result.get('expected_return', 0)
                total_value = amount + expected_return
                confidence = result.get('confidence', 0)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Risk Level", f"{risk_color(risk_level)} {risk_level}")
                with col2:
                    st.metric(
                        "Expected Profit",
                        f"₹{expected_return:,.0f}",
                        delta=f"Total: ₹{total_value:,.0f}",
                    )
                with col3:
                    st.metric("Model Confidence", f"{confidence:.1%}")

            # ── Explanation ──────────────────────────────────────────────────
            st.subheader("💡 Explanation")
            st.info(result['explanation'])

            # ── Detailed breakdown ───────────────────────────────────────────
            with st.expander("📈 Detailed Breakdown"):
                st.markdown(f"**Investment Type:** {investment_type}")
                st.markdown(f"**Principal:** ₹{amount:,}")
                st.markdown(f"**Duration:** {time_period} year{'s' if time_period > 1 else ''}")

                if internal_type == 'fd':
                    st.markdown(f"**Interest Rate:** {result['rate']:.1f}% per annum")
                    st.markdown(f"**Interest Earned:** ₹{result['interest_earned']:,.0f}")
                    st.markdown(f"**Maturity Value:** ₹{result['total_amount']:,.0f}")
                else:
                    rates_info = {
                        'stock': {'Low': '8%', 'Medium': '15%', 'High': '25%'},
                        'mutual_fund': {'Low': '6%', 'Medium': '12%', 'High': '20%'},
                    }
                    rate_table = rates_info.get(internal_type, {})
                    st.markdown(f"**Assumed Annual Rate ({risk_level} risk):** {rate_table.get(risk_level, 'N/A')} p.a.")
                    st.markdown(f"**Expected Profit:** ₹{result.get('expected_return', 0):,.0f}")
                    st.markdown(f"**Total Portfolio Value:** ₹{amount + result.get('expected_return', 0):,.0f}")
                    st.markdown(f"**Model Confidence:** {result.get('confidence', 0):.1%}")

            # ── Risk guide ───────────────────────────────────────────────────
            with st.expander("📘 Risk Level Guide"):
                st.markdown("""
| Risk | Stocks | Mutual Funds | Fixed Deposit |
|------|--------|--------------|---------------|
| 🟢 Low | ~8% p.a. | ~6% p.a. | ~5.5–7.5% p.a. |
| 🟡 Medium | ~15% p.a. | ~12% p.a. | — |
| 🔴 High | ~25% p.a. | ~20% p.a. | — |

> Returns are compound annual estimates. Past market data does not guarantee future performance.
""")

    # ── Footer ───────────────────────────────────────────────────────────────
    st.divider()
    st.markdown(
        "<div style='text-align:center; color:gray; font-size:0.85em;'>"
        "Built with ❤️ using Streamlit · scikit-learn Random Forest · Indian market data"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
