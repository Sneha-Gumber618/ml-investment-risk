# Investment Analysis App

A simple Python-based investment analysis application using Machine Learning and Streamlit.

## Project Structure

```
project/
|
|-- data/
|   |-- stock_data.csv          # Stock market data (to be added)
|   |-- mutual_fund_data.csv    # Mutual fund data (to be added)
|
|-- models/
|   |-- stock_model.pkl         # Trained stock model

The app uses Random Forest classifiers to predict risk levels (Low/Medium/High) and provides expected returns based on historical data patterns.

## 📁 Project Structure

2. Enter investment amount and time period
3. Click "Analyze" button
4. View results:
   - Risk level (Low/Medium/High)
   - Expected return amount
   - Simple explanation

## Output Example

```
Risk Level: High
Expected Return: $15,000
Explanation: High risk due to high volatility
```

## Notes

- Code kept simple for educational purposes
- No deep learning or complex patterns
- Basic Python, pandas, scikit-learn only
- Comments added for explanation
- Models trained to avoid always predicting same risk level
