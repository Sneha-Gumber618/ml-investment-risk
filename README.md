# heyyy — Investment Analyzer

Analyzes risk and expected returns for Stocks, Mutual Funds, and Fixed Deposits using ML.

## Project Structure

```
heyyy/
├── app/
│   └── streamlit_app.py          ← Streamlit UI
├── data/
│   ├── mutual_fund_nav_history.parquet
│   └── RELIANCE.csv
├── models/                        ← auto-created by train_models.py
│   ├── mf_model.pkl
│   └── stock_model.pkl
├── src/
│   ├── shared/                    ← shared utilities (no duplication)
│   │   ├── __init__.py
│   │   ├── base_predict.py        ← load_model, calculate_expected_return, generate_explanation
│   │   └── base_preprocess.py     ← add_rolling_features
│   ├── stock/
│   │   ├── predict.py             ← stock-specific prediction
│   │   ├── preprocess.py          ← stock CSV cleaning + feature engineering
│   │   └── train.py               ← stock model training
│   ├── mutual_fund/
│   │   ├── predict.py             ← MF-specific prediction
│   │   ├── preprocess.py          ← MF Parquet cleaning + feature engineering
│   │   └── train.py               ← MF model training
│   ├── fd_calculator.py           ← FD math (no ML needed, no subfolder)
│   └── pipeline/
│       └── main_pipeline.py       ← orchestrates everything
├── train_models.py                ← run this first
├── requirements.txt
└── README.md
```

## Files to DELETE from your old project

- `simple_app.py` (root) — old duplicate of `app/streamlit_app.py`
- `src/fd/` folder and everything in it — replaced by `src/fd_calculator.py`

## Setup

```bash
pip install -r requirements.txt

# Train models (run once)
python train_models.py

# Start the app
streamlit run app/streamlit_app.py
```

## What was fixed

| Problem | Old code | Fixed |
|---|---|---|
| `load_model()` duplicated | stock/predict.py + mf/predict.py | `src/shared/base_predict.py` |
| `generate_explanation()` duplicated | both predict.py files | `src/shared/base_predict.py` |
| `calculate_expected_return()` duplicated | both predict.py files | shared, rates passed as param |
| Rolling feature code duplicated | both preprocess.py files | `src/shared/base_preprocess.py` |
| `src/fd/` folder for 1 file | `src/fd/fd_calculator.py` | `src/fd_calculator.py` |
| `simple_app.py` stale duplicate | root level | deleted |