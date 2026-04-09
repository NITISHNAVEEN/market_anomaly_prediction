# OilSense — Market Anomaly Intelligence Dashboard

**Subject:** Time Series Analysis  
**Department:** Data Science and Artificial Intelligence  
**Institution:** IIIT Dharwad  
**Supervisor:** Dr. Nataraj K  

---

## Team

| Name | Roll Number |
|------|-------------|
| Saswath Chowta | 24BCS031 |
| Anchal Jaiswal | 24BDS003 |
| Nitish Naveen | 24BDS050 |



---

## Project Overview

OilSense is an end-to-end market anomaly detection and prediction system built on crude oil (WTI) time-series data. It combines a BERT-style Transformer encoder, XGBoost with SMOTE oversampling, and an Isolation Forest into a stacked ensemble — served through a live Flask dashboard that fetches daily OHLCV data from Yahoo Finance.

The system detects and predicts three market anomalies:
- **SPIKE** — abnormal upward price movement (predicted 3 days ahead)
- **CRASH** — abnormal downward price movement (predicted 3 days ahead)
- **BUBBLE** — sustained price extension above the 50-day moving average (current day)

---

## Repository Structure

```
project/
│
├── ensemble_anomaly_pipeline.ipynb   # Full training pipeline (Transformer + XGBoost + IsolationForest + Meta-LR)
├── crude_oil_history.csv          # Historical OHLCV data (CL=F, 2017–2026)
├── data_collection_timeseries.py                # Yahoo Finance data fetcher
│
├── ensemble_transformer.pt        # Saved Transformer weights + scaler
├── ensemble_xgb.pkl               # Saved XGBoost, IsolationForest, Meta-classifier
│
├── app.py                         # Flask backend (live + backtest modes)
│
└── templates/
    └── index.html                 # Full dashboard UI (Chart.js, dark theme)
```

---

## Setup & Installation

### 1. Install Python dependencies
```bash
pip install flask yfinance torch xgboost scikit-learn imbalanced-learn pandas numpy
```

### 2. Train the model (if starting fresh)
```bash
# Fetch fresh data first
python scrape_yahoo.py

# Train the ensemble
jupyter notebook ensemble_anomaly_pipeline.ipynb
# or run as script (remove %matplotlib inline first)
python ensemble_anomaly_pipeline.py
```
This saves `ensemble_transformer.pt` and `ensemble_xgb.pkl`.

### 3. Run the dashboard
```bash
python app.py
```
Open **http://127.0.0.1:5000** in your browser.

---

## Dashboard Features

### ⚡ Live Mode
- Fetches today's OHLCV data from Yahoo Finance automatically
- Runs the full ensemble pipeline in real-time
- Displays 3-day ahead SPIKE/CRASH prediction and current-day BUBBLE status
- Price chart with MA20, MA50, and forecast cone with uncertainty band
- Volatility stress indicator (LOW / MEDIUM / HIGH)
- Auto-refreshes every 5 minutes

### 🔍 Backtest Mode
- Select any date ≤ today minus 3 days
- Model predicts as if it were that date (no future data leakage)
- Actual prices for the next 3 days are fetched and overlaid on the chart in green
- Verdict banner shows whether the prediction was correct and % price change

### Multi-Ticker Support
- Type any Yahoo Finance ticker (e.g., `CL=F`,`BZ=F`, `GC=F`, `NG=F`, `AAPL`, `SPY`)
- Full pipeline re-runs against the new symbol

---

## Model Architecture

### Ensemble Stack

```
Input (30-day OHLCV window)
        │
   ┌────┴────────────┐
   │                 │
Transformer       XGBoost
(sequence)      (point features
                 + SMOTE)
   │                 │
   └────┬────────────┘
        │         │
   [4 probs] [4 probs] [ISO score]
        │
   Meta Logistic Regression
        │
   Final Prediction (NORMAL / SPIKE / CRASH / BUBBLE)
```

### Feature Groups

**Sequence features (Transformer):** Return, LogReturn, Range, MA5/20/50 ratios, Volatility5/20, RSI, MACD, VolumeRatio, Momentum5/20

**Point features (XGBoost):** All of the above + ZScore5, ZScore20, ATR14, VolumeSurge, PriceAccel, GapOpen, UpperShadow, LowerShadow

### Labelling Strategy (Horizon = 3 days)
- **SPIKE:** `max(return[t+1:t+3]) > +2σ`
- **CRASH:** `min(return[t+1:t+3]) < -2σ`
- **BUBBLE:** `MA50_ratio > 1.10 AND Momentum20 > 10%` (current day)

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Window size | 30 trading days |
| Train / Val / Test split | 80% / 10% / 10% |
| Epochs (max) | 25 |
| Early stopping patience | 6 |
| Optimizer | AdamW (lr=3e-4) |
| Scheduler | Cosine Annealing |
| Batch size | 64 |
| Class imbalance handling | Balanced weights × [1, 2, 2, 1] + SMOTE for XGBoost |

---

## Test Results

| Model | Accuracy |
|-------|----------|
| Transformer alone | ~37% |
| XGBoost alone | ~98.9% |
| **Ensemble** | **~98.9%** |

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| NORMAL | 0.99 | 0.99 | 0.99 |
| SPIKE | 1.00 | 1.00 | 1.00 |
| CRASH | 1.00 | 1.00 | 1.00 |
| BUBBLE | 0.95 | 0.95 | 0.95 |

ROC-AUC (macro OvR): **0.9997**

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Dashboard UI |
| `/api/predict?ticker=CL=F` | GET | Live prediction for given ticker |
| `/api/backtest?ticker=CL=F&date=2025-01-15` | GET | Backtest prediction for given date |
| `/api/health` | GET | Model load status and config |

---

## Notes

- The model was **trained on crude oil (CL=F)** data only. Applying it to other tickers uses the same feature logic but predictions may be less calibrated for markets with different volatility regimes.
- Predictions are most reliable **after market close** each day — intra-day data from Yahoo Finance is a partial candle.
- The forecast cone on the chart is an **illustrative projection** based on historical volatility, not a direct model output.
