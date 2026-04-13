"""
Crude Oil Market Anomaly Dashboard — Flask Backend
====================================================
Fetches live OHLCV from Yahoo Finance (CL=F),
engineers features, runs the saved ensemble model,
and serves predictions + chart data via JSON API.

Run:
    pip install flask yfinance xgboost scikit-learn imbalanced-learn torch
    python app.py

Then open:  http://127.0.0.1:5000
"""

import os, math, pickle, warnings, traceback

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yfinance as yf
from flask import Flask, jsonify, render_template, request
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# CONFIG  — adjust paths if needed
# ──────────────────────────────────────────────
TRANSFORMER_PT = "ensemble_transformer.pt"   # saved from ensemble pipeline
ENSEMBLE_PKL   = "ensemble_xgb.pkl"          # saved from ensemble pipeline
DEFAULT_TICKER = "CL=F"   # fallback if none provided
LOOKBACK_DAYS  = 120                          # days of history to fetch
WINDOW         = 30                           # must match training config
FORECAST_DAYS  = 3                            # horizon for spike/crash
LABEL_MAP      = {0:"NORMAL", 1:"SPIKE", 2:"CRASH", 3:"BUBBLE"}

SEQ_FEAT = [
    "Return","LogReturn","Range",
    "MA5_ratio","MA20_ratio","MA50_ratio",
    "Volatility5","Volatility20","RSI","MACD",
    "VolumeRatio","Momentum5","Momentum20"
]
XGB_FEAT = [
    "Return","LogReturn","Range","RSI","MACD",
    "VolumeRatio","Momentum5","Momentum20",
    "ZScore5","ZScore20","ATR14","VolumeSurge",
    "PriceAccel","GapOpen","UpperShadow","LowerShadow",
]

app = Flask(__name__)

# ──────────────────────────────────────────────
# MODEL DEFINITIONS  (must match training code)
# ──────────────────────────────────────────────
class PosEnc(nn.Module):
    def __init__(self, d, maxlen=512, p=0.1):
        super().__init__()
        self.drop = nn.Dropout(p)
        pe  = torch.zeros(maxlen, d)
        pos = torch.arange(maxlen).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x): return self.drop(x + self.pe[:, :x.size(1)])

class AnomalyTransformer(nn.Module):
    def __init__(self, in_dim, seq_len, d=128, heads=4, layers=3, ff=256, drop=0.2, nc=4):
        super().__init__()
        self.proj = nn.Linear(in_dim, d)
        self.cls  = nn.Parameter(torch.zeros(1, 1, d))
        self.pos  = PosEnc(d, seq_len+1, drop)
        enc_layer = nn.TransformerEncoderLayer(d, heads, ff, drop,
                      activation="gelu", batch_first=True, norm_first=True)
        self.enc  = nn.TransformerEncoder(enc_layer, layers, enable_nested_tensor=False)
        self.head = nn.Sequential(
            nn.LayerNorm(d), nn.Dropout(drop),
            nn.Linear(d, 64), nn.GELU(), nn.Linear(64, nc))
    def forward(self, x):
        B = x.size(0)
        h = self.proj(x)
        h = torch.cat([self.cls.expand(B, -1, -1), h], dim=1)
        h = self.pos(h); h = self.enc(h)
        return self.head(h[:, 0])

# ──────────────────────────────────────────────
# LOAD MODELS  (done once at startup)
# ──────────────────────────────────────────────
def load_models():
    device = torch.device("cpu")

    # Transformer
    ckpt = torch.load(TRANSFORMER_PT, map_location=device, weights_only=False)
    cfg  = ckpt.get("model_config", {})
    transformer = AnomalyTransformer(
        in_dim  = cfg.get("in_dim",  len(SEQ_FEAT)),
        seq_len = cfg.get("seq_len", WINDOW),
        d       = cfg.get("d",       128),
        heads   = cfg.get("heads",   4),
        layers  = cfg.get("layers",  3),
        ff      = cfg.get("ff",      256),
        drop    = cfg.get("drop",    0.2),
        nc      = cfg.get("nc",      4),
    ).to(device)
    transformer.load_state_dict(ckpt["model_state_dict"])
    transformer.eval()

    seq_scaler       = StandardScaler()
    seq_scaler.mean_ = np.array(ckpt["seq_scaler_mean"])
    seq_scaler.scale_= np.array(ckpt["seq_scaler_scale"])
    seq_scaler.var_  = seq_scaler.scale_ ** 2
    seq_scaler.n_features_in_ = len(SEQ_FEAT)
    seq_scaler.n_samples_seen_= 1000

    # XGBoost + IsolationForest + Meta-classifier
    with open(ENSEMBLE_PKL, "rb") as f:
        bundle = pickle.load(f)

    return transformer, seq_scaler, bundle, device

try:
    TRANSFORMER, SEQ_SCALER, BUNDLE, DEVICE = load_models()
    MODEL_LOADED = True
    print("✅  Models loaded successfully.")
except Exception as e:
    MODEL_LOADED = False
    MODEL_ERROR  = str(e)
    print(f"⚠️  Model load failed: {e}")

# ──────────────────────────────────────────────
# FEATURE ENGINEERING
# ──────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c = df["Close"]; h = df["High"]; l = df["Low"]; o = df["Open"]

    df["Return"]       = c.pct_change()
    df["LogReturn"]    = np.log(c / c.shift(1))
    df["Range"]        = (h - l) / c
    df["MA5"]          = c.rolling(5).mean()
    df["MA20"]         = c.rolling(20).mean()
    df["MA50"]         = c.rolling(50).mean()
    df["MA5_ratio"]    = c / df["MA5"]
    df["MA20_ratio"]   = c / df["MA20"]
    df["MA50_ratio"]   = c / df["MA50"]
    df["Volatility5"]  = df["Return"].rolling(5).std()
    df["Volatility20"] = df["Return"].rolling(20).std()
    d  = c.diff()
    g  = d.clip(lower=0).rolling(14).mean()
    ls = (-d.clip(upper=0)).rolling(14).mean()
    df["RSI"]          = 100 - 100 / (1 + g / (ls + 1e-9))
    df["MACD"]         = c.ewm(span=12).mean() - c.ewm(span=26).mean()
    df["VolumeRatio"]  = df["Volume"] / (df["Volume"].rolling(20).mean() + 1e-9)
    df["Momentum5"]    = c / c.shift(5)  - 1
    df["Momentum20"]   = c / c.shift(20) - 1

    # Spike/crash leading indicators
    r5m  = df["Return"].rolling(5).mean()
    r5s  = df["Return"].rolling(5).std()
    r20m = df["Return"].rolling(20).mean()
    r20s = df["Return"].rolling(20).std()
    df["ZScore5"]      = (df["Return"] - r5m)  / (r5s  + 1e-9)
    df["ZScore20"]     = (df["Return"] - r20m) / (r20s + 1e-9)
    tr                 = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    df["ATR14"]        = tr.rolling(14).mean() / (c + 1e-9)
    df["VolumeSurge"]  = df["Volume"] / (df["Volume"].rolling(5).mean() + 1e-9)
    df["PriceAccel"]   = df["Return"] - df["Return"].shift(1)
    df["GapOpen"]      = (o - c.shift()) / (c.shift() + 1e-9)
    df["UpperShadow"]  = (h - pd.concat([c, o], axis=1).max(axis=1)) / (c + 1e-9)
    df["LowerShadow"]  = (pd.concat([c, o], axis=1).min(axis=1) - l)  / (c + 1e-9)

    return df.dropna().reset_index(drop=True)

# ──────────────────────────────────────────────
# FETCH LIVE DATA FROM YAHOO FINANCE
# ──────────────────────────────────────────────
def fetch_live_data(ticker=DEFAULT_TICKER, days=LOOKBACK_DAYS):
    end   = datetime.today()
    start = end - timedelta(days=days + 30)   # extra buffer for dropna
    df    = yf.download(ticker, start=start.strftime("%Y-%m-%d"),
                         end=end.strftime("%Y-%m-%d"),
                         interval="1d", auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data returned from Yahoo Finance for {ticker}. "
                         "Check your internet connection.")

    # Flatten multi-level columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()
    df.columns = [c.strip() for c in df.columns]

    # Rename 'Adj Close' → 'Close' if needed
    if "Adj Close" in df.columns and "Close" not in df.columns:
        df.rename(columns={"Adj Close": "Close"}, inplace=True)

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    required = ["Date","Open","High","Low","Close","Volume"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in fetched data.")

    return df[required]

# ──────────────────────────────────────────────
# PREDICTION ENGINE
# ──────────────────────────────────────────────
def run_prediction(df_feat: pd.DataFrame):
    """
    Given a fully-featured dataframe, predict on the LAST row
    using the ensemble: Transformer + XGBoost + IsolationForest + Meta.
    Returns dict with prediction, probabilities, and bubble status.
    """
    if not MODEL_LOADED:
        raise RuntimeError("Models not loaded.")

    xgb_model   = BUNDLE["xgb"]
    xgb_scaler  = BUNDLE["xgb_scaler"]
    iso         = BUNDLE["iso"]
    meta_clf    = BUNDLE["meta_clf"]
    meta_scaler = BUNDLE["meta_scaler"]

    # ── Transformer: use last WINDOW rows ──
    seq_vals = df_feat[SEQ_FEAT].values.astype(np.float32)
    seq_norm = SEQ_SCALER.transform(seq_vals)
    window   = seq_norm[-WINDOW:]                            # (30, 13)
    window_t = torch.tensor(window, dtype=torch.float32).unsqueeze(0)  # (1, 30, 13)

    with torch.no_grad():
        tf_logits = TRANSFORMER(window_t)
        tf_probs  = torch.softmax(tf_logits, -1).squeeze().cpu().detach().tolist()
    tf_probs = np.array(tf_probs, dtype=np.float32)

    # ── XGBoost: last row point features ──
    pt_raw   = df_feat[XGB_FEAT].values[-1].astype(np.float32)
    pt_s     = xgb_scaler.transform(pt_raw.reshape(1, -1))
    xgb_probs= xgb_model.predict_proba(pt_s)[0].astype(np.float32)

    # ── Isolation Forest anomaly score ──
    iso_score= float(-iso.score_samples(pt_raw.reshape(1, -1))[0])

    # ── Meta-classifier ──
    meta_feat = np.hstack([tf_probs, xgb_probs, [iso_score]]).astype(np.float32)
    meta_s    = meta_scaler.transform(meta_feat.reshape(1, -1))
    meta_probs= meta_clf.predict_proba(meta_s)[0].astype(np.float32)
    pred      = int(meta_probs.argmax())

    # ── Bubble tracker (current-day rule — independent of ML) ──
    last = df_feat.iloc[-1]
    bubble_active   = bool(last["MA50_ratio"] > 1.10 and last["Momentum20"] > 0.10)
    bubble_severity = float(last["MA50_ratio"] - 1.0) * 100   # % above MA50

    # ── Volatility stress (precursor signal) ──
    vol_ratio     = float(last["Volatility5"] / (last["Volatility20"] + 1e-9))
    stress_level  = "HIGH" if vol_ratio > 1.5 else "MEDIUM" if vol_ratio > 1.0 else "LOW"

    return {
        "prediction"      : pred,
        "label"           : LABEL_MAP[pred],
        "probabilities"   : {LABEL_MAP[i]: round(float(meta_probs[i]) * 100, 1) for i in range(4)},
        "sub_predictions" : {
            "transformer" : LABEL_MAP[int(tf_probs.argmax())],
            "xgboost"     : LABEL_MAP[int(xgb_probs.argmax())],
            "iso_score"   : round(iso_score, 3),
        },
        "bubble_active"   : bubble_active,
        "bubble_severity" : round(bubble_severity, 2),
        "stress_level"    : stress_level,
        "vol_ratio"       : round(vol_ratio, 3),
        "last_close"      : round(float(df_feat["Close"].iloc[-1]), 2),
        "last_date"       : str(df_feat["Date"].iloc[-1].date()),
        "forecast_horizon": FORECAST_DAYS,
    }

# ──────────────────────────────────────────────
# CHART DATA BUILDER
# ──────────────────────────────────────────────
def build_chart_data(df_feat: pd.DataFrame, pred_result: dict):
    """
    Returns OHLCV history + projected price range for next 3 days
    + rolling volatility bands for context.
    """
    # Last 90 rows for chart readability
    df_chart = df_feat.tail(90).reset_index(drop=True)

    dates  = [str(d.date()) for d in df_chart["Date"]]
    closes = [round(float(v), 2) for v in df_chart["Close"]]
    highs  = [round(float(v), 2) for v in df_chart["High"]]
    lows   = [round(float(v), 2) for v in df_chart["Low"]]
    ma20   = [round(float(v), 2) if not np.isnan(v) else None
              for v in df_chart["MA20"]]
    ma50   = [round(float(v), 2) if not np.isnan(v) else None
              for v in df_chart["MA50"]]

    # Bollinger bands (20-day, 2σ)
    std20  = df_chart["Close"].rolling(20).std()
    bb_up  = [round(float(m + 2*s), 2) if not np.isnan(m) else None
              for m, s in zip(df_chart["MA20"], std20)]
    bb_lo  = [round(float(m - 2*s), 2) if not np.isnan(m) else None
              for m, s in zip(df_chart["MA20"], std20)]

    # Projected range for next FORECAST_DAYS
    last_close   = pred_result["last_close"]
    last_date    = pd.to_datetime(pred_result["last_date"])
    daily_vol    = float(df_feat["Volatility20"].iloc[-1])
    label        = pred_result["label"]

    # Directional bias based on prediction
    bias = {"SPIKE": 0.015, "CRASH": -0.015, "BUBBLE": 0.008, "NORMAL": 0.002}[label]

    proj_dates  = []
    proj_mid    = []
    proj_upper  = []
    proj_lower  = []
    price = last_close
    # Add anchor point at last known date
    proj_dates.append(str(last_date.date()))
    proj_mid.append(price)
    proj_upper.append(price)
    proj_lower.append(price)

    for d in range(1, FORECAST_DAYS + 1):
        next_date = last_date + timedelta(days=d)
        # Skip weekends
        while next_date.weekday() >= 5:
            next_date += timedelta(days=1)
        price_mid   = price * (1 + bias)
        uncertainty = daily_vol * math.sqrt(d) * price
        proj_dates.append(str(next_date.date()))
        proj_mid.append(round(price_mid, 2))
        proj_upper.append(round(price_mid + 1.5 * uncertainty, 2))
        proj_lower.append(round(price_mid - 1.5 * uncertainty, 2))
        price = price_mid

    return {
        "history": {
            "dates" : dates,
            "closes": closes,
            "highs" : highs,
            "lows"  : lows,
            "ma20"  : ma20,
            "ma50"  : ma50,
            "bb_up" : bb_up,
            "bb_lo" : bb_lo,
        },
        "forecast": {
            "dates" : proj_dates,
            "mid"   : proj_mid,
            "upper" : proj_upper,
            "lower" : proj_lower,
        }
    }

# ──────────────────────────────────────────────
# UTILITY FUNCTIONS
# ──────────────────────────────────────────────
def replace_nan(obj):
    """Recursively replace NaN with None for JSON serialization."""
    if isinstance(obj, dict):
        return {k: replace_nan(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [replace_nan(item) for item in obj]
    elif isinstance(obj, float) and np.isnan(obj):
        return None
    else:
        return obj

# ──────────────────────────────────────────────
# ROUTES
# ──────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/predict")
def api_predict():
    try:
        ticker = request.args.get("ticker", DEFAULT_TICKER).strip().upper()
        if not ticker:
            return jsonify({"error": "No ticker provided."}), 400

        df_raw  = fetch_live_data(ticker=ticker)          # already accepts ticker
        df_feat = engineer_features(df_raw)
        if len(df_feat) < WINDOW + 5:
            return jsonify({"error": "Not enough data after feature engineering."}), 400
        pred    = run_prediction(df_feat)
        pred["ticker"] = ticker                           # tag it for the frontend
        chart   = build_chart_data(df_feat, pred)
        result  = replace_nan({"prediction": pred, "chart": chart, "status": "ok"})
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500
@app.route("/api/health")
def health():
    return jsonify({
        "model_loaded": MODEL_LOADED,
        "default_ticker": DEFAULT_TICKER,
        "window"        : WINDOW,
        "horizon"       : FORECAST_DAYS,
    })
@app.route("/api/backtest")
def api_backtest():
    try:
        ticker   = request.args.get("ticker", DEFAULT_TICKER).strip().upper()
        date_str = request.args.get("date", "").strip()

        if not date_str:
            return jsonify({"error": "No date provided."}), 400

        # Validate date
        try:
            query_date = pd.to_datetime(date_str).date()
        except Exception:
            return jsonify({"error": f"Invalid date format: {date_str}. Use YYYY-MM-DD."}), 400

        # Must be at least 3 days before today
        min_allowed = (datetime.today() - timedelta(days=3)).date()
        if query_date > min_allowed:
            return jsonify({"error": f"Date must be ≤ {min_allowed} (today minus 3 days)."}), 400

        # Fetch enough history before the query date for feature engineering
        fetch_start = query_date - timedelta(days=180)
        fetch_end   = query_date + timedelta(days=10)  # grab a few days after for actuals

        df_raw = yf.download(
            ticker,
            start=fetch_start.strftime("%Y-%m-%d"),
            end=fetch_end.strftime("%Y-%m-%d"),
            interval="1d", auto_adjust=True, progress=False
        )
        if df_raw.empty:
            return jsonify({"error": f"No data for {ticker} around {date_str}."}), 400

        # Flatten columns
        if isinstance(df_raw.columns, pd.MultiIndex):
            df_raw.columns = df_raw.columns.get_level_values(0)
        df_raw = df_raw.reset_index()
        df_raw.columns = [c.strip() for c in df_raw.columns]
        if "Adj Close" in df_raw.columns and "Close" not in df_raw.columns:
            df_raw.rename(columns={"Adj Close": "Close"}, inplace=True)
        df_raw["Date"] = pd.to_datetime(df_raw["Date"])
        df_raw = df_raw.sort_values("Date").reset_index(drop=True)

        # Split: history up to and including query_date, actuals after
        df_history = df_raw[df_raw["Date"].dt.date <= query_date].reset_index(drop=True)
        df_actuals = df_raw[df_raw["Date"].dt.date >  query_date].head(FORECAST_DAYS).reset_index(drop=True)

        if len(df_history) < WINDOW + 20:
            return jsonify({"error": "Not enough historical data before the selected date."}), 400

        # Engineer features on history only (no future leakage)
        df_feat = engineer_features(df_history)
        if len(df_feat) < WINDOW + 5:
            return jsonify({"error": "Not enough data after feature engineering."}), 400

        # Run prediction as of that date
        pred          = run_prediction(df_feat)
        pred["ticker"]= ticker
        pred["mode"]  = "backtest"
        pred["query_date"] = str(query_date)

        # Build chart data (history slice)
        chart = build_chart_data(df_feat, pred)

        # Attach actual prices after query_date for comparison
        actuals = {
            "dates" : [str(d.date()) for d in df_actuals["Date"]],
            "closes": [round(float(v), 2) for v in df_actuals["Close"]],
        }

        # Score: did the prediction match what actually happened?
        verdict = _score_prediction(pred, df_actuals)

        result = replace_nan({
            "prediction" : pred,
            "chart"      : chart,
            "actuals"    : actuals,
            "verdict"    : verdict,
            "status"     : "ok"
        })
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


def _score_prediction(pred: dict, df_actuals: pd.DataFrame) -> dict:
    """Compare prediction label against what actually happened in the next N days."""
    if df_actuals.empty:
        return {"score": "unknown", "reason": "No actual data available yet."}

    actual_returns  = df_actuals["Close"].pct_change().dropna()
    daily_vol       = pred.get("vol_ratio", 1.0) * 0.01   # rough σ proxy
    sigma           = max(df_actuals["Close"].std() / df_actuals["Close"].mean(), 0.015)

    max_return = float(actual_returns.max()) if len(actual_returns) else 0.0
    min_return = float(actual_returns.min()) if len(actual_returns) else 0.0

    actual_label = "NORMAL"
    if min_return < -2 * sigma:
        actual_label = "CRASH"
    elif max_return > 2 * sigma:
        actual_label = "SPIKE"

    # Bubble uses price level not return
    last_feat_close = pred["last_close"]
    for _, row in df_actuals.iterrows():
        chg = (float(row["Close"]) - last_feat_close) / last_feat_close
        if chg > 0.10:
            actual_label = "BUBBLE"
            break

    correct = pred["label"] == actual_label
    price_change = round(
        (float(df_actuals["Close"].iloc[-1]) - pred["last_close"]) / pred["last_close"] * 100, 2
    ) if not df_actuals.empty else None

    return {
        "predicted"    : pred["label"],
        "actual"       : actual_label,
        "correct"      : correct,
        "price_change" : price_change,
        "reason"       : f"Price moved {price_change:+.2f}% over next {len(df_actuals)} day(s). "
                         f"Predicted {pred['label']}, actual outcome classified as {pred['label']}."
                         if price_change is not None else "Insufficient data."
    }
if __name__ == "__main__":
    app.run(debug=True, port=5000)

