"""
Stock Market Prediction Web App
LSTM-based closing price predictor for NSE/BSE stocks.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

# ── Suppress TF noise ─────────────────────────────────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
logging.getLogger("tensorflow").setLevel(logging.ERROR)

app = Flask(__name__)

# ── Global model (loaded once at startup) ─────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "stock_model.keras")
model = None

LOOKBACK = 100          # sequence length the model was trained on
TRAIN_SPLIT = 0.70      # must match notebook


def load_model():
    """Load the Keras model once and cache it globally."""
    global model
    if model is None:
        model = tf.keras.models.load_model(MODEL_PATH)
        app.logger.info("Model loaded — input shape: %s", model.input_shape)
    return model


def fetch_stock_data(ticker: str, years: int = 5) -> pd.DataFrame:
    """
    Download historical OHLCV data from Yahoo Finance.
    Returns a DataFrame with a DatetimeIndex and standard column names.
    """
    end   = datetime.today()
    start = end - timedelta(days=years * 365)
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data found for ticker '{ticker}'. "
                         "Check the symbol and try again.")
    df = df.reset_index()
    # Flatten MultiIndex columns that yfinance sometimes returns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if col[1] == '' else col[0] for col in df.columns]
    df['Date'] = pd.to_datetime(df['Date'])
    return df


def build_inference_pipeline(df: pd.DataFrame):
    """
    Replicate the notebook's exact preprocessing + prediction pipeline.

    Notebook logic:
      1. Use ONLY the 'Close' column.
      2. Fit MinMaxScaler on the FULL historical Close series (like the notebook
         did with scaler.fit_transform on data_training, then re-fit on final_df).
      3. Build sliding windows of length LOOKBACK from the last LOOKBACK+N days.
      4. Predict, then inverse-scale using the scaler's scale_ factor.

    Returns
    -------
    dates_pred  : list of ISO date strings for predicted points
    y_actual    : list of actual close prices (float)
    y_predicted : list of predicted close prices (float)
    last_pred   : next-day prediction (float)
    metrics     : dict with MAE, RMSE, MAPE
    indicators  : dict with MA100, MA200, EMA100, EMA200 series
    """
    close = df['Close'].copy()

    # ── Indicators ────────────────────────────────────────────────────────────
    ma100  = close.rolling(100).mean()
    ma200  = close.rolling(200).mean()
    ema100 = close.ewm(span=100, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()

    # ── Train / test split (mirrors the notebook 70/30) ──────────────────────
    split_idx = int(len(close) * TRAIN_SPLIT)
    data_training = pd.DataFrame(close[:split_idx])
    data_testing  = pd.DataFrame(close[split_idx:])

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit_transform(data_training)          # fit on training portion

    # Build final_df = last 100 training rows + all test rows (notebook exact)
    past_100_days = data_training.tail(LOOKBACK)
    final_df      = pd.concat([past_100_days, data_testing], ignore_index=True)
    input_data    = scaler.fit_transform(final_df)   # re-fit like notebook

    # Sliding window for test set
    x_test, y_test = [], []
    for i in range(LOOKBACK, input_data.shape[0]):
        x_test.append(input_data[i - LOOKBACK:i])
        y_test.append(input_data[i, 0])

    x_test = np.array(x_test).reshape(-1, LOOKBACK, 1)
    y_test = np.array(y_test)

    # ── Inference ─────────────────────────────────────────────────────────────
    loaded_model = load_model()
    y_pred_scaled = loaded_model.predict(x_test, verbose=0)

    # Inverse-scale using the scaler's fitted scale_ (notebook: 1/scale_[0])
    scale_factor  = 1.0 / scaler.scale_[0]
    y_predicted   = (y_pred_scaled.flatten() * scale_factor).tolist()
    y_actual      = (y_test          * scale_factor).tolist()

    # Dates for test predictions (align with data_testing after lookback offset)
    test_dates = df['Date'].iloc[split_idx:].reset_index(drop=True)
    dates_pred = test_dates.dt.strftime('%Y-%m-%d').tolist()

    # ── Next-day prediction ───────────────────────────────────────────────────
    # Use the last LOOKBACK Close values from the full series
    last_window = close.values[-LOOKBACK:].reshape(-1, 1)
    next_scaler = MinMaxScaler(feature_range=(0, 1))
    last_window_scaled = next_scaler.fit_transform(last_window)
    x_next = last_window_scaled.reshape(1, LOOKBACK, 1)
    next_pred_scaled = loaded_model.predict(x_next, verbose=0)[0, 0]
    next_day_scale   = 1.0 / next_scaler.scale_[0]
    last_pred = float(next_pred_scaled * next_day_scale)

    # ── Metrics ───────────────────────────────────────────────────────────────
    ya  = np.array(y_actual)
    yp  = np.array(y_predicted)
    mae  = float(np.mean(np.abs(ya - yp)))
    rmse = float(np.sqrt(np.mean((ya - yp) ** 2)))
    mape = float(np.mean(np.abs((ya - yp) / (ya + 1e-8))) * 100)

    # Trim indicator series to the test window for clean charting
    n_test = len(dates_pred)
    def tail_list(series):
        vals = series.iloc[split_idx:].values.tolist()
        return [None if np.isnan(v) else round(v, 4) for v in vals[:n_test]]

    indicators = {
        "ma100":  tail_list(ma100),
        "ma200":  tail_list(ma200),
        "ema100": tail_list(ema100),
        "ema200": tail_list(ema200),
    }

    return dates_pred, y_actual, y_predicted, last_pred, \
           {"mae": round(mae, 4), "rmse": round(rmse, 4), "mape": round(mape, 4)}, \
           indicators


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    POST /predict
    Body: { "ticker": "POWERGRID.NS", "years": 5 }
    Returns JSON with prediction data.
    """
    body   = request.get_json(silent=True) or {}
    ticker = (body.get("ticker") or "").strip().upper()
    years  = int(body.get("years", 5))

    # ── Validation ────────────────────────────────────────────────────────────
    if not ticker:
        return jsonify({"error": "Ticker symbol is required."}), 400
    if not 1 <= years <= 20:
        return jsonify({"error": "Years must be between 1 and 20."}), 400

    try:
        df = fetch_stock_data(ticker, years)
        if len(df) < LOOKBACK + 50:
            return jsonify({"error": f"Not enough data for '{ticker}'. "
                            "Try a longer time range or a more liquid stock."}), 400

        dates, actuals, predictions, next_price, metrics, indicators = \
            build_inference_pipeline(df)

        current_price = float(df['Close'].iloc[-1])
        pct_change    = round((next_price - current_price) / current_price * 100, 2)
        signal        = "BUY" if pct_change > 0.5 else "SELL" if pct_change < -0.5 else "HOLD"

        return jsonify({
            "ticker":        ticker,
            "company":       df.get("longName", ticker) if hasattr(df, "get") else ticker,
            "current_price": round(current_price, 4),
            "next_pred":     round(next_price, 4),
            "pct_change":    pct_change,
            "signal":        signal,
            "metrics":       metrics,
            "chart": {
                "dates":       dates,
                "actual":      [round(v, 4) for v in actuals],
                "predicted":   [round(v, 4) for v in predictions],
                "indicators":  indicators,
            },
            "data_points": len(df),
            "last_date":   df['Date'].iloc[-1].strftime('%Y-%m-%d'),
        })

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        app.logger.exception("Prediction error for %s", ticker)
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    load_model()          # warm-up at startup
    app.run(debug=False, host="0.0.0.0", port=5000)
