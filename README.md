# StockSense — LSTM Stock Price Predictor

A Flask web application that wraps a trained 4-layer LSTM model for
next-day stock closing price prediction on NSE/BSE stocks.

---

## Project Structure

```
stock_app/
├── app.py                 # Flask backend + inference pipeline
├── stock_model.keras      # Trained Keras model (copy here)
├── requirements.txt
├── Procfile               # For Heroku / Railway
├── README.md
├── templates/
│   └── index.html         # Single-page UI
└── static/
    ├── css/style.css
    └── js/app.js
```

---

## Local Setup

### 1. Copy your model file

```bash
# Place stock_model.keras in the project root
cp /path/to/stock_model.keras stock_app/
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
cd stock_app
python app.py
```

Open http://localhost:5000 in your browser.

---

## How to Use

1. Enter a Yahoo Finance ticker (e.g. `POWERGRID.NS`, `RELIANCE.NS`, `TCS.NS`).
2. Select the amount of historical data to fetch (3–20 years).
3. Click **Run Prediction**.
4. View the next-day predicted closing price, signal (BUY/SELL/HOLD),
   accuracy metrics, and an interactive price chart.

**Ticker format:**
- NSE stocks → `SYMBOL.NS` (e.g. `HDFCBANK.NS`)
- BSE stocks → `SYMBOL.BO` (e.g. `500325.BO`)
- US stocks  → `SYMBOL` (e.g. `AAPL`)

---

## Inference Pipeline

The backend replicates the notebook's exact preprocessing:

1. Fetch OHLCV data from Yahoo Finance via `yfinance`.
2. Extract `Close` price series only.
3. Apply 70/30 train-test split.
4. Fit `MinMaxScaler` on the training portion.
5. Build sliding windows of **100 days** for the test set.
6. Feed into the stacked LSTM (50 → 60 → 80 → 120 units).
7. Inverse-scale predictions using `1 / scaler.scale_[0]`.
8. Predict next day using the final 100 closing prices.

---

## Model Architecture

```
LSTM(50, relu, return_seq=True) → Dropout(0.2)
LSTM(60, relu, return_seq=True) → Dropout(0.3)
LSTM(80, relu, return_seq=True) → Dropout(0.4)
LSTM(120, relu)                 → Dropout(0.5)
Dense(1)
```

- Optimizer: Adam
- Loss: Mean Squared Error
- Lookback window: 100 days
- Originally trained on: NSE:POWERGRID (2000–2024)

> **Note:** The model was trained exclusively on POWERGRID.NS. Predictions
> on other stocks will use the same weights and may have reduced accuracy.
> For best results, retrain the model on the target stock.

---

## Deployment

### Render / Railway (recommended, free tier)

1. Push the `stock_app/` folder to a GitHub repo.
2. Create a new Web Service, set:
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `gunicorn app:app --workers 2 --timeout 120`
3. Add `stock_model.keras` to the repo (or use a persistent disk volume).

### Heroku

```bash
heroku create your-app-name
git push heroku main
```

### Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 5000
CMD ["gunicorn", "app:app", "--workers", "2", "--timeout", "120", "--bind", "0.0.0.0:5000"]
```

```bash
docker build -t stocksense .
docker run -p 5000:5000 stocksense
```

---

## Disclaimer

This application is for **educational purposes only**.
Predictions are not financial advice. Stock market investments
carry significant risk. Always consult a qualified financial advisor.
