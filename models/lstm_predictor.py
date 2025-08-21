# models/lstm_predictor.py
# import random
# import yfinance as yf
import pandas as pd
import numpy as np
import requests
import os

from .config import API_KEY, WINDOW_SIZE, EPOCHS, BATCH_SIZE, MODEL_PATH
from .preprocess import preprocess_data
from .utils import load_trained_model
from models.train import train_model
from models.utils import save_model
# In a real scenario, you would import tensorflow here
# import tensorflow as tf

# Placeholder for loading a trained model
# model = tf.keras.models.load_model('path/to/your/lstm_model.h5')


def fetch_stock_data(symbol, outputsize='compact'):
    url = 'https://www.alphavantage.co/query'
    params = {
        'function': 'TIME_SERIES_DAILY',
        'symbol': symbol,
        'apikey': API_KEY,
        'outputsize': outputsize  # 'compact' = 100 latest, 'full' = all
    }
    r = requests.get(url, params=params, timeout=15)
    data = r.json().get('Time Series (Daily)', {})
    if not data:
        raise ValueError("No data returned or API limit reached.")
    
    df = pd.DataFrame(data).T
    df = df.rename(columns={'4. close': 'close'})
    df['close'] = df['close'].astype(float)
    df = df.sort_index()
    return df['close'].values

def predict_next_price(symbol):
    # Fetch latest stock prices
    data = fetch_stock_data(symbol)
    if len(data) < WINDOW_SIZE:
        raise ValueError("Not enough data to predict.")
    # Preprocess (fit scaler on all history)
    x, y, scaler = preprocess_data(np.array(data), WINDOW_SIZE)
    
    print("Data for prediction:", data[-WINDOW_SIZE:])
    recent_prices = np.array(data[-WINDOW_SIZE:])
    prices_scaled = scaler.transform(recent_prices.reshape(-1, 1))
    X_pred = np.array([prices_scaled])

    if os.path.exists(MODEL_PATH):
        model = load_trained_model(MODEL_PATH)
    else:
        print("Training the LSTM model...")
        model = train_model(x, y, window_size=WINDOW_SIZE)
        save_model(model, MODEL_PATH)
        model = load_trained_model(MODEL_PATH)
    

    
    pred_scaled = model.predict(X_pred, verbose=0)
    projected_price = scaler.inverse_transform(pred_scaled).item()

    # Determine trend by comparing the projected price to the last price
    last_close = data[-1]
    if projected_price > last_close:
        trend = "upward"
    elif projected_price < last_close:
        trend = "downward"
    else:
        trend = "neutral"

    return {
        'symbol': symbol,
        'trend': trend,
        'projected_price': projected_price
    }

