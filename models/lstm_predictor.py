# models/lstm_predictor.py
import random
import yfinance as yf
import pandas as pd
import numpy as np
import requests

from .config import API_KEY, WINDOW_SIZE, EPOCHS, BATCH_SIZE, MODEL_PATH
from .preprocess import preprocess_data
from .utils import load_trained_model
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
    if not data:
        raise ValueError("No data returned or API limit reached.")
    data = r.json().get('Time Series (Daily)', {})
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
    _, _, scaler = preprocess_data(np.array(data), WINDOW_SIZE)
    recent_prices = np.array(data[-WINDOW_SIZE:])
    prices_scaled = scaler.transform(recent_prices.reshape(-1, 1))
    X_pred = np.array([prices_scaled])
    model = load_trained_model(MODEL_PATH)
    pred_scaled = model.predict(X_pred, verbose=0)
    projected_price = scaler.inverse_transform(pred_scaled)[0]

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

# Example:
# last_60 = data[-60:]   # last 60 closing prices
# predicted_price = predict_next_price(model, last_60, scaler)
# print("Predicted next price:", predicted_price)



# def get_prediction(symbol):
#     """
#     Simulates a price prediction.
#     In a real implementation, this function would use a trained LSTM model.
#     """
#     try:
#         ticker = yf.Ticker(symbol)
#         data = ticker.history(period="1d", interval="1m")
#         if not data.empty:
#             current_price = data['Close'].iloc[-1]
#             # --- SIMULATION LOGIC ---
#             # In a real app, you would preprocess recent data and use model.predict()
#             trend = random.choice(["upward", "downward"])
#             change_factor = random.uniform(0.01, 0.05)

#             if trend == "upward":
#                 predicted_price = current_price * (1 + change_factor)
#             elif trend == "downward":
#                 predicted_price = current_price * (1 - change_factor)
#             else:
#                 predicted_price = current_price

#             return {
#                 'symbol': symbol,
#                 'trend': trend,
#                 'projected_price': predicted_price
#             }
#         else:
#             return f"Cannot generate a prediction without current data for {symbol}."
#     except Exception as e:
#         return {'error': f"An error occurred while generating a prediction for {symbol}."}



# def main():
#     print("Fetching data...")
#     data = fetch_stock_data(SYMBOL, API_KEY)
#     print("Preprocessing...")
#     X, y, scaler = preprocess_data(np.array(data), WINDOW_SIZE)
#     print("Training...")
#     model = train_model(X, y, WINDOW_SIZE, EPOCHS, BATCH_SIZE)
#     print(f"Saving model to {MODEL_PATH} ...")
#     save_model(model, MODEL_PATH)
#     recent_prices = np.array(data[-WINDOW_SIZE:])
#     pred = predict_next_price(model, recent_prices, scaler)
#     print(f"Predicted next close for {SYMBOL}: {pred:.2f}")

# if __name__ == "__main__":
#     main()
