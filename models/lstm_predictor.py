# models/lstm_predictor.py
# import yfinance as yf
import pandas as pd
import numpy as np
import requests
import os
import time
import json
from pathlib import Path

from .config import API_KEY, WINDOW_SIZE, EPOCHS, BATCH_SIZE, MODEL_PATH
from .preprocess import preprocess_data
from .utils import load_trained_model




def _cache_paths(symbol: str, outputsize: str):
    cache_dir = Path(".cache/alphavantage")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{symbol}_{outputsize}.json"
    return cache_file


def _load_from_cache(cache_file: Path, ttl_seconds: int):
    if cache_file.exists():
        try:
            stat = cache_file.stat()
            age = time.time() - stat.st_mtime
            if age <= ttl_seconds:
                with cache_file.open("r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            return None
    return None


def _save_to_cache(cache_file: Path, payload: dict):
    try:
        with cache_file.open("w", encoding="utf-8") as f:
            json.dump(payload, f)
    except Exception:
        pass


def fetch_stock_data(symbol, outputsize='compact', cache_ttl_seconds: int = 6 * 60 * 60, max_retries: int = 3):
    url = 'https://www.alphavantage.co/query'
    params = {
        'function': 'TIME_SERIES_DAILY',
        'symbol': symbol,
        'apikey': API_KEY,
        'outputsize': outputsize  # 'compact' = 100 latest, 'full' = all
    }
    cache_file = _cache_paths(symbol, outputsize)

    # Try cache first
    cached = _load_from_cache(cache_file, cache_ttl_seconds)
    if cached is not None:
        data = cached.get('Time Series (Daily)', {})
    else:
        # Perform request with retry/backoff
        backoff = 1.5
        for attempt in range(1, max_retries + 1):
            r = requests.get(url, params=params, timeout=15)
            try:
                payload = r.json()
            except Exception:
                payload = {}

            data = payload.get('Time Series (Daily)', {})
            if data:
                _save_to_cache(cache_file, payload)
                break
            # API limit or error; backoff
            if attempt < max_retries:
                sleep_s = backoff ** attempt
                time.sleep(sleep_s)
            else:
                # last attempt failed
                if cached is not None:
                    # Use stale cache as fallback
                    data = cached.get('Time Series (Daily)', {})
                else:
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
    _, _, scaler = preprocess_data(np.array(data), WINDOW_SIZE)

    recent_prices = np.array(data[-WINDOW_SIZE:])
    prices_scaled = scaler.transform(recent_prices.reshape(-1, 1))
    X_pred = np.array([prices_scaled])

    # Resolve model path priority: per-symbol -> multi-stock -> default
    symbol_model_path = f"models/lstm_{symbol}.h5"
    multi_model_path = "models/lstm_multi.h5"
    candidate_paths = [symbol_model_path, multi_model_path, MODEL_PATH]
    selected_path = next((p for p in candidate_paths if os.path.exists(p)), None)
    if not selected_path:
        raise FileNotFoundError(
            f"No trained model found. Looked for: {candidate_paths}. "
            "Train a model first (e.g., run models/run_multi_training.py)."
        )

    model = load_trained_model(selected_path)

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

