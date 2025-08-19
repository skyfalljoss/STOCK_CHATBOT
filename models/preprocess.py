import numpy as np
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(prices, window_size=60):
    scaler = MinMaxScaler(feature_range=(0,1))
    prices_scaled = scaler.fit_transform(prices.reshape(-1, 1))
    X, y = [], []
    for i in range(window_size, len(prices_scaled)):
        X.append(prices_scaled[i-window_size:i, 0])
        y.append(prices_scaled[i, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape(X.shape[0], X.shape, 1)
    return X, y, scaler
