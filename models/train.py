# models/train.py
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split


def build_lstm_model(window_size, units=128, dropout=0.2):
    model = Sequential([
        LSTM(units=units, return_sequences=True, input_shape=(window_size, 1)),
        Dropout(dropout),
        LSTM(units=units),
        Dropout(dropout),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(X, y, window_size, epochs=20, batch_size=32, validation_split=0.2):
    model = build_lstm_model(window_size)
    # model.fit(X, y, epochs=epochs, batch_size=batch_size)
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
    ]
    history = model.fit(
        X, y, 
        epochs=epochs, 
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=callbacks,
        verbose=1
    )
    return model,history
