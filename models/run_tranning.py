# run_training.py
import numpy as np
from models.lstm_predictor import fetch_stock_data
from models.preprocess import preprocess_data
from models.train import train_model
from models.utils import save_model
from models.config import WINDOW_SIZE, MODEL_PATH

def main():
    """
    Main function to orchestrate the training of the LSTM model.
    """
    # 1. Fetch the data
    print("Fetching training data for a sample stock (e.g., AAPL)...")
    # Using a common stock like AAPL to train a general model
    # In a real-world scenario, you might want to train on a variety of stocks
    try:
        # Fetch a longer history for training
        data = fetch_stock_data('AAPL', outputsize='full') 
    except Exception as e:
        print(f"Error fetching data: {e}")
        print("Please ensure your API_KEY in models/config.py is set correctly.")
        return

    # 2. Preprocess the data
    print("Preprocessing data...")
    X, y, _ = preprocess_data(np.array(data), WINDOW_SIZE)

    # 3. Train the model
    print("Training the LSTM model...")
    model = train_model(X, y, window_size=WINDOW_SIZE)

    # 4. Save the model
    print(f"Saving the model to {MODEL_PATH}...")
    save_model(model, MODEL_PATH)
    print("Model training complete and saved!")

if __name__ == '__main__':
    main()