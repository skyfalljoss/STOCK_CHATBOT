# models/run_training.py
import numpy as np
from models.lstm_predictor import fetch_stock_data
from models.preprocess import preprocess_data
from models.train import train_model
from models.utils import save_model
from models.config import WINDOW_SIZE, MODEL_PATH, EPOCHS, BATCH_SIZE
from models.evaluate import generate_evaluation_report

import matplotlib.pyplot as plt

def main():
    """
    Main function to orchestrate the training of the LSTM model.
    """
    # 1. Fetch the data
    print("Fetching training data for a sample stock (e.g., AAPL)...")
    # Using a common stock like AAPL to train a general model
    symbol = 'AAPL'
    try:
        # Fetch a longer history for training
        data = fetch_stock_data(symbol, outputsize='full') 
    except Exception as e:
        print(f"Error fetching data: {e}")
        return

    # 2. Preprocess the data
    print("Preprocessing data...")
    X, y, _ = preprocess_data(np.array(data), WINDOW_SIZE)

    # 3. Train the model
    print("Training the LSTM model...")
    model, history = train_model(X, y, window_size=WINDOW_SIZE,epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2)

    # 4. Save the model
    print(f"Saving the model to {MODEL_PATH}...")
    save_model(model, MODEL_PATH)
    print("Model training complete and saved!")

    # 5. Evaluate the model# 5. Evaluate the model
    print("\nEvaluating model performance...")
    evaluation_results = generate_evaluation_report(symbol)

    # 6. Plot the training history
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss During Training')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return evaluation_results

if __name__ == '__main__':
    main()