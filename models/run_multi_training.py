# models/run_multi_training.py
import numpy as np
from models.lstm_predictor import fetch_stock_data
from models.preprocess import preprocess_data
from models.train import train_model
from models.utils import save_model
from models.config import WINDOW_SIZE, EPOCHS, BATCH_SIZE
from models.evaluate import evaluate_multiple_stocks


MODEL_PATH="models/lstm_multi.h5"

def train_general_model(symbols, model_path="models/lstm_multi.h5", eval_symbols=None):
    X_all, y_all = [], []
    for sym in symbols:
        print(f"Fetching {sym}...")
        data = fetch_stock_data(sym, outputsize='full')
        X_sym, y_sym, _ = preprocess_data(np.array(data), WINDOW_SIZE)
        X_all.append(X_sym)
        y_all.append(y_sym)


    X = np.concatenate(X_all, axis=0)
    y = np.concatenate(y_all, axis=0)



    print(f"Training on {X.shape[0]} sequences from {len(symbols)} symbols...")
    model, history = train_model(X, y, window_size=WINDOW_SIZE,epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2)
    save_model(model, model_path)
    print(f"Saved general model to {model_path}")

    # Optional: evaluate immediately on single and multiple stocks
    if eval_symbols is None:
        eval_symbols = [symbols[0]]

    # print("\nEvaluating the trained model...")
    # try:
    #     # Single report on first eval symbol
    #     _ = generate_evaluation_report(symbol=eval_symbols[0], model_path=model_path)
    # except Exception as e:
    #     print(f"Single evaluation error: {e}")

    try:
        # Multi-stock quick metrics
        metrics = evaluate_multiple_stocks(symbols=eval_symbols, model_path=model_path)
        print("\nMulti-stock evaluation summary:")
        for sym, m in metrics.items():
            if m:
                print(f"{sym}: RMSE={m['RMSE']:.4f}, MAPE={m['MAPE']:.2f}%, DirAcc={m['Directional_Accuracy']:.1f}%")
            else:
                print(f"{sym}: evaluation failed")
    except Exception as e:
        print(f"Multi evaluation error: {e}")

    return model, history

if __name__ == "__main__":
    symbols = ["AAPL","MSFT","GOOGL","AMZN","TSLA"]
    # choose a subset for evaluation to reduce time if desired
    eval_symbols = ["AAPL","MSFT","GOOGL"]
    train_general_model(symbols, model_path="models/lstm_multi.h5", eval_symbols=eval_symbols)