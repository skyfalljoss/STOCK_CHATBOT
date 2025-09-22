# models/evaluate.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from .utils import load_trained_model
from .preprocess import preprocess_data
from .lstm_predictor import fetch_stock_data
from .config import WINDOW_SIZE, MODEL_PATH
import os


def calculate_metrics(y_true, y_pred):
    """
    Calculate various regression metrics for model evaluation.
    """
    # Basic metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    # R² may be invalid for <2 samples; guard it
    try:
        r2 = r2_score(y_true, y_pred)
    except Exception:
        r2 = np.nan
    
    # Mean Absolute Percentage Error (MAPE)
    # Avoid division by zero in MAPE
    y_true_safe = np.where(y_true == 0, 1e-8, y_true)
    mape = np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100
    
    # Directional accuracy (for trend prediction)
    if len(y_true) >= 2 and len(y_pred) >= 2:
        direction_true = np.sign(np.diff(y_true))
        direction_pred = np.sign(np.diff(y_pred))
        directional_accuracy = np.mean(direction_true == direction_pred) * 100
    else:
        directional_accuracy = np.nan
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2,
        'MAPE': mape,
        'Directional_Accuracy': directional_accuracy
    }


def evaluate_model_on_test_data(symbol='AAPL', test_size=0.2, random_state=42, model_path=None):
    """
    Evaluate the model on held-out test data.
    """
    # Fetch data
    print(f"Fetching data for {symbol}...")
    data = fetch_stock_data(symbol, outputsize='full')
    
    # Preprocess
    X, y, scaler = preprocess_data(np.array(data), WINDOW_SIZE)
    
    # Split data (maintaining temporal order for time series)
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Load trained model
    path = model_path or MODEL_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"No trained model found at {path}. Train the model first.")
    
    model = load_trained_model(path)
    
    # Make predictions
    y_pred_scaled = model.predict(X_test, verbose=0)
    
    # Inverse transform to get actual prices
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_actual = scaler.inverse_transform(y_pred_scaled).flatten()
    
    # Calculate metrics
    metrics = calculate_metrics(y_test_actual, y_pred_actual)
    
    return {
        'metrics': metrics,
        'y_true': y_test_actual,
        'y_pred': y_pred_actual,
        'symbol': symbol
    }


def cross_validate_temporal(symbol='AAPL', n_splits=5):
    """
    Perform time-series cross-validation (walk-forward validation).
    """
    from .train import build_lstm_model
    
    # Fetch data
    data = fetch_stock_data(symbol, outputsize='full')
    X, y, scaler = preprocess_data(np.array(data), WINDOW_SIZE)
    
    # Calculate split points for temporal CV
    total_samples = len(X)
    test_size = total_samples // n_splits
    
    cv_scores = []
    
    for i in range(n_splits):
        # Define train and test indices
        test_start = (i + 1) * test_size
        test_end = min(test_start + test_size, total_samples)
        
        if test_start >= total_samples:
            break
            
        X_train = X[:test_start]
        y_train = y[:test_start]
        X_test = X[test_start:test_end]
        y_test = y[test_start:test_end]
        
        # Skip folds with too-small test sets (need >= 2 to compute robust metrics)
        if len(X_test) < 2 or len(y_test) < 2:
            print(f"Fold {i+1}: skipped (test set too small: {len(X_test)} samples)")
            continue

        # Train model for this fold
        model = build_lstm_model(WINDOW_SIZE)
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
        
        # Make predictions
        y_pred_scaled = model.predict(X_test, verbose=0)
        
        # Inverse transform
        y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        y_pred_actual = scaler.inverse_transform(y_pred_scaled).flatten()
        
        # Calculate metrics for this fold
        fold_metrics = calculate_metrics(y_test_actual, y_pred_actual)
        cv_scores.append(fold_metrics)
        
        print(f"Fold {i+1}: RMSE = {fold_metrics['RMSE']:.4f}, "
              f"MAPE = {fold_metrics['MAPE']:.2f}%, "
              f"Dir. Acc. = {fold_metrics['Directional_Accuracy']:.1f}%")
    
    # Calculate mean and std of metrics across folds
    if not cv_scores:
        return {
            'mean_metrics': {'RMSE': np.nan, 'MAPE': np.nan, 'Directional_Accuracy': np.nan, 'R²': np.nan},
            'std_metrics': {'RMSE': np.nan, 'MAPE': np.nan, 'Directional_Accuracy': np.nan, 'R²': np.nan},
            'fold_scores': []
        }

    mean_metrics = {}
    std_metrics = {}
    
    for metric in cv_scores[0].keys():
        values = np.array([fold[metric] for fold in cv_scores], dtype=float)
        mean_metrics[metric] = np.nanmean(values)
        std_metrics[metric] = np.nanstd(values)
    
    return {
        'mean_metrics': mean_metrics,
        'std_metrics': std_metrics,
        'fold_scores': cv_scores
    }


def plot_predictions(evaluation_results, save_path=None):
    """
    Plot actual vs predicted prices.
    """
    y_true = evaluation_results['y_true']
    y_pred = evaluation_results['y_pred']
    symbol = evaluation_results['symbol']
    
    plt.figure(figsize=(12, 6))
    
    # Plot actual vs predicted
    plt.subplot(1, 2, 1)
    plt.plot(y_true, label='Actual', alpha=0.7)
    plt.plot(y_pred, label='Predicted', alpha=0.7)
    plt.title(f'{symbol} - Actual vs Predicted Prices')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Scatter plot
    plt.subplot(1, 2, 2)
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title('Actual vs Predicted Scatter Plot')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def evaluate_multiple_stocks(symbols=['AAPL', 'GOOGL', 'MSFT', 'TSLA'], test_size=0.2, model_path=None):
    """
    Evaluate the model on multiple stocks to test generalization.
    """
    results = {}
    
    for symbol in symbols:
        try:
            print(f"\nEvaluating on {symbol}...")
            result = evaluate_model_on_test_data(symbol, test_size, model_path=model_path)
            results[symbol] = result['metrics']
            print(f"{symbol} - RMSE: {result['metrics']['RMSE']:.4f}, "
                  f"MAPE: {result['metrics']['MAPE']:.2f}%, "
                  f"Directional Accuracy: {result['metrics']['Directional_Accuracy']:.1f}%")
        except Exception as e:
            print(f"Error evaluating {symbol}: {e}")
            results[symbol] = None
    
    return results


def generate_evaluation_report(symbol='AAPL', model_path=None):
    """
    Generate a comprehensive evaluation report.
    """
    print("="*60)
    print(f"LSTM MODEL EVALUATION REPORT - {symbol}")
    print("="*60)
    
    # Single stock evaluation
    print("\n1. SINGLE STOCK EVALUATION")
    print("-" * 30)
    
    eval_results = evaluate_model_on_test_data(symbol, model_path=model_path)
    metrics = eval_results['metrics']
    
    print(f"Symbol: {symbol}")
    print(f"Mean Squared Error (MSE): {metrics['MSE']:.6f}")
    print(f"Root Mean Squared Error (RMSE): {metrics['RMSE']:.4f}")
    print(f"Mean Absolute Error (MAE): {metrics['MAE']:.4f}")
    print(f"R-squared (R²): {metrics['R²']:.4f}")
    print(f"Mean Absolute Percentage Error (MAPE): {metrics['MAPE']:.2f}%")
    print(f"Directional Accuracy: {metrics['Directional_Accuracy']:.1f}%")
    
    # Cross-validation
    print(f"\n2. CROSS-VALIDATION RESULTS")
    print("-" * 30)
    
    cv_results = cross_validate_temporal(symbol)
    mean_metrics = cv_results['mean_metrics']
    std_metrics = cv_results['std_metrics']

    
    print(f"RMSE: {mean_metrics['RMSE']:.4f} ± {std_metrics['RMSE']:.4f}")
    print(f"MAPE: {mean_metrics['MAPE']:.2f}% ± {std_metrics['MAPE']:.2f}%")
    print(f"Directional Accuracy: {mean_metrics['Directional_Accuracy']:.1f}% ± {std_metrics['Directional_Accuracy']:.1f}%")
    print(f"R²: {mean_metrics['R²']:.4f} ± {std_metrics['R²']:.4f}")
    

    # Plot results
    print(f"\n3. VISUALIZATION")
    print("-" * 30)
    plot_predictions(eval_results)

    # # Multi-stock evaluation
    # print(f"\n4. MULTI-STOCK GENERALIZATION")
    # print("-" * 30)
    # # default  ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
    # multi_stock_results = evaluate_multiple_stocks(model_path=model_path)
    
    
    
    return {
        'single_stock': eval_results,
        'cross_validation': cv_results,
        # 'multi_stock': multi_stock_results
    }