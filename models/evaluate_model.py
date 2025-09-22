# evaluate_model.py
"""
Standalone script to evaluate your trained LSTM model.
Run this after training your model.
"""

from models.evaluate import (
    generate_evaluation_report,
    evaluate_model_on_test_data,
    cross_validate_temporal,
    evaluate_multiple_stocks,
    plot_predictions
)

def main():
    """
    Run comprehensive model evaluation.
    """
    # Choose the stock symbol to evaluate
    symbol = 'AAPL'  # Change this to evaluate different stocks
    
    print("Starting model evaluation...")
    
    try:
        # Generate full evaluation report
        results = generate_evaluation_report(symbol)
        
        # Additional evaluations you can run:
        
        # 1. Evaluate on a different stock
        print(f"\nEvaluating model generalization on different stocks...")
        multi_results = evaluate_multiple_stocks(['AAPL', 'GOOGL', 'MSFT', 'TSLA'])
        
        # 2. Quick single evaluation with plotting
        # eval_results = evaluate_model_on_test_data('GOOGL')
        # plot_predictions(eval_results, save_path='predictions_plot.png')
        
        print("\nEvaluation complete!")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        print("Make sure you have:")
        print("1. Trained and saved a model")
        print("2. Valid API key for Alpha Vantage")
        print("3. Internet connection for fetching data")

if __name__ == '__main__':
    main()