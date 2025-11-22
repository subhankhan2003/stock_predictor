"""
Example usage of the Stock Prediction System
This file demonstrates different ways to use the StockPredictor class
"""

from stock_prediction import StockPredictor
import pandas as pd

def example_single_stock():
    """Example: Analyze a single stock"""
    print("=== Single Stock Analysis Example ===")
    
    predictor = StockPredictor('AAPL', sequence_length=60)
    
    # Fetch and preprocess data
    data = predictor.fetch_data(period="3y")
    predictor.preprocess_data()
    
    # Create sequences and train model
    X_train, X_test, y_train, y_test = predictor.create_sequences(predictor.scaled_data)
    predictor.build_model()
    
    # Train with fewer epochs for quick example
    history = predictor.train_model(X_train, y_train, X_test, y_test, epochs=50)
    
    # Evaluate and predict
    metrics, predictions = predictor.evaluate_model(X_test, y_test)
    future_predictions = predictor.predict_future(days=15)
    
    return future_predictions

def example_multiple_stocks():
    """Example: Compare multiple stocks"""
    print("=== Multiple Stocks Comparison Example ===")
    
    stocks = ['AAPL', 'TSLA', 'GOOGL', 'MSFT']
    results = {}
    
    for stock in stocks:
        print(f"\nAnalyzing {stock}...")
        predictor = StockPredictor(stock, sequence_length=30)  # Shorter sequence for speed
        
        # Quick analysis
        data = predictor.fetch_data(period="2y")
        if data is not None:
            predictor.preprocess_data()
            X_train, X_test, y_train, y_test = predictor.create_sequences(predictor.scaled_data)
            predictor.build_model()
            
            # Quick training
            history = predictor.train_model(X_train, y_train, X_test, y_test, epochs=30)
            metrics, predictions = predictor.evaluate_model(X_test, y_test)
            future_predictions = predictor.predict_future(days=7)
            
            # Store results
            current_price = data['Close'][-1]
            predicted_price = future_predictions['Predicted_Price'].iloc[-1]
            change_percent = ((predicted_price - current_price) / current_price) * 100
            
            results[stock] = {
                'current_price': current_price,
                'predicted_price': predicted_price,
                'change_percent': change_percent,
                'accuracy': metrics['Accuracy']
            }
    
    # Display comparison
    print("\n=== Comparison Results ===")
    comparison_df = pd.DataFrame(results).T
    print(comparison_df.round(2))
    
    return results

def example_different_timeframes():
    """Example: Analyze same stock with different prediction timeframes"""
    print("=== Different Timeframes Example ===")
    
    stock = 'AAPL'
    timeframes = [7, 15, 30]
    
    predictor = StockPredictor(stock)
    data = predictor.fetch_data(period="3y")
    predictor.preprocess_data()
    
    # Train model once
    X_train, X_test, y_train, y_test = predictor.create_sequences(predictor.scaled_data)
    predictor.build_model()
    history = predictor.train_model(X_train, y_train, X_test, y_test, epochs=50)
    
    current_price = data['Close'][-1]
    
    print(f"\nCurrent {stock} price: ${current_price:.2f}")
    print("\nPredictions for different timeframes:")
    
    for days in timeframes:
        future_predictions = predictor.predict_future(days=days)
        predicted_price = future_predictions['Predicted_Price'].iloc[-1]
        change_percent = ((predicted_price - current_price) / current_price) * 100
        
        print(f"{days:2d} days: ${predicted_price:7.2f} ({change_percent:+5.1f}%)")

def example_custom_analysis():
    """Example: Custom analysis with specific parameters"""
    print("=== Custom Analysis Example ===")
    
    # Custom configuration
    predictor = StockPredictor('TSLA', sequence_length=90)  # Longer sequence
    
    # Fetch more data
    data = predictor.fetch_data(period="max")
    predictor.preprocess_data()
    
    # Create sequences with custom test size
    X_train, X_test, y_train, y_test = predictor.create_sequences(
        predictor.scaled_data, test_size=0.15
    )
    
    # Build custom model (you can modify this in the class)
    predictor.build_model()
    
    # Train with custom parameters
    history = predictor.train_model(
        X_train, y_train, X_test, y_test,
        epochs=75,
        batch_size=16
    )
    
    # Comprehensive evaluation
    metrics, predictions = predictor.evaluate_model(X_test, y_test)
    
    # Multiple prediction horizons
    print("\nMultiple prediction horizons:")
    for days in [1, 7, 14, 21, 30]:
        future_pred = predictor.predict_future(days=days)
        last_pred = future_pred['Predicted_Price'].iloc[-1]
        print(f"{days:2d}-day prediction: ${last_pred:.2f}")
    
    return predictor

if __name__ == "__main__":
    print("Stock Prediction System - Example Usage\n")
    
    # Run different examples
    try:
        # Example 1: Single stock analysis
        single_result = example_single_stock()
        print("\n" + "="*50 + "\n")
        
        # Example 2: Multiple stocks comparison
        multi_result = example_multiple_stocks()
        print("\n" + "="*50 + "\n")
        
        # Example 3: Different timeframes
        example_different_timeframes()
        print("\n" + "="*50 + "\n")
        
        # Example 4: Custom analysis
        custom_predictor = example_custom_analysis()
        
        print("\n=== All Examples Completed Successfully ===")
        
    except Exception as e:
        print(f"Error running examples: {str(e)}")
        print("Make sure you have installed all required packages:")
        print("pip install -r requirements.txt")