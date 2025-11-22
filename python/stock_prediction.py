"""
AI Stock Price Prediction System
University AI Project

This is the complete Python implementation for stock price prediction using LSTM.
To run this, you'll need to install the required packages in your local Python environment:

pip install yfinance pandas numpy tensorflow scikit-learn matplotlib seaborn

Author: [Your Name]
Course: Artificial Intelligence
"""

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class StockPredictor:
    def __init__(self, symbol, sequence_length=60):
        """
        Initialize the Stock Predictor
        
        Args:
            symbol (str): Stock symbol (e.g., 'AAPL', 'TSLA')
            sequence_length (int): Number of days to look back for prediction
        """
        self.symbol = symbol
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.data = None
        self.scaled_data = None
        
    def fetch_data(self, period="5y"):
        """
        Fetch historical stock data from Yahoo Finance
        
        Args:
            period (str): Period to fetch data for ('1y', '2y', '5y', 'max')
        """
        print(f"Fetching data for {self.symbol}...")
        
        try:
            stock = yf.Ticker(self.symbol)
            self.data = stock.history(period=period)
            
            if self.data.empty:
                raise ValueError(f"No data found for symbol {self.symbol}")
            
            print(f"Successfully fetched {len(self.data)} days of data")
            return self.data
            
        except Exception as e:
            print(f"Error fetching data: {str(e)}")
            return None
    
    def preprocess_data(self):
        """
        Preprocess the stock data for LSTM training
        """
        if self.data is None:
            raise ValueError("No data available. Please fetch data first.")
        
        # Use closing prices for prediction
        close_prices = self.data['Close'].values.reshape(-1, 1)
        
        # Scale the data
        self.scaled_data = self.scaler.fit_transform(close_prices)
        
        print(f"Data preprocessed. Shape: {self.scaled_data.shape}")
        return self.scaled_data
    
    def create_sequences(self, data, test_size=0.2):
        """
        Create sequences for LSTM training
        
        Args:
            data: Scaled price data
            test_size (float): Proportion of data to use for testing
        """
        X, y = [], []
        
        for i in range(self.sequence_length, len(data)):
            X.append(data[i-self.sequence_length:i, 0])
            y.append(data[i, 0])
        
        X, y = np.array(X), np.array(y)
        
        # Split into training and testing sets
        split_index = int(len(X) * (1 - test_size))
        
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]
        
        # Reshape for LSTM input
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        
        print(f"Training sequences: {X_train.shape}")
        print(f"Testing sequences: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def build_model(self):
        """
        Build and compile the LSTM model
        """
        model = Sequential([
            # First LSTM layer with return sequences
            LSTM(128, return_sequences=True, input_shape=(self.sequence_length, 1)),
            Dropout(0.2),
            
            # Second LSTM layer
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            
            # Third LSTM layer
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            
            # Dense output layer
            Dense(1)
        ])
        
        # Compile the model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mean_squared_error',
            metrics=['mae']
        )
        
        self.model = model
        print("LSTM model built successfully")
        print(model.summary())
        
        return model
    
    def train_model(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        """
        Train the LSTM model
        """
        if self.model is None:
            raise ValueError("Model not built. Please build model first.")
        
        print("Training LSTM model...")
        
        # Define callbacks
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        )
        
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-7
        )
        
        # Train the model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        print("Model training completed")
        return history
    
    def evaluate_model(self, X_test, y_test):
        """
        Evaluate the trained model
        """
        if self.model is None:
            raise ValueError("Model not trained yet.")
        
        # Make predictions
        predictions = self.model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        # Calculate accuracy (custom metric for price prediction)
        accuracy = 100 - (rmse * 100)
        
        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2,
            'Accuracy': f"{accuracy:.1f}%"
        }
        
        print("\n=== Model Performance Metrics ===")
        for metric, value in metrics.items():
            if isinstance(value, float):
                print(f"{metric}: {value:.4f}")
            else:
                print(f"{metric}: {value}")
        
        return metrics, predictions
    
    def predict_future(self, days=30):
        """
        Predict future stock prices
        
        Args:
            days (int): Number of days to predict into the future
        """
        if self.model is None:
            raise ValueError("Model not trained yet.")
        
        # Get the last sequence_length days of data
        last_sequence = self.scaled_data[-self.sequence_length:]
        predictions = []
        
        current_sequence = last_sequence.copy()
        
        for _ in range(days):
            # Reshape for prediction
            X_pred = current_sequence.reshape(1, self.sequence_length, 1)
            
            # Make prediction
            next_pred = self.model.predict(X_pred, verbose=0)
            predictions.append(next_pred[0, 0])
            
            # Update sequence by removing first element and adding prediction
            current_sequence = np.roll(current_sequence, -1)
            current_sequence[-1] = next_pred[0, 0]
        
        # Inverse transform predictions to original scale
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = self.scaler.inverse_transform(predictions)
        
        # Create future dates
        last_date = self.data.index[-1]
        future_dates = [last_date + timedelta(days=i+1) for i in range(days)]
        
        # Create prediction DataFrame
        future_predictions = pd.DataFrame({
            'Date': future_dates,
            'Predicted_Price': predictions.flatten()
        })
        
        print(f"\n=== {days}-Day Price Predictions for {self.symbol} ===")
        for _, row in future_predictions.head(10).iterrows():
            print(f"{row['Date'].strftime('%Y-%m-%d')}: ${row['Predicted_Price']:.2f}")
        
        if days > 10:
            print("...")
            print(f"{future_predictions.iloc[-1]['Date'].strftime('%Y-%m-%d')}: ${future_predictions.iloc[-1]['Predicted_Price']:.2f}")
        
        return future_predictions
    
    def plot_results(self, X_test, y_test, predictions, future_predictions=None):
        """
        Plot the prediction results
        """
        # Convert scaled values back to original scale
        y_test_orig = self.scaler.inverse_transform(y_test.reshape(-1, 1))
        predictions_orig = self.scaler.inverse_transform(predictions.reshape(-1, 1))
        
        # Create test dates
        test_start_date = self.data.index[-(len(y_test)):]
        
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Historical data with predictions
        plt.subplot(2, 2, 1)
        plt.plot(self.data.index[-200:], self.data['Close'][-200:], 'b-', label='Historical Prices', alpha=0.7)
        plt.plot(test_start_date, y_test_orig, 'g-', label='Actual Prices', linewidth=2)
        plt.plot(test_start_date, predictions_orig, 'r--', label='Predicted Prices', linewidth=2)
        plt.title(f'{self.symbol} - Historical vs Predicted Prices')
        plt.xlabel('Date')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Future predictions
        if future_predictions is not None:
            plt.subplot(2, 2, 2)
            recent_data = self.data['Close'][-60:]
            plt.plot(recent_data.index, recent_data.values, 'b-', label='Recent Historical', linewidth=2)
            plt.plot(future_predictions['Date'], future_predictions['Predicted_Price'], 'r-', label='Future Predictions', linewidth=2, marker='o', markersize=3)
            plt.title(f'{self.symbol} - Future Price Predictions')
            plt.xlabel('Date')
            plt.ylabel('Price ($)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
        
        # Plot 3: Prediction accuracy
        plt.subplot(2, 2, 3)
        plt.scatter(y_test_orig, predictions_orig, alpha=0.5)
        plt.plot([y_test_orig.min(), y_test_orig.max()], [y_test_orig.min(), y_test_orig.max()], 'r--', lw=2)
        plt.xlabel('Actual Prices ($)')
        plt.ylabel('Predicted Prices ($)')
        plt.title('Prediction Accuracy')
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Price distribution
        plt.subplot(2, 2, 4)
        plt.hist(self.data['Close'], bins=50, alpha=0.7, label='Historical Distribution')
        if future_predictions is not None:
            plt.hist(future_predictions['Predicted_Price'], bins=20, alpha=0.7, label='Predicted Distribution')
        plt.xlabel('Price ($)')
        plt.ylabel('Frequency')
        plt.title('Price Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Save the plot
        plt.savefig(f'{self.symbol}_stock_prediction_results.png', dpi=300, bbox_inches='tight')
        print(f"\nPlot saved as '{self.symbol}_stock_prediction_results.png'")

def main():
    """
    Main function to run the stock prediction system
    """
    # Configuration
    STOCK_SYMBOL = 'AAPL'  # Change this to any stock symbol
    SEQUENCE_LENGTH = 60
    PREDICTION_DAYS = 30
    
    print("=== AI Stock Price Prediction System ===")
    print(f"Analyzing: {STOCK_SYMBOL}")
    print(f"Prediction horizon: {PREDICTION_DAYS} days\n")
    
    # Initialize the predictor
    predictor = StockPredictor(STOCK_SYMBOL, SEQUENCE_LENGTH)
    
    # Step 1: Fetch data
    data = predictor.fetch_data(period="5y")
    if data is None:
        return
    
    # Step 2: Preprocess data
    scaled_data = predictor.preprocess_data()
    
    # Step 3: Create sequences
    X_train, X_test, y_train, y_test = predictor.create_sequences(scaled_data)
    
    # Step 4: Build model
    model = predictor.build_model()
    
    # Step 5: Train model
    history = predictor.train_model(X_train, y_train, X_test, y_test, epochs=100)
    
    # Step 6: Evaluate model
    metrics, predictions = predictor.evaluate_model(X_test, y_test)
    
    # Step 7: Predict future prices
    future_predictions = predictor.predict_future(days=PREDICTION_DAYS)
    
    # Step 8: Plot results
    predictor.plot_results(X_test, y_test, predictions, future_predictions)
    
    # Step 9: Save results
    future_predictions.to_csv(f'{STOCK_SYMBOL}_future_predictions.csv', index=False)
    print(f"\nPredictions saved to '{STOCK_SYMBOL}_future_predictions.csv'")
    
    print("\n=== Analysis Complete ===")
    print("Key Findings:")
    print(f"• Model achieved {metrics['Accuracy']} accuracy")
    print(f"• RMSE: {metrics['RMSE']:.4f}")
    print(f"• R² Score: {metrics['R²']:.4f}")
    
    current_price = data['Close'][-1]
    future_price = future_predictions['Predicted_Price'].iloc[-1]
    price_change = ((future_price - current_price) / current_price) * 100
    
    print(f"• Current Price: ${current_price:.2f}")
    print(f"• {PREDICTION_DAYS}-Day Prediction: ${future_price:.2f}")
    print(f"• Expected Change: {price_change:+.1f}%")

if __name__ == "__main__":
    main()