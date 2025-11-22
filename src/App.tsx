import React, { useState, useEffect } from 'react';
import Dashboard from './components/Dashboard';
import StockSelector from './components/StockSelector';
import PredictionChart from './components/PredictionChart';
import ModelMetrics from './components/ModelMetrics';
import { TrendingUp, Brain, BarChart3, Download } from 'lucide-react';

function App() {
  const [selectedStock, setSelectedStock] = useState('AAPL');
  const [isLoading, setIsLoading] = useState(false);
  const [predictions, setPredictions] = useState([]);

  const handleStockSelect = (stock: string) => {
    setSelectedStock(stock);
    setIsLoading(true);
    // Simulate API call delay
    setTimeout(() => {
      setIsLoading(false);
      // Generate mock prediction data
      generateMockPredictions();
    }, 2000);
  };

  const generateMockPredictions = () => {
    const mockData = [];
    const basePrice = 150 + Math.random() * 100;
    
    for (let i = 0; i < 30; i++) {
      const date = new Date();
      date.setDate(date.getDate() + i);
      
      const variance = (Math.random() - 0.5) * 10;
      const trend = Math.sin(i * 0.1) * 5;
      const price = Math.max(basePrice + trend + variance + (i * 0.5), 10);
      
      mockData.push({
        date: date.toISOString().split('T')[0],
        predicted: Math.round(price * 100) / 100,
        confidence: Math.max(0.7, Math.random()),
        actual: i < 15 ? Math.round((price + (Math.random() - 0.5) * 5) * 100) / 100 : null
      });
    }
    
    setPredictions(mockData);
  };

  useEffect(() => {
    generateMockPredictions();
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <header className="text-center mb-12">
          <div className="flex items-center justify-center gap-3 mb-4">
            <Brain className="h-10 w-10 text-blue-400" />
            <h1 className="text-4xl font-bold text-white">AI Stock Prediction System</h1>
          </div>
          <p className="text-xl text-blue-200 max-w-2xl mx-auto">
            Advanced LSTM Neural Network for 30-Day Stock Price Forecasting
          </p>
        </header>

        {/* Main Content */}
        <div className="grid lg:grid-cols-4 gap-8">
          {/* Sidebar */}
          <div className="lg:col-span-1 space-y-6">
            <StockSelector 
              selectedStock={selectedStock}
              onStockSelect={handleStockSelect}
              isLoading={isLoading}
            />
            <ModelMetrics selectedStock={selectedStock} />
          </div>

          {/* Main Dashboard */}
          <div className="lg:col-span-3 space-y-6">
            <Dashboard 
              selectedStock={selectedStock}
              predictions={predictions}
              isLoading={isLoading}
            />
            
            <PredictionChart 
              data={predictions}
              selectedStock={selectedStock}
              isLoading={isLoading}
            />

            {/* Action Buttons */}
            <div className="flex flex-wrap gap-4 justify-center">
              <button className="flex items-center gap-2 bg-blue-600 hover:bg-blue-700 text-white px-6 py-3 rounded-lg transition-colors">
                <TrendingUp className="h-5 w-5" />
                Run Prediction
              </button>
              <button className="flex items-center gap-2 bg-green-600 hover:bg-green-700 text-white px-6 py-3 rounded-lg transition-colors">
                <BarChart3 className="h-5 w-5" />
                Analyze Trends
              </button>
              <button className="flex items-center gap-2 bg-purple-600 hover:bg-purple-700 text-white px-6 py-3 rounded-lg transition-colors">
                <Download className="h-5 w-5" />
                Export Report
              </button>
            </div>
          </div>
        </div>

        {/* Footer */}
        <footer className="mt-16 text-center text-blue-300 border-t border-blue-800 pt-8">
          <p className="mb-2">
            <strong>Technologies:</strong> Python, TensorFlow/Keras, LSTM, Yahoo Finance API, Pandas, Matplotlib
          </p>
          <p className="text-sm opacity-75">
            University AI Project - Stock Price Prediction using Deep Learning
          </p>
        </footer>
      </div>
    </div>
  );
}

export default App;