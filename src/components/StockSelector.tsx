import React from 'react';
import { Search, Loader, TrendingUp } from 'lucide-react';

interface StockSelectorProps {
  selectedStock: string;
  onStockSelect: (stock: string) => void;
  isLoading: boolean;
}

const StockSelector: React.FC<StockSelectorProps> = ({ selectedStock, onStockSelect, isLoading }) => {
  const popularStocks = [
    { symbol: 'AAPL', name: 'Apple Inc.', price: 175.43, change: 2.34 },
    { symbol: 'TSLA', name: 'Tesla Inc.', price: 242.68, change: -3.21 },
    { symbol: 'GOOGL', name: 'Alphabet Inc.', price: 138.21, change: 1.87 },
    { symbol: 'MSFT', name: 'Microsoft Corp.', price: 378.85, change: 4.56 },
    { symbol: 'AMZN', name: 'Amazon.com Inc.', price: 127.74, change: -1.23 },
    { symbol: 'NVDA', name: 'NVIDIA Corp.', price: 418.32, change: 8.91 },
    { symbol: 'META', name: 'Meta Platforms', price: 296.73, change: 2.18 },
    { symbol: 'NFLX', name: 'Netflix Inc.', price: 378.45, change: -2.67 }
  ];

  return (
    <div className="space-y-6">
      {/* Search */}
      <div className="bg-slate-800/50 backdrop-blur-sm rounded-xl p-6 border border-slate-700">
        <h3 className="text-white font-semibold mb-4 flex items-center gap-2">
          <Search className="h-5 w-5 text-blue-400" />
          Stock Selection
        </h3>
        <div className="relative">
          <input
            type="text"
            placeholder="Search stocks (e.g., AAPL)"
            className="w-full bg-slate-700 text-white rounded-lg px-4 py-3 pl-10 border border-slate-600 focus:border-blue-500 focus:outline-none transition-colors"
          />
          <Search className="absolute left-3 top-3.5 h-4 w-4 text-slate-400" />
        </div>
      </div>

      {/* Popular Stocks */}
      <div className="bg-slate-800/50 backdrop-blur-sm rounded-xl p-6 border border-slate-700">
        <h3 className="text-white font-semibold mb-4 flex items-center gap-2">
          <TrendingUp className="h-5 w-5 text-green-400" />
          Popular Stocks
        </h3>
        <div className="space-y-2">
          {popularStocks.map((stock) => (
            <button
              key={stock.symbol}
              onClick={() => onStockSelect(stock.symbol)}
              disabled={isLoading}
              className={`w-full p-3 rounded-lg text-left transition-all duration-200 ${
                selectedStock === stock.symbol
                  ? 'bg-blue-600 text-white'
                  : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
              } ${isLoading ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}`}
            >
              <div className="flex justify-between items-center">
                <div>
                  <div className="font-semibold">{stock.symbol}</div>
                  <div className="text-xs opacity-75 truncate">{stock.name}</div>
                </div>
                <div className="text-right">
                  <div className="font-semibold">${stock.price}</div>
                  <div className={`text-xs ${stock.change >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                    {stock.change >= 0 ? '+' : ''}{stock.change.toFixed(2)}
                  </div>
                </div>
              </div>
              {isLoading && selectedStock === stock.symbol && (
                <div className="mt-2 flex items-center gap-2 text-blue-200">
                  <Loader className="h-3 w-3 animate-spin" />
                  <span className="text-xs">Analyzing...</span>
                </div>
              )}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};

export default StockSelector;