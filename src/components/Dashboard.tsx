import React from 'react';
import { TrendingUp, TrendingDown, Target, Zap } from 'lucide-react';

interface DashboardProps {
  selectedStock: string;
  predictions: any[];
  isLoading: boolean;
}

const Dashboard: React.FC<DashboardProps> = ({ selectedStock, predictions, isLoading }) => {
  const currentPrice = predictions.length > 0 ? predictions[0].predicted : 0;
  const futurePrice = predictions.length > 0 ? predictions[predictions.length - 1].predicted : 0;
  const priceChange = futurePrice - currentPrice;
  const percentChange = currentPrice > 0 ? (priceChange / currentPrice) * 100 : 0;
  const avgConfidence = predictions.length > 0 
    ? predictions.reduce((sum, p) => sum + p.confidence, 0) / predictions.length 
    : 0;

  const stats = [
    {
      title: 'Current Price',
      value: `$${currentPrice.toFixed(2)}`,
      icon: Target,
      color: 'text-blue-400',
      bgColor: 'bg-blue-900/20'
    },
    {
      title: '30-Day Prediction',
      value: `$${futurePrice.toFixed(2)}`,
      icon: priceChange >= 0 ? TrendingUp : TrendingDown,
      color: priceChange >= 0 ? 'text-green-400' : 'text-red-400',
      bgColor: priceChange >= 0 ? 'bg-green-900/20' : 'bg-red-900/20'
    },
    {
      title: 'Expected Change',
      value: `${percentChange >= 0 ? '+' : ''}${percentChange.toFixed(1)}%`,
      icon: priceChange >= 0 ? TrendingUp : TrendingDown,
      color: priceChange >= 0 ? 'text-green-400' : 'text-red-400',
      bgColor: priceChange >= 0 ? 'bg-green-900/20' : 'bg-red-900/20'
    },
    {
      title: 'Model Confidence',
      value: `${(avgConfidence * 100).toFixed(1)}%`,
      icon: Zap,
      color: 'text-yellow-400',
      bgColor: 'bg-yellow-900/20'
    }
  ];

  if (isLoading) {
    return (
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {[...Array(4)].map((_, i) => (
          <div key={i} className="bg-slate-800/50 backdrop-blur-sm rounded-xl p-6 border border-slate-700 animate-pulse">
            <div className="h-4 bg-slate-700 rounded mb-2"></div>
            <div className="h-8 bg-slate-700 rounded mb-4"></div>
            <div className="h-4 bg-slate-700 rounded w-3/4"></div>
          </div>
        ))}
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
      {stats.map((stat, index) => (
        <div key={index} className="bg-slate-800/50 backdrop-blur-sm rounded-xl p-6 border border-slate-700 hover:border-slate-600 transition-all duration-300">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-slate-300 text-sm font-medium">{stat.title}</h3>
            <div className={`p-2 rounded-lg ${stat.bgColor}`}>
              <stat.icon className={`h-5 w-5 ${stat.color}`} />
            </div>
          </div>
          <div className={`text-2xl font-bold ${stat.color} mb-2`}>
            {stat.value}
          </div>
          <div className="text-xs text-slate-400">
            {selectedStock} • LSTM Model
          </div>
        </div>
      ))}
    </div>
  );
};

export default Dashboard;