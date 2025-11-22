import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Area, AreaChart } from 'recharts';
import { BarChart3, Loader } from 'lucide-react';

interface PredictionChartProps {
  data: any[];
  selectedStock: string;
  isLoading: boolean;
}

const PredictionChart: React.FC<PredictionChartProps> = ({ data, selectedStock, isLoading }) => {
  if (isLoading) {
    return (
      <div className="bg-slate-800/50 backdrop-blur-sm rounded-xl p-6 border border-slate-700">
        <div className="flex items-center justify-center h-96">
          <div className="text-center">
            <Loader className="h-12 w-12 text-blue-400 animate-spin mx-auto mb-4" />
            <p className="text-slate-400">Training LSTM model for {selectedStock}...</p>
            <p className="text-sm text-slate-500 mt-2">Analyzing historical patterns and market trends</p>
          </div>
        </div>
      </div>
    );
  }

  const formatTooltip = (value: any, name: string) => {
    if (name === 'predicted') return [`$${value.toFixed(2)}`, 'Predicted Price'];
    if (name === 'actual') return [`$${value.toFixed(2)}`, 'Actual Price'];
    if (name === 'confidence') return [`${(value * 100).toFixed(1)}%`, 'Confidence'];
    return [value, name];
  };

  return (
    <div className="bg-slate-800/50 backdrop-blur-sm rounded-xl p-6 border border-slate-700">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-white font-semibold text-xl flex items-center gap-2">
          <BarChart3 className="h-6 w-6 text-blue-400" />
          30-Day Price Prediction - {selectedStock}
        </h3>
        <div className="flex items-center gap-4 text-sm">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-blue-400 rounded-full"></div>
            <span className="text-slate-300">Predicted</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-green-400 rounded-full"></div>
            <span className="text-slate-300">Actual</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-yellow-400 rounded-full"></div>
            <span className="text-slate-300">Confidence</span>
          </div>
        </div>
      </div>

      <div className="h-96">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis 
              dataKey="date" 
              stroke="#9CA3AF"
              fontSize={12}
              tickFormatter={(value) => new Date(value).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}
            />
            <YAxis 
              stroke="#9CA3AF"
              fontSize={12}
              tickFormatter={(value) => `$${value.toFixed(0)}`}
            />
            <Tooltip 
              contentStyle={{ 
                backgroundColor: '#1F2937', 
                border: '1px solid #374151',
                borderRadius: '8px',
                color: '#F9FAFB'
              }}
              formatter={formatTooltip}
              labelFormatter={(value) => `Date: ${new Date(value).toLocaleDateString()}`}
            />
            <Legend />
            <Line 
              type="monotone" 
              dataKey="predicted" 
              stroke="#60A5FA" 
              strokeWidth={3}
              dot={{ r: 4, fill: '#60A5FA' }}
              name="Predicted Price"
            />
            <Line 
              type="monotone" 
              dataKey="actual" 
              stroke="#34D399" 
              strokeWidth={2}
              dot={{ r: 3, fill: '#34D399' }}
              connectNulls={false}
              name="Actual Price"
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Confidence Chart */}
      <div className="mt-8 h-32">
        <h4 className="text-slate-300 font-medium mb-4">Model Confidence Level</h4>
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis 
              dataKey="date" 
              stroke="#9CA3AF"
              fontSize={10}
              tickFormatter={(value) => new Date(value).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}
            />
            <YAxis 
              stroke="#9CA3AF"
              fontSize={10}
              tickFormatter={(value) => `${(value * 100).toFixed(0)}%`}
            />
            <Tooltip 
              contentStyle={{ 
                backgroundColor: '#1F2937', 
                border: '1px solid #374151',
                borderRadius: '8px',
                color: '#F9FAFB'
              }}
              formatter={formatTooltip}
            />
            <Area 
              type="monotone" 
              dataKey="confidence" 
              stroke="#FBBF24" 
              fill="#FBBF24"
              fillOpacity={0.3}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default PredictionChart;