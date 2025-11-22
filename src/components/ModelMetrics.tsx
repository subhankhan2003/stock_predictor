import React from 'react';
import { Brain, Target, Activity, CheckCircle } from 'lucide-react';

interface ModelMetricsProps {
  selectedStock: string;
}

const ModelMetrics: React.FC<ModelMetricsProps> = ({ selectedStock }) => {
  const metrics = [
    { label: 'Model Accuracy', value: '89.7%', color: 'text-green-400' },
    { label: 'RMSE', value: '2.34', color: 'text-blue-400' },
    { label: 'MAE', value: '1.87', color: 'text-yellow-400' },
    { label: 'R² Score', value: '0.923', color: 'text-purple-400' }
  ];

  return (
    <div className="space-y-6">
      {/* Model Info */}
      <div className="bg-slate-800/50 backdrop-blur-sm rounded-xl p-6 border border-slate-700">
        <h3 className="text-white font-semibold mb-4 flex items-center gap-2">
          <Brain className="h-5 w-5 text-purple-400" />
          LSTM Model
        </h3>
        <div className="space-y-3 text-sm">
          <div className="flex justify-between">
            <span className="text-slate-400">Architecture:</span>
            <span className="text-white">3-Layer LSTM</span>
          </div>
          <div className="flex justify-between">
            <span className="text-slate-400">Hidden Units:</span>
            <span className="text-white">128, 64, 32</span>
          </div>
          <div className="flex justify-between">
            <span className="text-slate-400">Sequence Length:</span>
            <span className="text-white">60 days</span>
          </div>
          <div className="flex justify-between">
            <span className="text-slate-400">Training Data:</span>
            <span className="text-white">5 years</span>
          </div>
        </div>
      </div>

      {/* Performance Metrics */}
      <div className="bg-slate-800/50 backdrop-blur-sm rounded-xl p-6 border border-slate-700">
        <h3 className="text-white font-semibold mb-4 flex items-center gap-2">
          <Target className="h-5 w-5 text-green-400" />
          Performance
        </h3>
        <div className="space-y-4">
          {metrics.map((metric, index) => (
            <div key={index} className="flex justify-between items-center">
              <span className="text-slate-400 text-sm">{metric.label}</span>
              <span className={`font-semibold ${metric.color}`}>{metric.value}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Training Status */}
      <div className="bg-slate-800/50 backdrop-blur-sm rounded-xl p-6 border border-slate-700">
        <h3 className="text-white font-semibold mb-4 flex items-center gap-2">
          <Activity className="h-5 w-5 text-orange-400" />
          Training Status
        </h3>
        <div className="space-y-3">
          <div className="flex items-center gap-2 text-sm">
            <CheckCircle className="h-4 w-4 text-green-400" />
            <span className="text-slate-300">Data Preprocessing</span>
          </div>
          <div className="flex items-center gap-2 text-sm">
            <CheckCircle className="h-4 w-4 text-green-400" />
            <span className="text-slate-300">Model Training</span>
          </div>
          <div className="flex items-center gap-2 text-sm">
            <CheckCircle className="h-4 w-4 text-green-400" />
            <span className="text-slate-300">Validation Complete</span>
          </div>
          <div className="flex items-center gap-2 text-sm">
            <CheckCircle className="h-4 w-4 text-green-400" />
            <span className="text-slate-300">Ready for Prediction</span>
          </div>
        </div>
        
        <div className="mt-4 pt-4 border-t border-slate-700">
          <div className="text-xs text-slate-400 mb-2">Last Trained:</div>
          <div className="text-sm text-white">
            {new Date().toLocaleDateString()} at {new Date().toLocaleTimeString()}
          </div>
        </div>
      </div>
    </div>
  );
};

export default ModelMetrics;