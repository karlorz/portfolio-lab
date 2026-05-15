import React from 'react';
import type { BondMomentumSignal } from '../types/live';

interface BondMomentumPanelProps {
  signals: BondMomentumSignal[];
  timestamp?: string;
  ensembleRecommendation?: {
    weight: number;
    confidence: string;
    action: string;
  };
}

const ETF_CONFIG: Record<string, {
  name: string;
  duration: string;
  color: string;
  formationMonths: number;
  description: string;
}> = {
  SHY: {
    name: 'SHY',
    duration: '1-3 Year Treasury',
    color: '#10b981',
    formationMonths: 12,
    description: 'Short duration - strong momentum effect'
  },
  IEF: {
    name: 'IEF',
    duration: '7-10 Year Treasury',
    color: '#8b5cf6',
    formationMonths: 12,
    description: 'Intermediate - moderate effectiveness'
  },
  TLT: {
    name: 'TLT',
    duration: '20+ Year Treasury',
    color: '#3b82f6',
    formationMonths: 18,
    description: 'Long duration - crisis detection focus'
  },
  BIL: {
    name: 'BIL',
    duration: '1-3 Month T-Bill',
    color: '#f59e0b',
    formationMonths: 12,
    description: 'Ultra short - conservative'
  }
};

export function BondMomentumPanel({ 
  signals, 
  timestamp,
  ensembleRecommendation 
}: BondMomentumPanelProps) {
  if (!signals || signals.length === 0) {
    return (
      <div className="p-4 bg-gray-50 rounded-lg border border-gray-200">
        <h3 className="text-lg font-semibold mb-2">Bond Momentum Overlay (v3.30)</h3>
        <p className="text-gray-500">Loading bond momentum signals...</p>
      </div>
    );
  }

  // Get action color
  const getActionColor = (action: string) => {
    switch (action) {
      case 'increase': return 'text-green-600 bg-green-50';
      case 'hold': return 'text-blue-600 bg-blue-50';
      case 'reduce': return 'text-yellow-600 bg-yellow-50';
      case 'avoid': return 'text-red-600 bg-red-50';
      default: return 'text-gray-600 bg-gray-50';
    }
  };

  // Get confidence badge
  const getConfidenceBadge = (confidence: string) => {
    switch (confidence) {
      case 'strong': return { color: 'bg-green-100 text-green-800', label: 'Strong' };
      case 'moderate': return { color: 'bg-blue-100 text-blue-800', label: 'Moderate' };
      case 'weak': return { color: 'bg-yellow-100 text-yellow-800', label: 'Weak' };
      default: return { color: 'bg-gray-100 text-gray-800', label: confidence };
    }
  };

  // Calculate average signal
  const avgSignal = signals.reduce((sum, s) => sum + s.signal, 0) / signals.length;
  
  // Calculate active signals (non-zero)
  const activeSignals = signals.filter(s => s.signal > 0);
  const activePct = (activeSignals.length / signals.length) * 100;

  return (
    <div className="p-4 bg-white rounded-lg shadow-sm border border-gray-200">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-lg font-semibold">Bond Momentum Overlay</h3>
          <p className="text-xs text-gray-500">v3.30 - TSMOM-Style Fixed Income Signals</p>
        </div>
        {ensembleRecommendation && (
          <div className={`px-3 py-1 rounded-full text-xs font-medium ${
            ensembleRecommendation.weight > 0 
              ? 'bg-blue-100 text-blue-800' 
              : 'bg-gray-100 text-gray-800'
          }`}>
            Ensemble: {ensembleRecommendation.weight}% weight
          </div>
        )}
      </div>

      {/* Summary Stats */}
      <div className="grid grid-cols-3 gap-3 mb-4">
        <div className="p-3 bg-gray-50 rounded-lg border border-gray-200">
          <p className="text-xs text-gray-600 mb-1">Avg Signal</p>
          <p className={`text-xl font-bold ${
            avgSignal > 1 ? 'text-green-600' : 
            avgSignal > 0.5 ? 'text-blue-600' : 'text-gray-600'
          }`}>
            {avgSignal.toFixed(2)}x
          </p>
        </div>
        <div className="p-3 bg-gray-50 rounded-lg border border-gray-200">
          <p className="text-xs text-gray-600 mb-1">Active Signals</p>
          <p className="text-xl font-bold text-gray-900">
            {activePct.toFixed(0)}%
          </p>
          <p className="text-xs text-gray-500">{activeSignals.length}/{signals.length} ETFs</p>
        </div>
        <div className="p-3 bg-gray-50 rounded-lg border border-gray-200">
          <p className="text-xs text-gray-600 mb-1">Last Update</p>
          <p className="text-sm font-medium text-gray-900">
            {timestamp ? new Date(timestamp).toLocaleTimeString() : 'N/A'}
          </p>
        </div>
      </div>

      {/* Individual ETF Signals */}
      <div className="space-y-3">
        <h4 className="text-sm font-medium text-gray-700">Individual ETF Signals</h4>
        {signals.map((signal) => {
          const config = ETF_CONFIG[signal.etf] || ETF_CONFIG['SHY'];
          const confBadge = getConfidenceBadge(signal.confidence);
          
          return (
            <div 
              key={signal.etf}
              className="p-3 bg-gray-50 rounded-lg border border-gray-200"
            >
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  <span 
                    className="w-3 h-3 rounded-sm"
                    style={{ backgroundColor: config.color }}
                  />
                  <span className="font-semibold text-gray-900">{config.name}</span>
                  <span className="text-xs text-gray-500">{config.duration}</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${confBadge.color}`}>
                    {confBadge.label}
                  </span>
                  <span className={`px-2 py-0.5 rounded text-xs font-medium ${getActionColor(signal.action)}`}>
                    {signal.action.toUpperCase()}
                  </span>
                </div>
              </div>
              
              {/* Signal Bar */}
              <div className="mb-2">
                <div className="flex items-center justify-between text-xs text-gray-500 mb-1">
                  <span>Signal Strength</span>
                  <span>{signal.signal.toFixed(3)}x</span>
                </div>
                <div className="relative h-2 bg-gray-200 rounded-full overflow-hidden">
                  <div 
                    className={`absolute h-full transition-all duration-500 ${
                      signal.signal > 1.5 ? 'bg-green-500' :
                      signal.signal > 0.5 ? 'bg-blue-500' :
                      signal.signal > 0 ? 'bg-yellow-500' : 'bg-gray-400'
                    }`}
                    style={{ width: `${Math.min((signal.signal / 2) * 100, 100)}%` }}
                  />
                </div>
              </div>

              {/* Metrics Row */}
              <div className="grid grid-cols-3 gap-2 text-xs">
                <div className="text-gray-600">
                  <span className="text-gray-400">Formation: </span>
                  <span className={signal.formation_return > 0 ? 'text-green-600' : 'text-red-600'}>
                    {signal.formation_return > 0 ? '+' : ''}{signal.formation_return.toFixed(2)}%
                  </span>
                </div>
                <div className="text-gray-600">
                  <span className="text-gray-400">Vol: </span>
                  <span>{(signal.realized_vol * 100).toFixed(1)}%</span>
                </div>
                <div className="text-gray-600">
                  <span className="text-gray-400">Position: </span>
                  <span>{signal.position_size.toFixed(2)}x</span>
                </div>
              </div>
              
              {/* Formation Period Note */}
              <div className="mt-2 text-xs text-gray-500">
                {config.formationMonths}-month formation, {config.description.toLowerCase()}
              </div>
            </div>
          );
        })}
      </div>

      {/* Ensemble Recommendation */}
      {ensembleRecommendation && (
        <div className="mt-4 p-3 bg-blue-50 rounded-lg border border-blue-200">
          <h4 className="text-sm font-medium text-blue-900 mb-2">Ensemble Integration</h4>
          <div className="grid grid-cols-2 gap-2 text-sm">
            <div>
              <span className="text-blue-700">Weight: </span>
              <span className="font-semibold text-blue-900">{ensembleRecommendation.weight}%</span>
            </div>
            <div>
              <span className="text-blue-700">Confidence: </span>
              <span className="font-semibold text-blue-900 capitalize">{ensembleRecommendation.confidence}</span>
            </div>
          </div>
          <p className="mt-2 text-xs text-blue-700">
            Recommended action: <span className="font-semibold uppercase">{ensembleRecommendation.action}</span>
          </p>
        </div>
      )}

      {/* Research Note */}
      <div className="mt-4 pt-4 border-t border-gray-200">
        <p className="text-xs text-gray-500">
          <strong>TSMOM-Style Bond Momentum:</strong> Uses time-series momentum with 
          volatility-scaled position sizing. Long-only constraint appropriate for fixed income. 
          Research shows stronger momentum effects in short-duration bonds (SHY) vs long-duration (TLT).
        </p>
      </div>
    </div>
  );
}
