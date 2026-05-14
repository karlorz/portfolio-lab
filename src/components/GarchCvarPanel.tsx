import React from 'react';
import type { GarchCvarData } from '../types/live';

interface GarchCvarPanelProps {
  data?: GarchCvarData | null;
}

export function GarchCvarPanel({ data }: GarchCvarPanelProps) {
  if (!data) {
    return (
      <div className="p-4 bg-gray-50 rounded-lg">
        <h3 className="text-lg font-semibold mb-2">GARCH-Filtered CVaR</h3>
        <p className="text-gray-500">Loading risk metrics...</p>
      </div>
    );
  }

  const { 
    cvar_95, 
    cvar_95_garch, 
    var_95, 
    var_95_garch,
    cvar_ratio,
    garch_active,
    current_volatility,
    forecast_volatility,
    volatility_clustering 
  } = data;

  // Severity color coding
  const getSeverityColor = (ratio: number) => {
    if (ratio < 1.3) return 'text-green-600';
    if (ratio < 1.5) return 'text-yellow-600';
    if (ratio < 1.8) return 'text-orange-600';
    return 'text-red-600';
  };

  const getSeverityBg = (ratio: number) => {
    if (ratio < 1.3) return 'bg-green-50 border-green-200';
    if (ratio < 1.5) return 'bg-yellow-50 border-yellow-200';
    if (ratio < 1.8) return 'bg-orange-50 border-orange-200';
    return 'bg-red-50 border-red-200';
  };

  const getVolClusterLabel = (clustering: string) => {
    switch (clustering) {
      case 'low': return { label: 'Low', color: 'bg-green-100 text-green-800' };
      case 'normal': return { label: 'Normal', color: 'bg-blue-100 text-blue-800' };
      case 'elevated': return { label: 'Elevated', color: 'bg-yellow-100 text-yellow-800' };
      case 'high': return { label: 'High', color: 'bg-red-100 text-red-800' };
      default: return { label: clustering, color: 'bg-gray-100 text-gray-800' };
    }
  };

  const volCluster = getVolClusterLabel(volatility_clustering || 'normal');

  return (
    <div className="p-4 bg-white rounded-lg shadow-sm border border-gray-200">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold">GARCH-Filtered CVaR</h3>
        <div className="flex items-center gap-2">
          <span className={`px-2 py-1 rounded-full text-xs font-medium ${volCluster.color}`}>
            Vol Clustering: {volCluster.label}
          </span>
          {garch_active && (
            <span className="px-2 py-1 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
              GARCH Active
            </span>
          )}
        </div>
      </div>

      {/* CVaR Comparison */}
      <div className="grid grid-cols-2 gap-4 mb-4">
        <div className={`p-3 rounded-lg border ${getSeverityBg(cvar_ratio)}`}>
          <p className="text-xs text-gray-600 mb-1">CVaR 95% (Historical)</p>
          <p className="text-2xl font-bold text-gray-900">
            {(cvar_95 * 100).toFixed(2)}%
          </p>
          <p className="text-xs text-gray-500 mt-1">
            Avg loss in worst 5%
          </p>
        </div>
        <div className={`p-3 rounded-lg border ${getSeverityBg(cvar_ratio)}`}>
          <p className="text-xs text-gray-600 mb-1">CVaR 95% (GARCH)</p>
          <p className="text-2xl font-bold text-gray-900">
            {(cvar_95_garch * 100).toFixed(2)}%
          </p>
          <p className="text-xs text-gray-500 mt-1">
            Volatility-adjusted
          </p>
        </div>
      </div>

      {/* VaR Comparison */}
      <div className="grid grid-cols-2 gap-4 mb-4">
        <div className="p-3 bg-gray-50 rounded-lg border border-gray-200">
          <p className="text-xs text-gray-600 mb-1">VaR 95% (Historical)</p>
          <p className="text-xl font-semibold text-gray-900">
            {(var_95 * 100).toFixed(2)}%
          </p>
        </div>
        <div className="p-3 bg-gray-50 rounded-lg border border-gray-200">
          <p className="text-xs text-gray-600 mb-1">VaR 95% (GARCH)</p>
          <p className="text-xl font-semibold text-gray-900">
            {(var_95_garch * 100).toFixed(2)}%
          </p>
        </div>
      </div>

      {/* CVaR Ratio Gauge */}
      <div className="mb-4">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm font-medium text-gray-700">Tail Risk Severity (CVaR/VaR Ratio)</span>
          <span className={`text-lg font-bold ${getSeverityColor(cvar_ratio)}`}>
            {cvar_ratio.toFixed(2)}x
          </span>
        </div>
        <div className="relative h-4 bg-gray-200 rounded-full overflow-hidden">
          <div 
            className={`absolute h-full transition-all duration-500 ${
              cvar_ratio < 1.3 ? 'bg-green-500' :
              cvar_ratio < 1.5 ? 'bg-yellow-500' :
              cvar_ratio < 1.8 ? 'bg-orange-500' : 'bg-red-500'
            }`}
            style={{ width: `${Math.min((cvar_ratio / 2.5) * 100, 100)}%` }}
          />
          {/* Threshold markers */}
          <div className="absolute top-0 bottom-0 w-0.5 bg-gray-400" style={{ left: '52%' }} />
          <div className="absolute top-0 bottom-0 w-0.5 bg-gray-400" style={{ left: '60%' }} />
          <div className="absolute top-0 bottom-0 w-0.5 bg-gray-400" style={{ left: '72%' }} />
        </div>
        <div className="flex justify-between text-xs text-gray-500 mt-1">
          <span>Normal (&lt;1.3)</span>
          <span>Monitor (1.5)</span>
          <span>Elevated (1.8)</span>
          <span>Severe (&gt;2.0)</span>
        </div>
      </div>

      {/* Volatility Metrics */}
      <div className="grid grid-cols-2 gap-4 mb-4">
        <div className="p-3 bg-blue-50 rounded-lg border border-blue-100">
          <p className="text-xs text-blue-700 mb-1">Current Volatility</p>
          <p className="text-lg font-semibold text-blue-900">
            {(current_volatility * 100).toFixed(2)}%
          </p>
        </div>
        <div className="p-3 bg-purple-50 rounded-lg border border-purple-100">
          <p className="text-xs text-purple-700 mb-1">Forecast Volatility (1-day)</p>
          <p className="text-lg font-semibold text-purple-900">
            {(forecast_volatility * 100).toFixed(2)}%
          </p>
        </div>
      </div>

      {/* Interpretation */}
      <div className="p-3 bg-gray-50 rounded-lg border border-gray-200">
        <p className="text-sm font-medium text-gray-700 mb-1">Interpretation</p>
        <p className="text-xs text-gray-600">
          {cvar_ratio < 1.3 && "Normal tail risk distribution. CVaR captures typical tail behavior."}
          {cvar_ratio >= 1.3 && cvar_ratio < 1.5 && "Moderate tail risk. Monitor for volatility clustering."}
          {cvar_ratio >= 1.5 && cvar_ratio < 1.8 && "Elevated tail risk. GARCH filtering active for better estimates."}
          {cvar_ratio >= 1.8 && "Severe tail risk detected. Consider reducing equity exposure 10-15%."}
        </p>
      </div>

      {/* Methodology Note */}
      <div className="mt-4 pt-4 border-t border-gray-200">
        <p className="text-xs text-gray-500">
          <strong>GARCH(1,1) Filtering:</strong> Standardizes returns by conditional volatility 
          to improve CVaR accuracy during volatility clustering periods. 
          Provides 15-20% better tail risk estimates when markets exhibit autocorrelated volatility.
        </p>
      </div>
    </div>
  );
}
