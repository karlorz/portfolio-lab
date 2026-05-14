import React from 'react';
import type { EntropyData } from '../types/live';

interface EntropyPanelProps {
  data?: EntropyData | null;
}

export function EntropyPanel({ data }: EntropyPanelProps) {
  if (!data) {
    return (
      <div className="p-4 bg-gray-50 rounded-lg">
        <h3 className="text-lg font-semibold mb-2">Diversification Monitor</h3>
        <p className="text-gray-500">Loading entropy metrics...</p>
      </div>
    );
  }

  const { 
    shannon_entropy,
    effective_n,
    max_possible,
    normalized_score,
    concentration_risk,
    hhi_index,
    correlation_entropy,
    participation_ratio
  } = data;

  // Risk level color coding
  const getRiskColor = (risk: string) => {
    switch (risk) {
      case 'critical': return 'text-red-600';
      case 'high': return 'text-orange-600';
      case 'medium': return 'text-yellow-600';
      case 'low':
      case 'good':
      default: return 'text-green-600';
    }
  };

  const getRiskBg = (risk: string) => {
    switch (risk) {
      case 'critical': return 'bg-red-50 border-red-200';
      case 'high': return 'bg-orange-50 border-orange-200';
      case 'medium': return 'bg-yellow-50 border-yellow-200';
      case 'low':
      case 'good':
      default: return 'bg-green-50 border-green-200';
    }
  };

  const getRiskBadgeColor = (risk: string) => {
    switch (risk) {
      case 'critical': return 'bg-red-100 text-red-800';
      case 'high': return 'bg-orange-100 text-orange-800';
      case 'medium': return 'bg-yellow-100 text-yellow-800';
      case 'low': return 'bg-blue-100 text-blue-800';
      case 'good':
      default: return 'bg-green-100 text-green-800';
    }
  };

  // Calculate gauge percentage
  const gaugePercentage = Math.min((normalized_score / 100) * 100, 100);

  return (
    <div className="p-4 bg-white rounded-lg shadow-sm border border-gray-200">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold">Diversification Monitor (Entropy)</h3>
        <span className={`px-2 py-1 rounded-full text-xs font-medium ${getRiskBadgeColor(concentration_risk)}`}>
          {concentration_risk.charAt(0).toUpperCase() + concentration_risk.slice(1)} Risk
        </span>
      </div>

      {/* Main Metrics */}
      <div className="grid grid-cols-2 gap-4 mb-4">
        <div className={`p-3 rounded-lg border ${getRiskBg(concentration_risk)}`}>
          <p className="text-xs text-gray-600 mb-1">Shannon Entropy</p>
          <p className="text-2xl font-bold text-gray-900">
            {shannon_entropy.toFixed(2)}
          </p>
          <p className="text-xs text-gray-500 mt-1">
            Max possible: {max_possible.toFixed(2)}
          </p>
        </div>
        <div className={`p-3 rounded-lg border ${getRiskBg(concentration_risk)}`}>
          <p className="text-xs text-gray-600 mb-1">Effective N (Diversification)</p>
          <p className="text-2xl font-bold text-gray-900">
            {effective_n.toFixed(2)}
          </p>
          <p className="text-xs text-gray-500 mt-1">
            Uncorrelated bets
          </p>
        </div>
      </div>

      {/* Normalized Score Gauge */}
      <div className="mb-4">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm font-medium text-gray-700">Diversification Score</span>
          <span className={`text-lg font-bold ${getRiskColor(concentration_risk)}`}>
            {normalized_score.toFixed(1)}%
          </span>
        </div>
        <div className="relative h-4 bg-gray-200 rounded-full overflow-hidden">
          <div 
            className={`absolute h-full transition-all duration-500 ${
              normalized_score > 80 ? 'bg-green-500' :
              normalized_score > 60 ? 'bg-yellow-500' :
              normalized_score > 40 ? 'bg-orange-500' : 'bg-red-500'
            }`}
            style={{ width: `${gaugePercentage}%` }}
          />
          {/* Threshold markers */}
          <div className="absolute top-0 bottom-0 w-0.5 bg-gray-400" style={{ left: '50%' }} />
          <div className="absolute top-0 bottom-0 w-0.5 bg-gray-400" style={{ left: '70%' }} />
          <div className="absolute top-0 bottom-0 w-0.5 bg-gray-400" style={{ left: '90%' }} />
        </div>
        <div className="flex justify-between text-xs text-gray-500 mt-1">
          <span>Critical (&lt;50)</span>
          <span>Warning (70)</span>
          <span>Good (90)</span>
          <span>Excellent (100)</span>
        </div>
      </div>

      {/* Additional Metrics */}
      <div className="grid grid-cols-2 gap-4 mb-4">
        <div className="p-3 bg-gray-50 rounded-lg border border-gray-200">
          <p className="text-xs text-gray-600 mb-1">Herfindahl-Hirschman Index</p>
          <p className="text-xl font-semibold text-gray-900">
            {hhi_index.toFixed(4)}
          </p>
          <p className="text-xs text-gray-500 mt-1">
            Lower = more diversified
          </p>
        </div>
        {correlation_entropy !== undefined && correlation_entropy !== null && (
          <div className="p-3 bg-gray-50 rounded-lg border border-gray-200">
            <p className="text-xs text-gray-600 mb-1">Correlation Entropy</p>
            <p className="text-xl font-semibold text-gray-900">
              {correlation_entropy.toFixed(3)}
            </p>
            <p className="text-xs text-gray-500 mt-1">
              Structure diversity
            </p>
          </div>
        )}
      </div>

      {/* Participation Ratio */}
      {participation_ratio !== undefined && participation_ratio !== null && (
        <div className="p-3 bg-blue-50 rounded-lg border border-blue-100 mb-4">
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium text-blue-900">Participation Ratio</span>
            <span className="text-lg font-bold text-blue-900">
              {participation_ratio.toFixed(2)}
            </span>
          </div>
          <p className="text-xs text-blue-700 mt-1">
            Number of significant eigenvalues in correlation matrix. 
            Higher values indicate more independent risk factors.
          </p>
        </div>
      )}

      {/* Interpretation */}
      <div className="p-3 bg-gray-50 rounded-lg border border-gray-200">
        <p className="text-sm font-medium text-gray-700 mb-1">Interpretation</p>
        <p className="text-xs text-gray-600">
          {concentration_risk === 'critical' && "Severe concentration risk. Portfolio is heavily dependent on few assets. Consider immediate diversification."}
          {concentration_risk === 'high' && "High concentration detected. Portfolio may suffer during correlated drawdowns. Increase asset diversity."}
          {concentration_risk === 'medium' && "Moderate diversification. Acceptable for tactical portfolios but monitor for concentration increases."}
          {concentration_risk === 'low' && "Good diversification. Portfolio benefits from multiple independent return sources."}
          {concentration_risk === 'good' && "Excellent diversification. Well-balanced portfolio with broad risk distribution."}
        </p>
      </div>

      {/* Formula Reference */}
      <div className="mt-4 pt-4 border-t border-gray-200">
        <p className="text-xs text-gray-500 mb-2">
          <strong>Shannon Entropy:</strong> H = -Σ(wᵢ × ln(wᵢ)) measures portfolio weight concentration
        </p>
        <p className="text-xs text-gray-500 mb-2">
          <strong>Effective N:</strong> Nₑff = exp(H) represents equivalent number of uncorrelated bets
        </p>
        <p className="text-xs text-gray-500">
          <strong>HHI Index:</strong> Σ(wᵢ²) is the Herfindahl-Hirschman concentration measure
        </p>
      </div>
    </div>
  );
}
