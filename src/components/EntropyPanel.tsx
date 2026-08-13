import type { EntropyData } from '../types/live';

interface EntropyPanelProps {
  data?: EntropyData | null;
}

type EntropyTone = 'success' | 'info' | 'warning' | 'caution' | 'danger';

function riskTone(risk: string): EntropyTone {
  switch (risk) {
    case 'critical':
      return 'danger';
    case 'high':
      return 'caution';
    case 'medium':
      return 'warning';
    case 'low':
      return 'info';
    case 'good':
    default:
      return 'success';
  }
}

export function EntropyPanel({ data }: EntropyPanelProps) {
  if (!data) {
    return (
      <div className="risk-card risk-card-empty">
        <h3>Diversification Monitor</h3>
        <p>Loading entropy metrics...</p>
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
    participation_ratio,
  } = data;

  const tone = riskTone(concentration_risk);
  const gaugePercentage = Math.min(normalized_score, 100);
  const gaugeTone =
    normalized_score > 80 ? 'success' :
    normalized_score > 60 ? 'warning' :
    normalized_score > 40 ? 'caution' : 'danger';

  return (
    <div className="risk-card">
      <div className="risk-card-header">
        <h3>Diversification Monitor (Entropy)</h3>
        <span className={`risk-badge risk-badge-${tone}`}>
          {concentration_risk.charAt(0).toUpperCase() + concentration_risk.slice(1)} Risk
        </span>
      </div>

      <div className="risk-metric-grid">
        <div className={`risk-metric risk-tone-${tone}`}>
          <span className="risk-label">Shannon Entropy</span>
          <span className="risk-value">{shannon_entropy.toFixed(2)}</span>
          <small>Max possible: {max_possible.toFixed(2)}</small>
        </div>
        <div className={`risk-metric risk-tone-${tone}`}>
          <span className="risk-label">Effective N (Diversification)</span>
          <span className="risk-value">{effective_n.toFixed(2)}</span>
          <small>Uncorrelated bets</small>
        </div>
      </div>

      <div className="risk-gauge">
        <div className="risk-gauge-header">
          <span>Diversification Score</span>
          <strong className={`risk-text-${gaugeTone}`}>{normalized_score.toFixed(1)}%</strong>
        </div>
        <div className="risk-gauge-track risk-gauge-track-tall">
          <div
            className={`risk-gauge-fill risk-gauge-fill-${gaugeTone}`}
            style={{ width: `${gaugePercentage}%` }}
          />
          <span className="risk-gauge-marker" style={{ left: '50%' }} />
          <span className="risk-gauge-marker" style={{ left: '70%' }} />
          <span className="risk-gauge-marker" style={{ left: '90%' }} />
        </div>
        <div className="risk-gauge-labels">
          <span>Critical (&lt;50)</span>
          <span>Warning (70)</span>
          <span>Good (90)</span>
          <span>Excellent (100)</span>
        </div>
      </div>

      <div className="risk-metric-grid">
        <div className="risk-metric">
          <span className="risk-label">Herfindahl-Hirschman Index</span>
          <span className="risk-value compact">{hhi_index.toFixed(4)}</span>
          <small>Lower = more diversified</small>
        </div>
        {correlation_entropy !== undefined && correlation_entropy !== null && (
          <div className="risk-metric">
            <span className="risk-label">Correlation Entropy</span>
            <span className="risk-value compact">{correlation_entropy.toFixed(3)}</span>
            <small>Structure diversity</small>
          </div>
        )}
      </div>

      {participation_ratio !== undefined && participation_ratio !== null && (
        <div className="risk-note risk-note-info">
          <div className="risk-note-row">
            <strong>Participation Ratio</strong>
            <span>{participation_ratio.toFixed(2)}</span>
          </div>
          <p>
            Number of significant eigenvalues in correlation matrix. Higher values indicate
            more independent risk factors.
          </p>
        </div>
      )}

      <div className="risk-note">
        <strong>Interpretation</strong>
        <p>
          {concentration_risk === 'critical' && 'Severe concentration risk. Portfolio is heavily dependent on few assets. Consider immediate diversification.'}
          {concentration_risk === 'high' && 'High concentration detected. Portfolio may suffer during correlated drawdowns. Increase asset diversity.'}
          {concentration_risk === 'medium' && 'Moderate diversification. Acceptable for tactical portfolios but monitor for concentration increases.'}
          {concentration_risk === 'low' && 'Good diversification. Portfolio benefits from multiple independent return sources.'}
          {concentration_risk === 'good' && 'Excellent diversification. Well-balanced portfolio with broad risk distribution.'}
        </p>
      </div>

      <div className="risk-footnote">
        <p><strong>Shannon Entropy:</strong> H = -Σ(wᵢ × ln(wᵢ)) measures portfolio weight concentration</p>
        <p><strong>Effective N:</strong> Nₑff = exp(H) represents equivalent number of uncorrelated bets</p>
        <p><strong>HHI Index:</strong> Σ(wᵢ²) is the Herfindahl-Hirschman concentration measure</p>
      </div>
    </div>
  );
}
