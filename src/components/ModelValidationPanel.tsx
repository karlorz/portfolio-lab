import React from 'react';

interface ModelValidationProps {
  /** Deflated Sharpe Ratio value (0-1), null if unavailable */
  dsr: number | null;
  /** Number of grid search trials used for DSR */
  nTrials?: number;
  /** Champion portfolio Sharpe ratio */
  championSharpe?: number | null;
  /** BL posterior weights for comparison, null if unavailable */
  blWeights: Record<string, number> | null;
  /** Current overlay-based weights for comparison */
  overlayWeights?: Record<string, number> | null;
}

export function ModelValidationPanel({
  dsr,
  nTrials = 94,
  championSharpe,
  blWeights,
  overlayWeights,
}: ModelValidationProps) {
  const getDsrColor = (val: number) => {
    if (val >= 0.95) return '#10b981'; // High confidence - green
    if (val >= 0.50) return '#f59e0b'; // Moderate - amber
    return '#ef4444'; // Low confidence - red
  };

  const formatWeight = (w: number) => `${(w * 100).toFixed(1)}%`;

  return (
    <div className="model-validation-panel">
      <div className="panel-header">
        <h3>Model Validation</h3>
      </div>

      <div className="validation-section">
        <h4>Deflated Sharpe Ratio</h4>
        {dsr !== null ? (
          <div className="dsr-display">
            <span
              className="dsr-value"
              style={{ color: getDsrColor(dsr) }}
            >
              {dsr.toFixed(3)}
            </span>
            <span className="dsr-label">
              {dsr >= 0.95 ? 'Champion validated' : dsr >= 0.50 ? 'Borderline' : 'Not significant'}
            </span>
            <div className="dsr-meta">
              {championSharpe !== null && championSharpe !== undefined && (
                <span>Sharpe: {championSharpe.toFixed(2)}</span>
              )}
              <span>{nTrials} configs tested</span>
            </div>
          </div>
        ) : (
          <span className="unavailable">DSR unavailable</span>
        )}
      </div>

      {blWeights && overlayWeights && (
        <div className="validation-section">
          <h4>BL vs Overlay Weights</h4>
          <table className="weight-comparison">
            <thead>
              <tr>
                <th>Asset</th>
                <th>Overlay</th>
                <th>BL Posterior</th>
                <th>Delta</th>
              </tr>
            </thead>
            <tbody>
              {Object.keys(blWeights)
                .sort()
                .map((sym) => {
                  const bl = blWeights[sym] || 0;
                  const ov = overlayWeights[sym] || 0;
                  const delta = bl - ov;
                  return (
                    <tr key={sym}>
                      <td>{sym}</td>
                      <td>{formatWeight(ov)}</td>
                      <td>{formatWeight(bl)}</td>
                      <td
                        style={{
                          color:
                            Math.abs(delta) > 0.05
                              ? '#f59e0b'
                              : '#10b981',
                        }}
                      >
                        {delta >= 0 ? '+' : ''}
                        {(delta * 100).toFixed(1)}pp
                      </td>
                    </tr>
                  );
                })}
            </tbody>
          </table>
        </div>
      )}

      {(!blWeights || !overlayWeights) && (
        <div className="validation-section">
          <h4>BL Posterior Weights</h4>
          <span className="unavailable">
            {blWeights
              ? 'Overlay weights unavailable'
              : 'BL weights unavailable (PyPortfolioOpt may not be installed)'}
          </span>
        </div>
      )}
    </div>
  );
}
