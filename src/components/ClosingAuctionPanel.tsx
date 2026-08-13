import type { ClosingAuctionSignal } from '../types/live';

interface ClosingAuctionPanelProps {
  signals: ClosingAuctionSignal[];
  isMarketOpen?: boolean;
}

export function ClosingAuctionPanel({ signals, isMarketOpen = true }: ClosingAuctionPanelProps) {
  const getDirectionColor = (direction: string) => {
    switch (direction) {
      case 'STRONG_BUY':
      case 'BUY':
        return '#10b981'; // green
      case 'WEAK_BUY':
        return '#34d399'; // light green
      case 'STRONG_SELL':
      case 'SELL':
        return '#ef4444'; // red
      case 'WEAK_SELL':
        return '#f87171'; // light red
      default:
        return '#6b7280'; // gray
    }
  };

  const getConfidenceBadge = (confidence: string) => {
    switch (confidence) {
      case 'high':
        return { bg: '#10b981', text: 'High' };
      case 'medium':
        return { bg: '#f59e0b', text: 'Medium' };
      case 'low':
        return { bg: '#ef4444', text: 'Low' };
      default:
        return { bg: '#6b7280', text: 'N/A' };
    }
  };

  const getUrgencyIcon = (urgency: string) => {
    switch (urgency) {
      case 'immediate':
        return '🔴';
      case 'high':
        return '🟡';
      case 'normal':
        return '🟢';
      default:
        return '⚪';
    }
  };

  const activeSignals = signals?.filter(s => s.should_trade) || [];
  const allSignals = signals || [];

  return (
    <div className="closing-auction-panel">
      <div className="auction-header">
        <h3>
          📊 Closing Auction (MOC)
          <small style={{ fontWeight: 400, color: '#9ca3af', fontSize: '12px' }}>
            3:50-4:00pm ET
          </small>
        </h3>
        <span className={`market-status ${isMarketOpen ? 'open' : 'closed'}`}>
          {isMarketOpen ? 'Market Open' : 'Market Closed'}
        </span>
      </div>

      {isMarketOpen && (
        <div className="entry-window">
          <span className="label">Entry Window</span>
          <span className="time">3:50 PM - 3:55 PM ET</span>
        </div>
      )}

      <div className="auction-summary">
        <div className="summary-card">
          <div className="value" style={{ color: activeSignals.length > 0 ? '#10b981' : '#9ca3af' }}>
            {activeSignals.length}
          </div>
          <div className="label">Active Signals</div>
        </div>
        <div className="summary-card">
          <div className="value" style={{ color: allSignals.length > 0 ? '#3b82f6' : '#9ca3af' }}>
            {allSignals.length}
          </div>
          <div className="label">Total Monitored</div>
        </div>
        <div className="summary-card">
          <div className="value" style={{ color: '#f59e0b' }}>
            {activeSignals.filter(s => s.urgency === 'immediate').length}
          </div>
          <div className="label">Immediate</div>
        </div>
      </div>

      {allSignals.length > 0 ? (
        <table className="signals-table">
          <thead>
            <tr>
              <th>Symbol</th>
              <th>Direction</th>
              <th>Confidence</th>
              <th>Win Rate</th>
              <th>Urgency</th>
            </tr>
          </thead>
          <tbody>
            {allSignals.map((signal, idx) => {
              const confidenceBadge = getConfidenceBadge(signal.confidence);
              return (
                <tr key={idx} style={{ opacity: signal.should_trade ? 1 : 0.5 }}>
                  <td>
                    <strong>{signal.symbol}</strong>
                  </td>
                  <td>
                    <span 
                      className="direction-badge"
                      style={{ color: getDirectionColor(signal.direction) }}
                    >
                      {signal.direction.replace('_', ' ')}
                    </span>
                  </td>
                  <td>
                    <span 
                      className="confidence-badge"
                      style={{ 
                        background: confidenceBadge.bg + '30',
                        color: confidenceBadge.bg 
                      }}
                    >
                      {confidenceBadge.text}
                    </span>
                  </td>
                  <td>
                    {signal.historical_win_rate ? (
                      <span className={`win-rate ${signal.historical_win_rate < 0.55 ? 'low' : ''}`}>
                        {(signal.historical_win_rate * 100).toFixed(0)}%
                      </span>
                    ) : (
                      <span style={{ color: '#6b7280' }}>N/A</span>
                    )}
                  </td>
                  <td>
                    {getUrgencyIcon(signal.urgency)} {signal.urgency}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      ) : (
        <div className="no-signals">
          <h4>No MOC Signals</h4>
          <p>Closing auction data available 3:30-4:00pm ET</p>
        </div>
      )}

      <div className="legend">
        <div className="legend-item">
          <span className="legend-icon">🔴</span> Immediate: Enter now
        </div>
        <div className="legend-item">
          <span className="legend-icon">🟡</span> High: Within 2 min
        </div>
        <div className="legend-item">
          <span className="legend-icon">🟢</span> Normal: Before 3:55pm
        </div>
        <div className="legend-item">
          <span style={{ color: '#10b981' }}>●</span> High Conf (65%+ win rate)
        </div>
        <div className="legend-item">
          <span style={{ color: '#f59e0b' }}>●</span> Medium Conf (55-65%)
        </div>
      </div>
    </div>
  );
}
