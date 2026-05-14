import React from 'react';
import type { ClosingAuctionSignal } from '../types/live';

interface ClosingAuctionPanelProps {
  signals: ClosingAuctionSignal[];
  isMarketOpen?: boolean;
}

export function ClosingAuctionPanel({ signals, isMarketOpen = true }: ClosingAuctionPanelPropsProps) {
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
      <style>{`
        .closing-auction-panel {
          background: #1a1a2e;
          border-radius: 8px;
          padding: 16px;
          margin: 16px 0;
        }
        
        .auction-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 16px;
          padding-bottom: 12px;
          border-bottom: 1px solid #2d2d44;
        }
        
        .auction-header h3 {
          margin: 0;
          color: #fff;
          font-size: 16px;
          display: flex;
          align-items: center;
          gap: 8px;
        }
        
        .market-status {
          font-size: 12px;
          padding: 4px 12px;
          border-radius: 12px;
          font-weight: 500;
        }
        
        .market-status.open {
          background: #10b98120;
          color: #10b981;
        }
        
        .market-status.closed {
          background: #ef444420;
          color: #ef4444;
        }
        
        .auction-summary {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
          gap: 12px;
          margin-bottom: 16px;
        }
        
        .summary-card {
          background: #2d2d44;
          border-radius: 6px;
          padding: 12px;
          text-align: center;
        }
        
        .summary-card .value {
          font-size: 20px;
          font-weight: 600;
          color: #fff;
        }
        
        .summary-card .label {
          font-size: 11px;
          color: #9ca3af;
          margin-top: 4px;
        }
        
        .signals-table {
          width: 100%;
          border-collapse: collapse;
          font-size: 13px;
        }
        
        .signals-table th {
          text-align: left;
          padding: 10px 8px;
          color: #9ca3af;
          font-weight: 500;
          border-bottom: 1px solid #2d2d44;
        }
        
        .signals-table td {
          padding: 10px 8px;
          border-bottom: 1px solid #2d2d44;
          color: #e5e7eb;
        }
        
        .signals-table tr:last-child td {
          border-bottom: none;
        }
        
        .direction-badge {
          display: inline-flex;
          align-items: center;
          gap: 6px;
          padding: 4px 10px;
          border-radius: 4px;
          font-weight: 500;
          font-size: 12px;
          background: rgba(0,0,0,0.2);
        }
        
        .confidence-badge {
          display: inline-block;
          padding: 2px 8px;
          border-radius: 4px;
          font-size: 11px;
          font-weight: 500;
        }
        
        .win-rate {
          font-family: monospace;
          color: #10b981;
        }
        
        .win-rate.low {
          color: #f59e0b;
        }
        
        .no-signals {
          text-align: center;
          padding: 40px 20px;
          color: #6b7280;
        }
        
        .no-signals h4 {
          margin: 0 0 8px 0;
          color: #9ca3af;
        }
        
        .entry-window {
          background: #2d2d44;
          border-radius: 6px;
          padding: 12px 16px;
          margin-bottom: 16px;
          display: flex;
          justify-content: space-between;
          align-items: center;
        }
        
        .entry-window .time {
          font-family: monospace;
          font-size: 14px;
          color: #f59e0b;
        }
        
        .entry-window .label {
          font-size: 12px;
          color: #9ca3af;
        }
        
        .legend {
          display: flex;
          gap: 16px;
          margin-top: 16px;
          padding-top: 12px;
          border-top: 1px solid #2d2d44;
          font-size: 11px;
          color: #9ca3af;
          flex-wrap: wrap;
        }
        
        .legend-item {
          display: flex;
          align-items: center;
          gap: 4px;
        }
        
        .legend-icon {
          font-size: 10px;
        }
      `}</style>

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
