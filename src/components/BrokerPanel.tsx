import React from 'react';
import type { BrokerData } from '../types/live';

interface BrokerPanelProps {
  data?: BrokerData;
}

export function BrokerPanel({ data }: BrokerPanelProps) {
  if (!data) {
    return (
      <div className="broker-panel">
        <div className="broker-header">
          <h3>Broker Connection</h3>
          <span className="broker-status disconnected">Not Configured</span>
        </div>
        <div className="broker-empty">
          <p>No broker connection configured.</p>
          <small>Set ALPACA_API_KEY and ALPACA_API_SECRET to enable paper trading.</small>
        </div>
      </div>
    );
  }

  const formatCurrency = (v: number) =>
    new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(v);

  return (
    <div className="broker-panel">
      <div className="broker-header">
        <h3>Broker Connection</h3>
        <div className="broker-status-row">
          <span className={`broker-status ${data.connected ? 'connected' : 'disconnected'}`}>
            {data.connected ? 'Connected' : 'Disconnected'}
          </span>
          {data.kill_switch && (
            <span className="broker-kill-switch">KILL SWITCH ACTIVE</span>
          )}
        </div>
      </div>

      {data.last_sync && (
        <div className="broker-sync-info">
          Last sync: {new Date(data.last_sync).toLocaleString()}
        </div>
      )}

      {/* Broker Positions */}
      {data.positions.length > 0 && (
        <div className="broker-positions">
          <h4>Broker Positions</h4>
          <table className="broker-table">
            <thead>
              <tr>
                <th>Symbol</th>
                <th>Qty</th>
                <th>Value</th>
                <th>P&L</th>
              </tr>
            </thead>
            <tbody>
              {data.positions.map((pos) => (
                <tr key={pos.symbol}>
                  <td><strong>{pos.symbol}</strong></td>
                  <td>{pos.qty.toFixed(2)}</td>
                  <td>{formatCurrency(pos.market_value)}</td>
                  <td className={pos.unrealized_pl >= 0 ? 'positive' : 'negative'}>
                    {formatCurrency(pos.unrealized_pl)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Drift Detection */}
      {data.drift.length > 0 && (
        <div className="broker-drift">
          <h4>Position Drift</h4>
          <table className="broker-table">
            <thead>
              <tr>
                <th>Symbol</th>
                <th>Broker</th>
                <th>Local</th>
                <th>Drift</th>
              </tr>
            </thead>
            <tbody>
              {data.drift.map((d) => (
                <tr key={d.symbol}>
                  <td><strong>{d.symbol}</strong></td>
                  <td>{d.broker_qty.toFixed(2)}</td>
                  <td>{d.local_qty.toFixed(2)}</td>
                  <td className={Math.abs(d.drift_pct) > 5 ? 'warning' : ''}>
                    {d.drift_pct.toFixed(1)}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Recent Broker Orders */}
      {data.recent_orders.length > 0 && (
        <div className="broker-orders">
          <h4>Recent Broker Orders</h4>
          <table className="broker-table">
            <thead>
              <tr>
                <th>Symbol</th>
                <th>Side</th>
                <th>Qty</th>
                <th>Status</th>
                <th>Time</th>
              </tr>
            </thead>
            <tbody>
              {data.recent_orders.map((order, i) => (
                <tr key={i}>
                  <td><strong>{order.symbol}</strong></td>
                  <td className={order.side === 'buy' ? 'positive' : 'negative'}>
                    {order.side.toUpperCase()}
                  </td>
                  <td>{order.qty.toFixed(2)}</td>
                  <td>
                    <span className={`order-status ${order.status}`}>
                      {order.dry_run ? 'DRY RUN' : order.status}
                    </span>
                  </td>
                  <td>{new Date(order.timestamp).toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {!data.connected && (
        <div className="broker-empty">
          <p>Broker not connected. Check API credentials.</p>
        </div>
      )}
    </div>
  );
}
