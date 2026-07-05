import { describe, expect, it } from 'bun:test';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { DecisionReplayPanel } from '../../src/components/DecisionReplayPanel';
import { PanelErrorBoundary } from '../../src/components/PanelErrorBoundary';
import { RebalancePanel } from '../../src/components/RebalancePanel';
import {
  DECISION_REGISTRY_SCHEMA_VERSION,
  type DecisionRegistryData,
} from '../../src/schemas/decision_registry';
import type { SignalsData } from '../../src/types/live';

function renderPanel(name: string, child: React.ReactElement): string {
  return renderToStaticMarkup(
    <PanelErrorBoundary name={name}>
      {child}
    </PanelErrorBoundary>,
  );
}

function validSignals(): SignalsData {
  return {
    timestamp: '2026-07-04T12:00:00Z',
    regime: { regime: 'normal', vix: 15.2, detected: '2026-07-04T12:00:00Z' },
    latest_prices: { SPY: 550, GLD: 195, TLT: 92 },
    current_positions: [
      { symbol: 'SPY', shares: 124, value: 68200, weight: 0.62, unrealized: 1200 },
      { symbol: 'GLD', shares: 158, value: 30800, weight: 0.28, unrealized: 400 },
      { symbol: 'TLT', shares: 120, value: 11000, weight: 0.10, unrealized: -200 },
    ],
    target_allocations: { SPY: 0.46, GLD: 0.38, TLT: 0.16 },
    cash: 1000,
    total_value: 110000,
    recent_orders: [],
    ml_signals: {
      available: false,
      timestamp: null,
      predictions: {},
      features: {},
      grid_search: {
        available: false,
        timestamp: null,
        top_allocation: null,
        sharpe: null,
        volatility: null,
      },
    },
  };
}

function decisionRegistry(): DecisionRegistryData {
  return {
    schema_version: DECISION_REGISTRY_SCHEMA_VERSION,
    generated_at: '2026-07-04T12:05:00Z',
    recent_decisions: [
      {
        decision_id: 'decision-1',
        timestamp_utc: '2026-07-04T12:00:00Z',
        run_id: 'dashboard-cycle',
        action: 'hold',
        reason: 'Drift below threshold',
        regime: 'NORMAL',
        regime_confidence: 0.82,
        current_weights: { SPY: 0.46, GLD: 0.38, TLT: 0.16 },
        target_weights: { SPY: 0.46, GLD: 0.38, TLT: 0.16 },
        gates_triggered: [],
      },
    ],
    recent_experiments: [],
    replay_summaries: [
      {
        decision_id: 'decision-1',
        found: true,
        replay: {
          summary: 'Held allocation after replaying drift and signal gates.',
          action: 'hold',
          top_signal_weights: [
            { signal: 'ALT_DATA', weight: 0.22 },
            { signal: 'INTL_MOM', weight: 0.21 },
          ],
        },
      },
    ],
    promotion_evaluations: [],
    promotion_coverage: {
      scope: 'recent_experiments',
      recent_experiment_count: 0,
      evaluated_experiment_count: 0,
      unmatched_experiment_count: 0,
      unmatched_experiment_ids: [],
      disclosure: 'complete_promotion_evaluation_coverage',
    },
    counts: { decisions: 1, experiments: 0 },
  };
}

describe('panel smoke rendering', () => {
  it('renders allocation drift panel through PanelErrorBoundary', () => {
    const html = renderPanel(
      'Rebalance',
      <RebalancePanel signals={validSignals()} readOnly={true} />,
    );

    expect(html).toContain('Allocation Drift Monitor');
    expect(html).toContain('REBALANCE NEEDED');
    expect(html).toContain('SPY');
    expect(html).toContain('Read-only mode');
    expect(html).not.toContain('panel-error-boundary');
  });

  it('renders pre-fetched decision replay detail through PanelErrorBoundary on first paint', () => {
    const html = renderPanel(
      'Decisions',
      <DecisionReplayPanel initialData={decisionRegistry()} />,
    );

    expect(html).toContain('Decision Replay');
    expect(html).toContain('Recent decisions');
    expect(html).toContain('Held allocation after replaying drift and signal gates.');
    expect(html).toContain('dashboard-cycle');
    expect(html).not.toContain('Select a decision row.');
    expect(html).not.toContain('panel-error-boundary');
  });

  it('renders unmatched promotion rows with explicit fallback text', () => {
    const registry = decisionRegistry();
    registry.recent_experiments = [
      {
        experiment_id: 'unmatched-exp',
        timestamp_utc: '2026-07-04T12:01:00Z',
        name: 'Unmatched experiment',
        metrics: { sharpe: 1.05 },
        benchmark_metrics: { sharpe: 0.95 },
        promotion_status: 'candidate',
      },
    ];
    registry.promotion_coverage = {
      scope: 'recent_experiments',
      recent_experiment_count: 1,
      evaluated_experiment_count: 0,
      unmatched_experiment_count: 1,
      unmatched_experiment_ids: ['unmatched-exp'],
      disclosure: 'partial_promotion_evaluation_coverage',
    };
    registry.counts.experiments = 1;

    const html = renderPanel(
      'Decisions',
      <DecisionReplayPanel initialData={registry} />,
    );

    expect(html).toContain('unmatched-exp');
    expect(html).toContain('Not evaluated');
    expect(html).toContain('Promotion evaluation not published');
  });
});
