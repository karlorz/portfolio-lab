import { describe, expect, it } from 'bun:test';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { DecisionReplayPanel } from '../../src/components/DecisionReplayPanel';
import { MLSignalsPanel } from '../../src/components/MLSignalsPanel';
import { PanelErrorBoundary } from '../../src/components/PanelErrorBoundary';
import { RebalancePanel } from '../../src/components/RebalancePanel';
import {
  StackingEnsemblePanel,
  type StackingEnsembleData,
} from '../../src/components/StackingEnsemblePanel';
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

function emptyShadowEvidence(): DecisionRegistryData['shadow_evidence'] {
  return {
    shadow_decision_count: 0,
    linked_shadow_decision_count: 0,
    controllers: [],
    promotion_review_status_counts: {},
    latest_shadow_decisions: [],
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
        decision_role: 'live_executed',
        decision_source: 'dashboard_cycle',
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
      recent_experiment_count: 0,
      evaluated_count: 0,
      unmatched_count: 0,
      unmatched_experiment_ids: [],
      disclosure: 'complete_promotion_evaluation_coverage',
    },
    shadow_evidence: emptyShadowEvidence(),
    counts: { decisions: 1, experiments: 0 },
  };
}

function decisionRegistryWithShadowEvidence(): DecisionRegistryData {
  return {
    schema_version: DECISION_REGISTRY_SCHEMA_VERSION,
    generated_at: '2026-07-04T12:05:00Z',
    recent_decisions: [
      {
        decision_id: 'shadow-1',
        timestamp_utc: '2026-07-04T12:00:00Z',
        run_id: 'dashboard-shadow-cycle',
        decision_role: 'shadow',
        decision_source: 'dashboard_cycle',
        controller_id: 'ensemble_voter_shadow',
        live_decision_id: 'live-1',
        baseline_controller_id: 'regime_allocation_live',
        benchmark_window: { label: 'dashboard_cycle', observations: 1 },
        divergence_metrics: { max_weight_delta: 0.04, action_mismatch: 1 },
        promotion_review_status: 'pending_review',
        action: 'rebalance',
        reason: 'Shadow candidate would rebalance',
        regime: 'NORMAL',
        current_weights: { SPY: 0.5, GLD: 0.3, TLT: 0.2 },
        target_weights: { SPY: 0.44, GLD: 0.36, TLT: 0.2 },
        gates_triggered: [],
      },
    ],
    recent_experiments: [],
    replay_summaries: [
      {
        decision_id: 'shadow-1',
        found: true,
        replay: {
          summary: 'Shadow controller diverged from live allocation.',
          action: 'rebalance',
          decision_role: 'shadow',
          controller_id: 'ensemble_voter_shadow',
          live_decision_id: 'live-1',
          baseline_controller_id: 'regime_allocation_live',
          benchmark_window: { label: 'dashboard_cycle', observations: 1 },
          divergence_metrics: { max_weight_delta: 0.04, action_mismatch: 1 },
          promotion_review_status: 'pending_review',
        },
      },
    ],
    promotion_evaluations: [],
    promotion_coverage: {
      recent_experiment_count: 0,
      evaluated_count: 0,
      unmatched_count: 0,
      unmatched_experiment_ids: [],
      disclosure: 'complete_promotion_evaluation_coverage',
    },
    shadow_evidence: {
      shadow_decision_count: 1,
      linked_shadow_decision_count: 1,
      controllers: ['ensemble_voter_shadow'],
      promotion_review_status_counts: { pending_review: 1 },
      latest_shadow_decisions: [
        {
          decision_id: 'shadow-1',
          controller_id: 'ensemble_voter_shadow',
          live_decision_id: 'live-1',
          baseline_controller_id: 'regime_allocation_live',
          benchmark_window: { label: 'dashboard_cycle', observations: 1 },
          divergence_metrics: { max_weight_delta: 0.04, action_mismatch: 1 },
          promotion_review_status: 'pending_review',
        },
      ],
    },
    counts: { decisions: 1, experiments: 0 },
  };
}

function decisionRegistryWithGovernanceBlockedPromotion(): DecisionRegistryData {
  return {
    schema_version: DECISION_REGISTRY_SCHEMA_VERSION,
    generated_at: '2026-07-04T12:05:00Z',
    recent_decisions: [],
    recent_experiments: [
      {
        experiment_id: 'artifact:watch-row',
        timestamp_utc: '2026-07-04T12:00:00Z',
        name: 'artifact:watch-row',
        metrics: { sharpe: 1.05 },
        promotion_status: 'candidate',
        artifacts: { provenance_status: 'missing' },
      },
    ],
    replay_summaries: [],
    promotion_evaluations: [
      {
        experiment_id: 'artifact:watch-row',
        recommended_status: 'candidate',
        metric_gate_status: 'promoted',
        metric_gate_pass: true,
        pass: false,
        failures: ['provenance_missing'],
        semantic_disclosure: {
          state: 'governance_blocked',
          recommendation_type: 'metric_gate',
          governance_status: 'candidate',
          provenance_status: 'missing',
          metric_gate_status: 'promoted',
          reasons: ['provenance_missing'],
        },
      },
    ],
    counts: { decisions: 0, experiments: 1 },
    promotion_coverage: {
      recent_experiment_count: 1,
      evaluated_count: 1,
      unmatched_count: 0,
      unmatched_experiment_ids: [],
      disclosure: 'complete_promotion_evaluation_coverage',
    },
    shadow_evidence: emptyShadowEvidence(),
  };
}

function decisionRegistryWithUnmatchedPromotion(): DecisionRegistryData {
  return {
    schema_version: DECISION_REGISTRY_SCHEMA_VERSION,
    generated_at: '2026-07-04T12:05:00Z',
    recent_decisions: [],
    recent_experiments: [
      {
        experiment_id: 'experiment-10',
        timestamp_utc: '2026-07-04T12:00:00Z',
        name: 'experiment-10',
        metrics: { sharpe: 1.1 },
        promotion_status: 'candidate',
        promotion_disclosure: {
          state: 'not_evaluated',
          reason: 'promotion_evaluation_limit',
          message:
            'No promotion evaluation was published for this experiment in the current snapshot.',
        },
      },
    ],
    replay_summaries: [],
    promotion_evaluations: [],
    promotion_coverage: {
      recent_experiment_count: 12,
      evaluated_count: 10,
      unmatched_count: 2,
      unmatched_experiment_ids: ['experiment-10', 'experiment-11'],
      disclosure: 'partial_promotion_evaluation_coverage',
    },
    shadow_evidence: emptyShadowEvidence(),
    counts: { decisions: 0, experiments: 12 },
  };
}

describe('panel smoke rendering', () => {
  it('renders ML predictions from the live regime-probability contract', () => {
    const mlSignals: SignalsData['ml_signals'] = {
      available: true,
      timestamp: '2026-07-05T08:46:05Z',
      generated_at: '2026-07-05T08:46:05Z',
      feature_source_artifact: 'features.jsonl',
      feature_as_of: '2026-05-08T00:00:00Z',
      feature_freshness_status: 'stale',
      feature_staleness_days: 58,
      prediction_source_mode: 'stale_features',
      execution_role: {
        role: 'advisory_non_routed',
        routed: false,
        routed_by: null,
        live_authoritative: false,
      },
      predictions: {
        GLD: {
          predicted_regime: 'bull',
          confidence: 0.6,
          probabilities: { bear: 0.1, neutral: 0.3, bull: 0.6 },
          heuristic: true,
          feature_timestamp: '2026-05-08T00:00:00Z',
          feature_freshness_status: 'stale',
          source_artifact: 'features.jsonl',
        },
      },
      features: {
        GLD: {
          vix_level: 18.4,
          trend_direction: 1,
          price_vs_sma20: 0.02,
          return_5d: 0.015,
          spy_correlation: 0.24,
          feature_timestamp: '2026-05-08T00:00:00Z',
        },
      },
      grid_search: {
        available: true,
        timestamp: '2026-07-05T08:45:00Z',
        top_allocation: { SPY: 0.46, GLD: 0.38, TLT: 0.16 },
        sharpe: 0.95,
        volatility: 0.111,
        source_artifact: 'grid_search_results.jsonl',
        benchmark_timestamp: '2026-05-22T00:00:00Z',
        observation_semantics: 'frozen_benchmark_not_live_snapshot',
        freshness_status: 'frozen_benchmark',
        staleness_days: 44,
        live_authoritative: false,
      },
    };

    const html = renderPanel('ML Signals', <MLSignalsPanel data={mlSignals} />);

    expect(html).toContain('Predicted Regime');
    expect(html).toContain('bull');
    expect(html).toContain('bear 10%');
    expect(html).toContain('neutral 30%');
    expect(html).toContain('bull 60%');
    expect(html).toContain('Heuristic');
    expect(html).toContain('Feature as of');
    expect(html).toContain('2026-05-08');
    expect(html).toContain('stale');
    expect(html).toContain('Frozen benchmark');
    expect(html).toContain('grid_search_results.jsonl');
    expect(html).not.toContain('Return</th>');
    expect(html).not.toContain('Dir</th>');
    expect(html).not.toContain('panel-error-boundary');
  });

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
    expect(html).toContain('Live executed');
    expect(html).toContain('Dashboard cycle');
    expect(html).toContain('ALT_DATA');
    expect(html).not.toContain('Select a decision row');
    expect(html).not.toContain('panel-error-boundary');
  });

  it('renders shadow evidence fields in the decision replay detail', () => {
    const html = renderPanel(
      'Decisions',
      <DecisionReplayPanel initialData={decisionRegistryWithShadowEvidence()} />,
    );

    expect(html).toContain('Shadow controller diverged from live allocation.');
    expect(html).toContain('Shadow evidence');
    expect(html).toContain('ensemble_voter_shadow');
    expect(html).toContain('live-1');
    expect(html).toContain('regime_allocation_live');
    expect(html).toContain('Pending review');
    expect(html).toContain('max_weight_delta');
    expect(html).toContain('4.00%');
    expect(html).not.toContain('panel-error-boundary');
  });

  it('renders metric-only promotions as governance-blocked disclosures', () => {
    const html = renderPanel(
      'Decisions',
      <DecisionReplayPanel initialData={decisionRegistryWithGovernanceBlockedPromotion()} />,
    );

    expect(html).toContain('Governance blocked');
    expect(html).toContain('Canonical recommendation: candidate');
    expect(html).toContain('Metric gate: promoted');
    expect(html).toContain('governance: candidate');
    expect(html).toContain('provenance: missing');
    expect(html).not.toContain('<td>promoted ✓</td>');
    expect(html).not.toContain('panel-error-boundary');
  });

  it('renders unmatched promotion rows as explicit partial coverage disclosures', () => {
    const html = renderPanel(
      'Decisions',
      <DecisionReplayPanel initialData={decisionRegistryWithUnmatchedPromotion()} />,
    );

    expect(html).toContain('Not evaluated');
    expect(html).toContain('Promotion evaluation not published');
    expect(html).toContain('Partial coverage: 10/12 evaluated');
    expect(html).not.toContain('<td>—</td>');
    expect(html).not.toContain('panel-error-boundary');
  });

  it('renders no-model stacking as dormant and suppresses live prediction metrics', () => {
    const dormantStacking = {
      active: false,
      stacking_available: false,
      runtime_role: 'research_dormant',
      runtime_status: 'unavailable_no_model',
      live_authoritative: false,
      routed: false,
      routed_by: null,
      prediction_available: false,
      prediction_direction: 'unavailable',
      confidence: 0,
      probability_bullish: 0,
      probability_bearish: 0,
      probability_neutral: 0,
      fallback_used: false,
      model_version: 'unavailable_no_model',
      voting_accuracy: null,
      stacking_accuracy: null,
      accuracy_metrics_available: false,
      feature_count: null,
      feature_count_metadata_available: false,
      feature_count_source: 'unavailable_no_model',
      source_roster: [],
      source_roster_version: 'unavailable_no_model',
      fallback_semantics: 'no_model_feature_count_unavailable',
      latency_ms: 0,
      status_reason: 'No stacking model artifact is loaded and no runtime base-signal input path is available.',
      operator_message: 'Stacking ensemble is research/dormant and not order-routed.',
    } as unknown as StackingEnsembleData;

    const html = renderPanel(
      'Stacking Ensemble',
      <StackingEnsemblePanel data={dormantStacking} />,
    );

    expect(html).toContain('Research dormant');
    expect(html).toContain('No stacking model artifact is loaded');
    expect(html).toContain('not order-routed');
    expect(html).not.toContain('Current Prediction');
    expect(html).not.toContain('Directional Accuracy');
    expect(html).not.toContain('65%');
    expect(html).not.toContain('76%');
    expect(html).not.toContain('panel-error-boundary');
  });
});
