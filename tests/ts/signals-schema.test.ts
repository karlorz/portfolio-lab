import { describe, expect, it } from 'bun:test';
import {
  AdaptiveSizingSchema,
  BlackLittermanSchema,
  EnsembleVotingSchema,
  GarchCvarSchema,
  IcDecaySchema,
  MarlStatusSchema,
  FredMacroSchema,
  RegimeAuthoritySchema,
  RegimeSchema,
  SignalWFESchema,
  SignalsDataSchema,
  StackingEnsembleSchema,
  SmartRebalanceSchema,
  YieldCurveSchema,
} from '../../src/schemas/signals';

function makeBaseSignals(overrides: Record<string, unknown> = {}) {
  return {
    timestamp: '2026-07-05T00:00:00+00:00',
    regime: { regime: 'normal', vix: 16, detected: '2026-07-05T00:00:00+00:00' },
    latest_prices: { SPY: 625 },
    current_positions: [],
    target_allocations: { SPY: 0.46, GLD: 0.38, TLT: 0.16 },
    cash: 1000,
    total_value: 100000,
    recent_orders: [],
    ml_signals: {
      available: false,
      timestamp: null,
      generated_at: null,
      feature_source_artifact: null,
      feature_as_of: null,
      feature_freshness_status: 'missing',
      feature_staleness_days: null,
      prediction_source_mode: 'unavailable',
      execution_role: {
        role: 'advisory_non_routed',
        routed: false,
        routed_by: null,
        live_authoritative: false,
      },
      predictions: {},
      features: {},
      grid_search: {
        available: false,
        timestamp: null,
        top_allocation: null,
        sharpe: null,
        volatility: null,
        source_artifact: null,
        benchmark_timestamp: null,
        observation_semantics: 'unavailable',
        freshness_status: 'missing',
        staleness_days: null,
        live_authoritative: false,
      },
    },
    marl_status: {
      schema_version: 'marl-runtime-status/v1',
      available: true,
      timestamp: '2026-07-05T00:00:00+00:00',
      runtime: {
        version: '2.51.0',
        device: 'cpu',
        agents_loaded: ['analyst', 'sentiment', 'risk', 'execution', 'controller'],
        signal_integrator_connected: false,
        checkpoint_loaded: false,
        inference_count: 0,
        current_allocation: { SPY: 0.46, GLD: 0.38, TLT: 0.16, CASH: 0 },
        graph_metrics: { messages_routed: 0 },
      },
      execution_role: {
        role: 'research_shadow_non_routed',
        routed: false,
        routed_by: null,
        live_authoritative: false,
        description: 'MARL status is visible for research/shadow diagnostics; order routing still consumes target_allocations.',
      },
    },
    allocation_surface_roles: {
      schema_version: 'allocation-surface-roles/v1',
      routed_surface: 'target_allocations',
      routed_by: 'src.broker.order_router',
      surfaces: {
        target_allocations: {
          label: 'Target Allocation',
          role: 'execution_routed',
          routed: true,
          routed_by: 'src.broker.order_router',
          description: 'Current order-routing input.',
        },
        ensemble_voting: {
          label: 'Ensemble Voting',
          role: 'advisory_non_routed',
          routed: false,
          routed_by: null,
          live_authoritative: false,
          canonical_controller: 'signals.json.target_allocations',
          description: 'Diagnostic ensemble output, not order-routed.',
        },
        adaptive_sizing: {
          label: 'Adaptive Sizing',
          role: 'advisory_non_routed',
          routed: false,
          routed_by: null,
          live_authoritative: false,
          canonical_controller: 'signals.json.target_allocations',
          description: 'Adaptive sizing output is advisory; live routing uses target_allocations.',
        },
        black_litterman: {
          label: 'Black-Litterman',
          role: 'advisory_non_routed',
          routed: false,
          routed_by: null,
          live_authoritative: false,
          canonical_controller: 'signals.json.target_allocations',
          description: 'Black-Litterman output is advisory; live routing uses target_allocations.',
        },
      },
    },
    regime_authority: {
      schema_version: 'regime-authority/v1',
      live_controller: 'classify_vix_regime',
      live_controller_module: 'src.utils.classify_vix_regime',
      live_regime: 'vol_spike',
      allocation_regime: 'high_vol',
      routed_surface: 'target_allocations',
      target_allocations: { SPY: 0.38, GLD: 0.42, TLT: 0.20 },
      advanced_regime_signals: {
        two_stage_regime: { role: 'advisory_shadow', routed: false },
        bocd_regime: { role: 'advisory_shadow', routed: false },
        regime_transition: { role: 'advisory_shadow', routed: false },
      },
    },
    ...overrides,
  };
}

describe('signals.ts Zod schemas (non-ML contract)', () => {
  it('RegimeSchema accepts minimal regime payload', () => {
    const r = RegimeSchema.safeParse({ regime: 'NORMAL', vix: 18.2, detected: '2026-07-01' });
    expect(r.success).toBe(true);
  });

  it('GarchCvarSchema rejects missing required tail fields', () => {
    const bad = GarchCvarSchema.safeParse({ cvar_95: -0.02 });
    expect(bad.success).toBe(false);
  });

  it('GarchCvarSchema accepts full garch_cvar panel', () => {
    expect(Object.keys(GarchCvarSchema.shape)).toContain('coverage_diagnostics');

    const good = GarchCvarSchema.safeParse({
      cvar_95: -0.04,
      cvar_95_garch: -0.0215,
      var_95: -0.69,
      var_95_garch: -0.0142,
      cvar_ratio: 1.51,
      garch_active: true,
      current_volatility: 0.0054,
      forecast_volatility: 0.015,
      volatility_clustering: 'elevated',
      coverage_diagnostics: {
        schema_version: 'conformal-coverage/v1',
        observations: 500,
        alpha: 0.05,
        expected_exceedance_rate: 0.05,
        exceedance_count: 25,
        exceedance_rate: 0.05,
        coverage_rate: 0.95,
        coverage_pass: true,
        rolling_window: 252,
        rolling_exceedance_rate: 0.0476,
        longest_violation_cluster: 1,
        kupiec_statistic: 0,
        kupiec_p_value: 1,
        kupiec_pass: true,
        christoffersen_statistic: 0,
        christoffersen_p_value: 1,
        christoffersen_pass: true,
        conditional_coverage_statistic: 0,
        conditional_coverage_p_value: 1,
        conditional_coverage_pass: true,
      },
    });
    expect(good.success).toBe(true);
  });

  it('SmartRebalanceSchema accepts defer hold decision', () => {
    const good = SmartRebalanceSchema.safeParse({
      should_execute: false,
      decision: 'defer',
      urgency: 'moderate',
      max_drift: 0.03,
      estimated_cost_bps: 12,
      reason: 'vpin_high',
      drift_details: { SPY: 0.02 },
      vpin: 0.8,
      in_optimal_window: true,
      ytd_cost_bps: 5,
      remaining_budget_pct: 0.9,
      status: {
        ytd_cost_bps: 5,
        ytd_cost_pct: 0.0005,
        remaining_budget_pct: 0.9,
        is_over_budget: false,
        is_warning: false,
        last_rebalance: null,
        deferred_until: null,
        config: {
          drift_threshold: 0.1,
          vpin_threshold: 0.7,
          optimal_window: '10:00-15:30',
          annual_cost_limit: '50bps',
        },
      },
    });
    expect(good.success).toBe(true);
  });

  it('YieldCurveSchema allows nullable spreads', () => {
    expect(YieldCurveSchema.safeParse({ spread2s10s: null, dgs2: null, dgs10: null, duration_regime: null }).success).toBe(
      true,
    );
  });

  it('YieldCurveSchema keeps compact source metadata for display badges', () => {
    const parsed = YieldCurveSchema.safeParse({
      spread2s10s: -15,
      dgs2: 4.6,
      dgs10: 4.45,
      duration_regime: 'inverted',
      source_mode: 'synthetic',
      source_status: 'degraded',
      source_reason: 'using fallback Treasury curve',
    });
    expect(parsed.success).toBe(true);
    if (!parsed.success) throw new Error('expected yield provenance to parse');
    expect(parsed.data.source_mode).toBe('synthetic');
    expect(parsed.data.source_status).toBe('degraded');
  });

  it('FredMacroSchema accepts unavailable readiness metadata', () => {
    const parsed = FredMacroSchema.safeParse({
      regime: 'UNKNOWN',
      confidence: 0,
      recession_probability: 0,
      inflation_pressure: 0,
      monetary_stance: 'unknown',
      manufacturing_health: 50,
      credit_conditions: 'unknown',
      indicators: {},
      timestamp: '2026-07-06T00:00:00Z',
      source_mode: 'unavailable',
      cache_status: 'empty',
      api_key_configured: false,
      indicators_observed: false,
      reason: 'empty_cache',
    });
    expect(parsed.success).toBe(true);
    if (!parsed.success) throw new Error('expected FRED metadata to parse');
    expect(parsed.data.source_mode).toBe('unavailable');
  });

  it('IcDecaySchema accepts empty signals map', () => {
    const good = IcDecaySchema.safeParse({ signals: {} });
    expect(good.success).toBe(true);
  });

  it('IcDecaySchema accepts pending forward-return label state', () => {
    const good = IcDecaySchema.safeParse({
      status: 'waiting_for_forward_returns',
      signals: {},
      resolved_signal_count: 0,
      pending_predictions: 2,
      staged_date: '2026-07-02',
      label_horizon: 'SPY close-to-close forward return',
    });
    expect(good.success).toBe(true);
  });

  it('SignalWFESchema accepts pending resolved-history state', () => {
    const good = SignalWFESchema.safeParse({
      status: 'waiting_for_forward_returns',
      signals: {},
      resolved_signal_count: 0,
      pending_predictions: 1,
      staged_date: '2026-07-02',
      label_horizon: 'Uses resolved IC prediction/forward-return pairs',
    });
    expect(good.success).toBe(true);
  });

  it('StackingEnsembleSchema accepts unavailable feature count with provenance', () => {
    const good = StackingEnsembleSchema.safeParse({
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
    });

    expect(good.success).toBe(true);
    if (good.success) {
      expect(good.data.feature_count).toBeNull();
      expect(good.data.feature_count_metadata_available).toBe(false);
      expect(good.data.feature_count_source).toBe('unavailable_no_model');
      expect(good.data.source_roster).toEqual([]);
      expect(good.data.source_roster_version).toBe('unavailable_no_model');
      expect(good.data.fallback_semantics).toBe('no_model_feature_count_unavailable');
    }
  });

  it('StackingEnsembleSchema requires dormant no-model runtime disclosure', () => {
    expect(Object.keys(StackingEnsembleSchema.shape)).toContain('runtime_role');
    expect(Object.keys(StackingEnsembleSchema.shape)).toContain('runtime_status');
    expect(Object.keys(StackingEnsembleSchema.shape)).toContain('live_authoritative');
    expect(Object.keys(StackingEnsembleSchema.shape)).toContain('routed');
    expect(Object.keys(StackingEnsembleSchema.shape)).toContain('routed_by');
    expect(Object.keys(StackingEnsembleSchema.shape)).toContain('prediction_available');
    expect(Object.keys(StackingEnsembleSchema.shape)).toContain('accuracy_metrics_available');
    expect(Object.keys(StackingEnsembleSchema.shape)).toContain('status_reason');
    expect(Object.keys(StackingEnsembleSchema.shape)).toContain('operator_message');

    const good = StackingEnsembleSchema.safeParse({
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
    });

    expect(good.success).toBe(true);
    if (good.success) {
      expect(good.data.active).toBe(false);
      expect(good.data.prediction_available).toBe(false);
      expect(good.data.voting_accuracy).toBeNull();
      expect(good.data.stacking_accuracy).toBeNull();
      expect(good.data.accuracy_metrics_available).toBe(false);
    }
  });

  it('MarlStatusSchema models runtime status and non-routed execution role', () => {
    const good = MarlStatusSchema.safeParse({
      schema_version: 'marl-runtime-status/v1',
      available: true,
      timestamp: '2026-07-05T00:00:00+00:00',
      runtime: {
        version: '2.51.0',
        device: 'cpu',
        agents_loaded: ['analyst', 'sentiment', 'risk', 'execution', 'controller'],
        signal_integrator_connected: false,
        checkpoint_loaded: false,
        inference_count: 0,
        current_allocation: { SPY: 0.46, GLD: 0.38, TLT: 0.16, CASH: 0 },
        graph_metrics: { messages_routed: 0, alerts_triggered: 0 },
      },
      execution_role: {
        role: 'research_shadow_non_routed',
        routed: false,
        routed_by: null,
        live_authoritative: false,
        description: 'MARL status is visible for research/shadow diagnostics; order routing still consumes target_allocations.',
      },
    });

    expect(good.success).toBe(true);
    if (good.success) {
      expect(good.data.execution_role.routed).toBe(false);
      expect(good.data.execution_role.role).toBe('research_shadow_non_routed');
      expect(good.data.runtime.agents_loaded).toContain('controller');
    }
  });

  it('AdaptiveSizingSchema requires advisory allocation authority metadata', () => {
    const good = AdaptiveSizingSchema.safeParse({
      base_allocation: { SPY: 0.46, GLD: 0.38, TLT: 0.16 },
      adjusted_allocation: { SPY: 0.44, GLD: 0.40, TLT: 0.16 },
      adjustments: { SPY: -0.02, GLD: 0.02, TLT: 0 },
      authority: {
        schema_version: 'allocation-artifact-role/v1',
        surface: 'adaptive_sizing',
        allocation_field: 'adjusted_allocation',
        runtime_role: 'advisory_non_routed',
        live_authoritative: false,
        routed: false,
        routed_by: null,
        canonical_controller: 'signals.json.target_allocations',
        routed_surface: 'target_allocations',
        routed_surface_path: 'public/data/signals.json#target_allocations',
        description: 'adaptive_sizing is advisory; live order routing continues to consume signals.json.target_allocations.',
      },
      generated_at: '2026-07-06T00:00:00Z',
    });

    expect(good.success).toBe(true);
    expect(AdaptiveSizingSchema.safeParse({
      adjusted_allocation: { SPY: 1 },
      generated_at: '2026-07-06T00:00:00Z',
    }).success).toBe(false);
  });

  it('AdaptiveSizingSchema accepts producer per-asset adjustment maps', () => {
    // public/data/adaptive_sizing.json emits per-symbol adjustment maps, not scalars.
    const good = AdaptiveSizingSchema.safeParse({
      base_allocation: { SPY: 0.46, GLD: 0.38, TLT: 0.16 },
      adjusted_allocation: { SPY: 0.46, GLD: 0.38, TLT: 0.16 },
      adjustments: { SPY: 0.0, GLD: 0.0, TLT: 0.0 },
      regime_adjustment: { SPY: 0.0, GLD: 0.0, TLT: 0.0 },
      volatility_adjustment: { SPY: 0.0, GLD: 0.0, TLT: 0.0 },
      signal_adjustment: { SPY: 0.0, GLD: 0.0, TLT: 0.0 },
      drawdown_adjustment: { SPY: 0.0, GLD: 0.0, TLT: 0.0 },
      factors: {
        timestamp: '2026-07-12T03:57:50.292076',
        regime: 'normal',
        regime_confidence: 0.3,
      },
      authority: {
        schema_version: 'allocation-artifact-role/v1',
        surface: 'adaptive_sizing',
        allocation_field: 'adjusted_allocation',
        runtime_role: 'advisory_non_routed',
        live_authoritative: false,
        routed: false,
        routed_by: null,
        canonical_controller: 'signals.json.target_allocations',
        routed_surface: 'target_allocations',
        routed_surface_path: 'public/data/signals.json#target_allocations',
        description: 'adaptive_sizing is advisory; live order routing continues to consume signals.json.target_allocations.',
      },
      generated_at: '2026-07-12T03:57:50.292076',
    });
    expect(good.success).toBe(true);
  });

  it('BlackLittermanSchema requires authority metadata and uppercase symbol keys', () => {
    const good = BlackLittermanSchema.safeParse({
      prior_weights: { SPY: 0.46, GLD: 0.38, TLT: 0.16 },
      posterior_weights: { SPY: 0.44, GLD: 0.40, TLT: 0.16 },
      posterior_returns: { SPY: 0.08, GLD: 0.04, TLT: 0.03 },
      views: [{
        signal_name: 'ensemble_consensus',
        asset: 'SPY',
        direction: 'bullish',
        confidence: 0.7,
        expected_return_delta: 0.01,
      }],
      tau: 0.15,
      view_confidence_method: 'idzorek',
      excluded_assets: [],
      zero_weight_assets: [],
      authority: {
        schema_version: 'allocation-artifact-role/v1',
        surface: 'black_litterman',
        allocation_field: 'posterior_weights',
        runtime_role: 'advisory_non_routed',
        live_authoritative: false,
        routed: false,
        routed_by: null,
        canonical_controller: 'signals.json.target_allocations',
        routed_surface: 'target_allocations',
        routed_surface_path: 'public/data/signals.json#target_allocations',
        description: 'black_litterman is advisory; live order routing continues to consume signals.json.target_allocations.',
      },
      generated_at: '2026-07-06T00:00:00Z',
    });

    expect(good.success).toBe(true);
    const validPayload = good.success ? good.data : {};
    expect(BlackLittermanSchema.safeParse({
      ...validPayload,
      posterior_weights: { spy: 0.44, GLD: 0.40, TLT: 0.16 },
    }).success).toBe(false);
  });

  it('MarlStatusSchema accepts generated nullable error field', () => {
    const good = MarlStatusSchema.safeParse({
      schema_version: 'marl-runtime-status/v1',
      available: true,
      timestamp: '2026-07-05T00:00:00+00:00',
      runtime: {
        version: '2.51.0',
        device: 'cpu',
        agents_loaded: ['analyst', 'sentiment', 'risk', 'execution', 'controller'],
        signal_integrator_connected: false,
        checkpoint_loaded: false,
        inference_count: 0,
        current_allocation: { SPY: 0.46, GLD: 0.38, TLT: 0.16, CASH: 0 },
        graph_metrics: { messages_routed: 0, alerts_triggered: 0 },
      },
      execution_role: {
        role: 'research_shadow_non_routed',
        routed: false,
        routed_by: null,
        live_authoritative: false,
        description: 'MARL status is visible for research/shadow diagnostics; order routing still consumes target_allocations.',
      },
      error: null,
    });

    expect(good.success).toBe(true);
  });

  it('EnsembleVotingSchema explicitly models source count semantics', () => {
    expect(Object.keys(EnsembleVotingSchema.shape)).toContain('configured_source_count');
    expect(Object.keys(EnsembleVotingSchema.shape)).toContain('collected_source_count');
    expect(Object.keys(EnsembleVotingSchema.shape)).toContain('contributing_source_count');
    expect(Object.keys(EnsembleVotingSchema.shape)).toContain('inactive_source_count');
    expect(Object.keys(EnsembleVotingSchema.shape)).toContain('inactive_sources');
    expect(Object.keys(EnsembleVotingSchema.shape)).toContain('configured_source_status');

    const good = EnsembleVotingSchema.safeParse({
      regime: 'normal',
      regime_confidence: 0.63,
      weighted_consensus: 0,
      agreement_ratio: 1,
      action: 'neutral',
      confidence: 0.5,
      equity_bias: 0.1,
      duration_bias: -0.1,
      gold_bias: 0.05,
      num_sources: 4,
      configured_source_count: 9,
      collected_source_count: 4,
      contributing_source_count: 2,
      inactive_source_count: 2,
      inactive_sources: ['cross_asset_rv', 'multi_speed_momentum'],
      configured_source_status: [{
        source: 'google_trends',
        label: 'Google Trends',
        configured: true,
        collected: false,
        active: false,
        contributing: false,
        status: 'stale',
        reason: 'Data is 37 days old (max 14)',
        configured_weight: 0.04762,
      }],
      source_breakdown: [],
    });

    expect(good.success).toBe(true);
    if (good.success) {
      expect(good.data.configured_source_count).toBe(9);
      expect(good.data.collected_source_count).toBe(4);
      expect(good.data.contributing_source_count).toBe(2);
      expect(good.data.inactive_source_count).toBe(2);
      expect(good.data.inactive_sources).toEqual(['cross_asset_rv', 'multi_speed_momentum']);
      expect(good.data.configured_source_status?.[0]?.source).toBe('google_trends');
      expect(good.data.configured_source_status?.[0]?.status).toBe('stale');
    }
  });

  it('SignalsDataSchema rejects malformed active panel objects', () => {
    const invalidPanels = [
      'crypto_allocation',
      'calendar_seasonality',
      'ensemble_voting',
      'alternative_data',
      'factor_rotation',
      'stacking_ensemble',
      'convexity_harvest',
      'llm_sentiment',
      'sector_rotation',
      'factor_rotation_dashboard',
      'collar',
      'kurtosis_regime',
      'volatility_parity',
    ];

    for (const panel of invalidPanels) {
      const parsed = SignalsDataSchema.safeParse(makeBaseSignals({
        [panel]: { active: true },
      }));
      expect(parsed.success).toBe(false);
    }
  });

  it('SignalsDataSchema preserves ML source freshness and frozen benchmark metadata', () => {
    const parsed = SignalsDataSchema.safeParse(makeBaseSignals({
      ml_signals: {
        available: true,
        timestamp: '2026-07-06T12:00:00+00:00',
        generated_at: '2026-07-06T12:00:00+00:00',
        feature_source_artifact: 'features.jsonl',
        feature_as_of: '2026-05-08T00:00:00+00:00',
        feature_freshness_status: 'stale',
        feature_staleness_days: 59,
        prediction_source_mode: 'stale_features',
        execution_role: {
          role: 'advisory_non_routed',
          routed: false,
          routed_by: null,
          live_authoritative: false,
        },
        predictions: {
          SPY: {
            predicted_regime: 'neutral',
            confidence: 0.6,
            probabilities: { bear: 0.2, neutral: 0.6, bull: 0.2 },
            heuristic: true,
            feature_timestamp: '2026-05-08T00:00:00+00:00',
            feature_freshness_status: 'stale',
            source_artifact: 'features.jsonl',
          },
        },
        features: {
          SPY: {
            vix_level: 18,
            trend_direction: 0,
            price_vs_sma20: 0,
            return_5d: 0,
            spy_correlation: 0.2,
            feature_timestamp: '2026-05-08T00:00:00+00:00',
          },
        },
        grid_search: {
          available: true,
          timestamp: '2026-05-22T00:00:00+00:00',
          top_allocation: { SPY: 0.46, GLD: 0.38, TLT: 0.16 },
          sharpe: 0.95,
          volatility: 0.11,
          source_artifact: 'grid_search_results.jsonl',
          benchmark_timestamp: '2026-05-22T00:00:00+00:00',
          observation_semantics: 'frozen_benchmark_not_live_snapshot',
          freshness_status: 'frozen_benchmark',
          staleness_days: 45,
          live_authoritative: false,
        },
      },
    }));

    expect(parsed.success).toBe(true);
    if (!parsed.success) throw new Error('schema should parse');
    expect(parsed.data.ml_signals.feature_freshness_status).toBe('stale');
    expect(parsed.data.ml_signals.predictions.SPY.feature_timestamp).toBe('2026-05-08T00:00:00+00:00');
    expect(parsed.data.ml_signals.grid_search.observation_semantics).toBe('frozen_benchmark_not_live_snapshot');
  });

  it('SignalsDataSchema rejects malformed allocation surface role disclosure', () => {
    const parsed = SignalsDataSchema.safeParse(makeBaseSignals({
      allocation_surface_roles: {
        schema_version: 'allocation-surface-roles/v1',
        routed_surface: 123,
        routed_by: 'src.broker.order_router',
        surfaces: {},
      },
    }));

    expect(parsed.success).toBe(false);
  });

  it('SignalsDataSchema rejects malformed regime authority disclosure', () => {
    const parsed = SignalsDataSchema.safeParse(makeBaseSignals({
      regime_authority: {
        schema_version: 'regime-authority/v1',
        live_controller: 'classify_vix_regime',
        live_regime: 'vol_spike',
        allocation_regime: 'high_vol',
        routed_surface: 'target_allocations',
        target_allocations: { SPY: 0.38, GLD: 0.42, TLT: 0.20 },
        advanced_regime_signals: {
          two_stage_regime: { role: 'advisory_shadow', routed: 'no' },
        },
      },
    }));

    expect(parsed.success).toBe(false);
  });

  it('RegimeAuthoritySchema requires boolean published availability disclosure', () => {
    const base = {
      schema_version: 'regime-authority/v1',
      live_controller: 'classify_vix_regime',
      live_controller_module: 'src.utils.classify_vix_regime',
      live_regime: 'normal',
      allocation_regime: 'normal',
      routed_surface: 'target_allocations',
      target_allocations: { SPY: 0.46, GLD: 0.38, TLT: 0.16 },
      advanced_regime_signals: {
        two_stage_regime: {
          role: 'advisory_shadow',
          routed: false,
          availability: 'unavailable',
          published: false,
        },
        bocd_regime: {
          role: 'advisory_shadow',
          routed: false,
          availability: 'present',
          published: true,
        },
        regime_transition: {
          role: 'advisory_shadow',
          routed: false,
          availability: 'stale',
          published: false,
        },
      },
    };

    const good = RegimeAuthoritySchema.safeParse(base);
    const bad = RegimeAuthoritySchema.safeParse({
      ...base,
      advanced_regime_signals: {
        ...base.advanced_regime_signals,
        two_stage_regime: {
          ...base.advanced_regime_signals.two_stage_regime,
          published: 'false',
        },
      },
    });

    expect(good.success).toBe(true);
    expect(bad.success).toBe(false);
  });

  it('SignalsDataSchema accepts representative current active panel payloads', () => {
    const good = SignalsDataSchema.safeParse(makeBaseSignals({
      crypto_allocation: {
        active: true,
        btc_weight: 0.15,
        eth_weight: 0.25,
        total_crypto: 0.012,
        btc_momentum_6m: 0.22,
        eth_momentum_6m: 0.25,
        btc_vol_regime: 'high',
        eth_vol_regime: 'normal',
        confidence: 55,
      },
      calendar_seasonality: {
        active: false,
        modifier: 1,
        active_windows: [],
        next_window: 'tom_window',
        days_to_next: 3,
        recommendation: 'proceed',
        effect: 'neutral',
      },
      ensemble_voting: {
        regime: 'normal',
        regime_confidence: 0.63,
        weighted_consensus: 0,
        agreement_ratio: 1,
        action: 'neutral',
        confidence: 0.5,
        equity_bias: 0.1,
        duration_bias: -0.1,
        gold_bias: 0.05,
        num_sources: 1,
        configured_source_count: 9,
        collected_source_count: 1,
        contributing_source_count: 1,
        inactive_source_count: 0,
        inactive_sources: [],
        adaptive_learning: {
          bandit: {
            status: 'non_effective',
            enabled: true,
            observations: 0,
            warmup_days: 252,
            max_blend: 0.7,
            current_blend: 0,
            reason: 'cold_start_no_regime_weights',
          },
          online_ic: {
            status: 'disabled',
            enabled: false,
            state_available: false,
            blend_alpha: 0.3,
            reason: 'env_disabled',
          },
        },
        source_breakdown: [{
          source: 'alternative_data',
          direction: 'bullish',
          strength: 0.4,
          confidence: 0.6,
          weight: 0.2,
        }],
      },
      alternative_data: {
        regime: 'bull',
        probability: 0.9,
        confidence: 0.6,
        timestamp: '2026-07-05T00:00:00+00:00',
        components: {
          news: { score: null, confidence: null, weight: null },
        },
        composite_score: 0.4,
        z_score: 1.2,
        sources_count: 7,
        data_freshness_hours: 12,
      },
      factor_rotation: {
        selected_factors: ['VLUE', 'VBR'],
        allocation: { VLUE: 0.27, VBR: 0.73 },
        signal_strength: 0.53,
        recommendation: 'Rotate to Value',
      },
      stacking_ensemble: {
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
      },
      convexity_harvest: {
        date: '2026-07-03',
        allocation_pct: 0,
        position_type: 'flat',
        vix_level: 0,
        contango_pct: 0,
        expected_roll_yield: 0,
        risk_score: 1,
        exit_triggered: false,
        exit_reason: null,
      },
      llm_sentiment: {
        timestamp: '2026-07-05T00:00:00+00:00',
        technical_regime: 'normal',
        technical_confidence: 0.6,
        sentiment_regime: 'neutral',
        sentiment_confidence: 0,
        combined_score: 0,
        combined_regime: 'neutral',
        technical_weight: 0.7,
        sentiment_weight: 0.3,
        circuit_breaker_level: 'yellow',
        position_scaling_factor: 0.85,
        equity_tilt: 0,
        bond_duration_tilt: 0,
        gold_tilt: 0,
      },
      sector_rotation: {
        timestamp: '2026-07-05T00:00:00+00:00',
        status: 'active',
        top_sectors: [{ symbol: 'XLK', name: 'Technology', momentumScore: 0.4, allocation: 0.03, rank: 1 }],
      },
      factor_rotation_dashboard: {
        active: true,
        selected_factors: ['VLUE'],
        signal_strength: 0.53,
        factor_allocations: { VLUE: 0.27 },
        backtest_finding: 'Defensive tool',
      },
      collar: {
        active: true,
        regime: 'normal',
        call_strike: 566,
        put_strike: 529,
        net_premium: 1.7,
        is_cashless: false,
        max_upside_pct: 3.0,
        max_downside_pct: 3.6,
        vix_level: 16,
        confidence: 60,
      },
      kurtosis_regime: {
        active: true,
        kurtosis_20d: 1.7,
        kurtosis_60d: 3.1,
        ker_ratio: 0.56,
        regime: 'low_kurtosis',
        transitioning: true,
        strategy_preference: 'balanced',
        tsom_weight: 0.74,
        mr_weight: 0.26,
        fat_tail_risk: 0.046,
      },
      volatility_parity: {
        date: '2026-07-03',
        target_volatility: 10,
        spy_pct: 40,
        gld_pct: 28,
        tlt_pct: 12,
        core_vol_contribution: 11.36,
        vix_short_pct: 0,
        vix_tail_pct: 2,
        vix_vol_contribution: -30,
        cash_pct: 18,
        expected_portfolio_vol: 11.66,
        expected_max_dd: 17.49,
        rebalance_triggered: false,
        rebalance_reason: null,
      },
    }));

    expect(good.success).toBe(true);
    if (good.success) {
      expect(good.data.allocation_surface_roles.routed_surface).toBe('target_allocations');
      expect(good.data.allocation_surface_roles.surfaces.ensemble_voting.role).toBe('advisory_non_routed');
      expect(good.data.regime_authority?.allocation_regime).toBe('high_vol');
      expect(good.data.regime_authority?.advanced_regime_signals.bocd_regime.routed).toBe(false);
      expect(good.data.ensemble_voting?.adaptive_learning?.bandit.status).toBe('non_effective');
      expect(good.data.ensemble_voting?.adaptive_learning?.online_ic.status).toBe('disabled');
    }
  });
});
