import { describe, expect, it } from 'bun:test';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { ActionCenter } from '../../src/components/control-plane/ActionCenter';
import {
  buildDashboardIncidents,
  buildDecisionIncidents,
  buildRiskIncidents,
  getIncidentsForTab,
  getTabIncidentBadge,
} from '../../src/components/dashboardIncidents';
import type { Alert, HealthData, IncidentLifecycleSummary, SignalsData } from '../../src/types/live';

function signalsWithCvar(cvarRatio: number): SignalsData {
  return {
    timestamp: '2026-06-11T12:32:03.929282',
    garch_cvar: {
      cvar_95: -0.04,
      cvar_95_garch: -0.0215,
      var_95: -0.69,
      var_95_garch: -0.0142,
      cvar_ratio: cvarRatio,
      garch_active: true,
      current_volatility: 0.0054,
      forecast_volatility: 0.015,
      volatility_clustering: 'elevated',
      conformal_cvar_95: -0.033468,
      conformal_var_95: -0.020799,
      conformal_cvar_ratio: 1.609,
    },
  } as SignalsData;
}

function alert(overrides: Partial<Alert>): Alert {
  return {
    level: 'info',
    type: 'generic',
    title: 'Alert',
    message: 'Needs attention',
    timestamp: '2026-06-11T12:32:14.473414',
    requires_action: true,
    ...overrides,
  };
}

function health(status: HealthData['system_status']): HealthData {
  return {
    system_status: status,
    generated_at: '2026-06-11T12:32:14.473414',
    cron_jobs: [],
    data_freshness: {},
  };
}

function incidentSummary(overrides: Partial<IncidentLifecycleSummary> = {}): IncidentLifecycleSummary {
  return {
    generated_at: '2026-06-11T12:35:00+00:00',
    open_count: 1,
    metrics: {
      incident_frequency: 2,
      open_count: 1,
      resolved_count: 1,
      mean_mttr_seconds: 1800,
    },
    incidents: [
      {
        incident_id: 'inc-001',
        channel: 'cron_failure',
        severity: 'p0',
        state: 'firing',
        message: 'Scheduler backends disagree for 2 consecutive checks',
        details: { consecutive_mismatches: 2 },
        created_at: '2026-06-11T12:30:00+00:00',
        updated_at: '2026-06-11T12:34:00+00:00',
        resolved_at: null,
        resolution_notes: null,
        mttr_seconds: null,
      },
    ],
    ...overrides,
  };
}

describe('dashboard incident derivation', () => {
  it('maps a live missing-title kill alert defensively and prefers its incident id', () => {
    const incidents = buildDashboardIncidents({
      alerts: [{
        level: 'warning',
        type: 'kill_switch',
        message: 'New orders are blocked pending operator review.',
        requires_action: true,
        incident_id: 'kill-incident-42',
        stable_id: 'kill-incident-42',
      } as Alert],
      signals: null,
      health: null,
    });

    expect(incidents).toHaveLength(1);
    expect(incidents[0]).toMatchObject({
      id: 'alert:kill-incident-42',
      severity: 'critical',
      title: 'Kill Switch',
      source: 'Kill switch',
      message: 'New orders are blocked pending operator review.',
      nextAction: 'Review kill-switch state before placing new orders.',
    });
  });

  it('keeps producer-shaped critical alerts as critical dashboard incidents', () => {
    const incidents = buildDashboardIncidents({
      alerts: [alert({
        level: 'critical' as Alert['level'],
        type: 'ic_decay',
        title: 'IC decay requires review',
        message: 'ensemble_duration and ensemble_consensus are below the IC gate.',
      })],
      signals: null,
      health: null,
    });

    expect(incidents).toHaveLength(1);
    expect(incidents[0]).toMatchObject({
      severity: 'critical',
      source: 'IC decay monitor',
    });
  });

  it('derives a critical Risk incident when GARCH CVaR ratio is severe', () => {
    const incidents = buildRiskIncidents(signalsWithCvar(3));

    expect(incidents).toHaveLength(1);
    expect(incidents[0]).toMatchObject({
      id: 'risk:garch-cvar:severe-tail-risk',
      tab: 'risk',
      severity: 'critical',
      title: 'Severe Tail Risk',
      source: 'GARCH CVaR',
      currentValue: '3.00x CVaR/VaR',
      threshold: '>= 1.80x severe, >= 1.50x warning',
      timestamp: '2026-06-11T12:32:03.929282',
    });
    expect(incidents[0]?.nextAction).toContain('Review equity exposure');
  });

  it('derives a warning Risk incident for elevated but not severe CVaR ratio', () => {
    const incidents = buildRiskIncidents(signalsWithCvar(1.6));

    expect(incidents).toHaveLength(1);
    expect(incidents[0]).toMatchObject({
      id: 'risk:garch-cvar:elevated-tail-risk',
      tab: 'risk',
      severity: 'warning',
      title: 'Elevated Tail Risk',
      currentValue: '1.60x CVaR/VaR',
    });
  });

  it('does not derive a Risk incident below the warning threshold', () => {
    expect(buildRiskIncidents(signalsWithCvar(1.49))).toEqual([]);
    expect(buildRiskIncidents(null)).toEqual([]);
  });

  it('maps success alerts that require action to info incidents, not critical incidents', () => {
    const incidents = buildDashboardIncidents({
      alerts: [
        alert({
          level: 'success',
          type: 'graduation_candidate',
          title: 'Paper Trading Graduation Ready',
          message: 'Sharpe: 0.86, ready for live approval',
          requires_action: true,
        }),
      ],
      signals: null,
      health: null,
    });

    expect(incidents).toHaveLength(1);
    expect(incidents[0]).toMatchObject({
      tab: 'overview',
      severity: 'info',
      title: 'Paper Trading Graduation Ready',
      source: 'Graduation checklist',
      currentValue: 'Sharpe 0.86',
      nextAction: 'Review graduation checklist before live approval.',
    });
  });

  it('does not turn non-action success alerts into badge incidents', () => {
    const incidents = buildDashboardIncidents({
      alerts: [
        alert({
          level: 'success',
          type: 'heartbeat',
          title: 'Heartbeat OK',
          message: 'All good',
          requires_action: false,
        }),
      ],
      signals: null,
      health: null,
    });

    expect(incidents).toEqual([]);
  });

  it('does not turn non-action info alerts into badge incidents', () => {
    const incidents = buildDashboardIncidents({
      alerts: [
        alert({
          level: 'info',
          type: 'heartbeat',
          title: 'Heartbeat OK',
          message: 'All good',
          requires_action: false,
        }),
      ],
      signals: null,
      health: null,
    });

    expect(incidents).toEqual([]);
  });

  it('keeps non-action provider warnings visible as advisory without action count', () => {
    const incidents = buildDashboardIncidents({
      alerts: [
        alert({
          level: 'warning',
          type: 'provider_cached',
          title: 'Provider data cached',
          message: 'Signal breadth is degraded: 1/9 healthy.',
          requires_action: false,
        }),
      ],
      signals: null,
      health: null,
    });

    expect(incidents).toHaveLength(1);
    expect(incidents[0]).toMatchObject({
      attention: 'advisory',
      severity: 'warning',
      title: 'Provider data cached',
    });
    expect(getTabIncidentBadge(incidents, 'overview')).toEqual({ count: 1, severity: 'warning' });

    const html = renderToStaticMarkup(React.createElement(ActionCenter, { incidents }));
    expect(html).toContain('aria-label="0 actions required"');
    expect(html).toContain('advisory');
    expect(html).toContain('Provider data cached');
  });

  it('aggregates Overview incidents across tabs and reports badge count by highest severity', () => {
    const incidents = buildDashboardIncidents({
      alerts: [
        alert({
          level: 'success',
          type: 'graduation_candidate',
          title: 'Paper Trading Graduation Ready',
          message: 'Sharpe: 0.86, ready for live approval',
          requires_action: true,
        }),
      ],
      signals: signalsWithCvar(3),
      health: health('healthy'),
    });

    expect(getIncidentsForTab(incidents, 'overview').map((incident) => incident.id)).toEqual([
      'risk:garch-cvar:severe-tail-risk',
      'alert:graduation_candidate:paper-trading-graduation-ready',
    ]);
    expect(getTabIncidentBadge(incidents, 'overview')).toEqual({ count: 2, severity: 'critical' });
    expect(getTabIncidentBadge(incidents, 'risk')).toEqual({ count: 1, severity: 'critical' });
  });

  it('derives Health tab warning and critical incidents from health status', () => {
    const warning = buildDashboardIncidents({ alerts: [], signals: null, health: health('degraded') });
    const critical = buildDashboardIncidents({ alerts: [], signals: null, health: health('critical') });

    expect(warning[0]).toMatchObject({
      id: 'health:system:degraded',
      tab: 'health',
      severity: 'warning',
      source: 'Health check',
    });
    expect(critical[0]).toMatchObject({
      id: 'health:system:critical',
      tab: 'health',
      severity: 'critical',
      source: 'Health check',
    });
  });

  it('renders persisted open incident lifecycle records as dashboard incidents', () => {
    const incidents = buildDashboardIncidents({
      alerts: [],
      signals: null,
      health: null,
      incidentSummary: incidentSummary(),
    });

    expect(incidents).toHaveLength(1);
    expect(incidents[0]).toMatchObject({
      id: 'persisted:cron_failure:inc-001',
      tab: 'health',
      severity: 'critical',
      title: 'Cron Failure Incident',
      source: 'Incident lifecycle',
      currentValue: 'State: firing',
      threshold: 'Severity: p0',
      message: 'Scheduler backends disagree for 2 consecutive checks',
      timestamp: '2026-06-11T12:34:00+00:00',
    });
    expect(incidents[0]?.nextAction).toContain('Health');
    expect(getTabIncidentBadge(incidents, 'overview')).toEqual({ count: 1, severity: 'critical' });
    expect(getTabIncidentBadge(incidents, 'health')).toEqual({ count: 1, severity: 'critical' });
  });

  it('deduplicates the persisted IC incident and the derived quality condition', () => {
    const signals = {
      timestamp: '2026-08-01T09:16:18Z',
      ic_decay: {
        status: 'critical',
        signals: {
          ensemble_duration: { ic_rolling: 0.0045, ic_trend: 'unknown', observations: 20, status: 'critical' },
          ensemble_consensus: { ic_rolling: 0.0391, ic_trend: 'unknown', observations: 20, status: 'critical' },
        },
      },
    } as SignalsData;
    const healthWithQuality = {
      ...health('healthy'),
      ic_decay_summary: {
        status: 'critical',
        critical_signals: ['ensemble_consensus', 'ensemble_duration'],
        warning_signals: [],
        resolved_signal_count: 2,
        min_observations: 20,
        staged_pending_predictions: 7,
        staged_date: '2026-08-01',
        staged_pending_scope: 'ic_staged_date_window',
        historical_unlabeled_rows: 1663,
        historical_unlabeled_dates: 2,
        historical_unlabeled_scope: 'historical_db_unlabeled_rows',
        evidence_generated_at: '2026-08-01T09:40:23Z',
        evidence_freshness: 'captured_runtime_snapshot',
        routing_authority: 'advisory_only',
        control_effect: 'paper_warning',
      },
    } as HealthData;
    const persisted = incidentSummary({
      incidents: [{
        incident_id: 'ic-decay-1',
        channel: 'ic_decay',
        severity: 'p0',
        state: 'firing',
        message: 'IC quality event retained for review.',
        details: {},
        created_at: '2026-08-01T09:00:00Z',
        updated_at: '2026-08-01T09:16:15Z',
        resolved_at: null,
        resolution_notes: null,
        mttr_seconds: null,
      }],
    });

    const incidents = buildDashboardIncidents({
      alerts: [],
      signals,
      health: healthWithQuality,
      incidentSummary: persisted,
    });

    expect(incidents.filter((incident) => incident.source === 'Incident lifecycle')).toHaveLength(1);
    expect(incidents.filter((incident) => incident.source === 'IC decay monitor')).toHaveLength(0);
    expect(incidents[0]?.message).toContain('ensemble_duration');
    expect(incidents[0]?.currentValue).toContain('0.0045');
  });

  it('derives one reviewable IC condition when no persisted incident exists', () => {
    const qualityHealth = {
      ...health('healthy'),
      ic_decay_summary: {
        status: 'critical',
        critical_signals: ['ensemble_consensus', 'ensemble_duration'],
        warning_signals: [],
        resolved_signal_count: 2,
        min_observations: 20,
        staged_pending_predictions: 7,
        staged_date: '2026-08-01',
        staged_pending_scope: 'ic_staged_date_window',
        historical_unlabeled_rows: 1663,
        historical_unlabeled_dates: 2,
        historical_unlabeled_scope: 'historical_db_unlabeled_rows',
        evidence_generated_at: '2026-08-01T09:40:23Z',
        evidence_freshness: 'captured_runtime_snapshot',
        routing_authority: 'advisory_only',
        control_effect: 'paper_warning',
      },
    } as HealthData;

    const incidents = buildDashboardIncidents({
      alerts: [],
      signals: null,
      health: qualityHealth,
    });

    expect(incidents).toHaveLength(1);
    expect(incidents[0]).toMatchObject({
      id: 'quality:ic-decay',
      tab: 'health',
      severity: 'critical',
      title: 'Signal quality: IC decay',
    });
  });

  it('merges a matching critical IC kill alert into one persisted critical incident', () => {
    const qualityHealth = {
      ...health('healthy'),
      ic_decay_summary: {
        status: 'warning',
        critical_signals: [],
        warning_signals: ['ensemble_equity'],
        resolved_signal_count: 1,
        min_observations: 20,
        staged_pending_predictions: 7,
        staged_date: '2026-08-01',
        staged_pending_scope: 'ic_staged_date_window',
        historical_unlabeled_rows: 1663,
        historical_unlabeled_dates: 2,
        historical_unlabeled_scope: 'historical_db_unlabeled_rows',
        evidence_generated_at: '2026-08-01T09:40:23Z',
        evidence_freshness: 'captured_runtime_snapshot',
        routing_authority: 'advisory_only',
        routing_control: 'routing_blocked',
        control_effect: 'paper_warning',
      },
    } as HealthData;
    const persisted = incidentSummary({
      incidents: [{
        incident_id: 'ic-decay-1',
        channel: 'ic_decay',
        severity: 'p1',
        state: 'firing',
        message: 'IC quality event retained for review.',
        details: {},
        created_at: '2026-08-01T09:00:00Z',
        updated_at: '2026-08-01T09:16:15Z',
        resolved_at: null,
        resolution_notes: null,
        mttr_seconds: null,
      }],
    });

    const incidents = buildDashboardIncidents({
      alerts: [alert({
        level: 'critical',
        type: 'kill_switch',
        incident_id: 'ic-decay-1',
        reason: 'unresolved_incident:ic_decay',
        title: 'PAPER Kill Switch Triggered',
        message: 'IC decay remains unresolved.',
      })],
      signals: null,
      health: qualityHealth,
      incidentSummary: persisted,
    });

    expect(incidents).toHaveLength(1);
    expect(incidents[0]).toMatchObject({
      id: 'persisted:ic_decay:ic-decay-1',
      severity: 'critical',
      source: 'Incident lifecycle',
    });
    expect(incidents.some((incident) => incident.id.startsWith('alert:'))).toBe(false);
  });

  it('names the collection behind Action Center counts', () => {
    const incidents = [
      alert({ type: 'one', level: 'warning', title: 'One' }),
      alert({ type: 'two', level: 'warning', title: 'Two' }),
      alert({ type: 'three', level: 'warning', title: 'Three' }),
      alert({ type: 'four', level: 'warning', title: 'Four' }),
      alert({ type: 'advisory', level: 'warning', title: 'Advisory', requires_action: false }),
    ].map((row) => buildDashboardIncidents({ alerts: [row], signals: null, health: null })[0]!);
    const html = renderToStaticMarkup(React.createElement(ActionCenter, { incidents, limit: 3 }));

    expect(html).toContain('4 actions required');
    expect(html).toContain('Showing 3 of 5 conditions');
    expect(html).not.toContain('Showing 3 of 5 actions');
  });

  it('derives Decisions-tab incidents from staleness and smart rebalance hold', () => {
    const signals = {
      timestamp: '2026-07-01T12:00:00Z',
      staleness: { stale_signals: ['fred_macro', 'ensemble_voting'] },
      smart_rebalance: {
        decision: 'defer',
        reason: 'vpin_high',
        should_execute: false,
      },
    } as SignalsData;

    const incidents = buildDecisionIncidents(signals);
    expect(incidents).toHaveLength(2);
    expect(incidents[0]?.tab).toBe('decisions');
    expect(getIncidentsForTab(incidents, 'decisions')).toHaveLength(2);
    expect(getTabIncidentBadge(incidents, 'decisions')).toEqual({ count: 2, severity: 'warning' });
  });
});
