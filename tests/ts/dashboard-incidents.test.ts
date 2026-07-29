import { describe, expect, it } from 'bun:test';
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
