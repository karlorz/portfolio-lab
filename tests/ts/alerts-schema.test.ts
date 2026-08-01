import { describe, expect, it, spyOn } from 'bun:test';
import { validateAlertsData } from '../../src/schemas/signals';

const GENERATED_AT = '2026-07-28T03:30:00Z';

function healthJobKillAlert(overrides: Record<string, unknown> = {}) {
  return {
    type: 'kill_switch',
    level: 'warning',
    reason: 'Manual halt remains active',
    incident_id: 'kill-incident-42',
    enabled: true,
    message: 'New orders are blocked pending operator review.',
    ...overrides,
  };
}

describe('validateAlertsData', () => {
  it('normalizes the live health-job kill alert without optional display fields', () => {
    const result = validateAlertsData({
      alerts: [healthJobKillAlert()],
      count: 1,
      generated_at: GENERATED_AT,
    });

    expect(result).not.toBeNull();
    expect(result?.alerts).toEqual([
      expect.objectContaining({
        type: 'kill_switch',
        level: 'warning',
        title: 'Kill Switch',
        message: 'New orders are blocked pending operator review.',
        timestamp: GENERATED_AT,
        requires_action: true,
        stable_id: 'kill-incident-42',
        incident_id: 'kill-incident-42',
      }),
    ]);
  });

  it('uses a deterministic type/message identity when no incident id exists', () => {
    const first = validateAlertsData({
      alerts: [healthJobKillAlert({ incident_id: undefined })],
      count: 1,
      generated_at: GENERATED_AT,
    });
    const second = validateAlertsData({
      alerts: [healthJobKillAlert({ incident_id: '   ' })],
      count: 1,
      generated_at: GENERATED_AT,
    });

    expect(first?.alerts[0]?.stable_id).toBe(
      'kill_switch:new-orders-are-blocked-pending-operator-review',
    );
    expect(second?.alerts[0]?.stable_id).toBe(first?.alerts[0]?.stable_id);
  });

  it('normalizes whitespace and preserves complete alert display fields', () => {
    const result = validateAlertsData({
      alerts: [{
        type: ' portfolio_drift ',
        level: 'warning',
        title: ' Drift Alert ',
        message: ' SPY drift exceeds threshold ',
        timestamp: ' 2026-07-28T03:25:00Z ',
        requires_action: false,
      }],
      count: 1,
      generated_at: GENERATED_AT,
    });

    expect(result?.alerts[0]).toMatchObject({
      type: 'portfolio_drift',
      title: 'Drift Alert',
      message: 'SPY drift exceeds threshold',
      timestamp: '2026-07-28T03:25:00Z',
      requires_action: false,
    });
  });

  it('omits malformed rows and emits one bounded diagnostic', () => {
    const warn = spyOn(console, 'warn').mockImplementation(() => undefined);
    try {
      const result = validateAlertsData({
        alerts: [
          healthJobKillAlert(),
          { type: 'kill_switch', level: 'warning', message: '   ' },
          { type: 'kill_switch', level: 'fatal', message: 'Bad level' },
        ],
        count: 3,
        generated_at: GENERATED_AT,
      });

      expect(result?.alerts).toHaveLength(1);
      expect(result?.count).toBe(1);
      expect(warn).toHaveBeenCalledTimes(1);
      expect(warn.mock.calls[0]?.[0]).toContain('[alerts] Omitted 2 malformed alert rows');
    } finally {
      warn.mockRestore();
    }
  });

  it('returns null for an unusable envelope and accepts an honest empty state', () => {
    expect(validateAlertsData({ generated_at: GENERATED_AT })).toBeNull();
    expect(validateAlertsData({
      alerts: [],
      count: 0,
      generated_at: GENERATED_AT,
    })).toEqual({
      alerts: [],
      count: 0,
      generated_at: GENERATED_AT,
    });
  });
});
