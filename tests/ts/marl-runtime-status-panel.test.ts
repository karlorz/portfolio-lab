import React from 'react';
import { describe, expect, it } from 'bun:test';
import { renderToStaticMarkup } from 'react-dom/server';
import {
  MarlRuntimeStatusPanel,
  type MarlRuntimeStatusData,
} from '../../src/components/MarlRuntimeStatusPanel';

describe('MarlRuntimeStatusPanel', () => {
  it('renders MARL controller status separately from heuristic ML signals', () => {
    const data: MarlRuntimeStatusData = {
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
    };

    const html = renderToStaticMarkup(React.createElement(MarlRuntimeStatusPanel, { data }));

    expect(html).toContain('MARL Runtime Status');
    expect(html).toContain('Research Shadow');
    expect(html).toContain('Not order-routed');
    expect(html).toContain('2.51.0');
    expect(html).toContain('analyst');
    expect(html).toContain('target_allocations');
  });
});
