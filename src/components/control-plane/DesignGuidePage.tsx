import React from 'react';
import { ActionCenter } from './ActionCenter';
import { AllocationSpine } from './AllocationSpine';
import { AppErrorBoundary } from './AppErrorBoundary';
import { AppShell } from './AppShell';
import { AuthorityBadge } from './AuthorityBadge';
import { CommandPalette } from './CommandPalette';
import { ContextRail } from './ContextRail';
import { MetricCard } from './MetricCard';
import { NavigationRail } from './NavigationRail';
import { OverflowRegion } from './OverflowRegion';
import { StatusBadge } from './StatusBadge';
import { WorkspaceHeader } from './WorkspaceHeader';

const exampleActions = [
  {
    id: 'guide:warning',
    tab: 'health' as const,
    severity: 'warning' as const,
    attention: 'advisory' as const,
    title: 'Regime transition data is stale',
    source: 'Signal freshness',
    message: 'The advisory regime-transition feed missed its expected update.',
    nextAction: 'Open Health and inspect the producer before relying on this signal.',
  },
  {
    id: 'guide:critical',
    tab: 'risk' as const,
    severity: 'critical' as const,
    attention: 'action' as const,
    title: 'Order routing is halted',
    source: 'Kill authority',
    message: 'The current kill policy blocks new paper orders.',
    nextAction: 'Resolve the owning incident before placing new orders.',
  },
];

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="design-guide-section">
      <h2>{title}</h2>
      {children}
    </section>
  );
}

export function DesignGuidePage() {
  return (
    <main className="design-guide-page">
      <header className="design-guide-hero">
        <a href="/">← Portfolio Lab</a>
        <p className="control-eyebrow">Living interface reference</p>
        <h1>Control Plane Design Guide</h1>
        <p>
          Authority is prominent, exceptions are actionable, and healthy state stays quiet.
          These production components define the cockpit vocabulary.
        </p>
      </header>

      <Section title="Semantic palette">
        <div className="design-guide-swatches">
          {[
            ['Canvas', '--surface-canvas'],
            ['Panel', '--surface-panel'],
            ['Authority', '--authority-accent'],
            ['Warning', '--status-warning'],
            ['Critical', '--status-critical'],
            ['Success', '--status-success'],
          ].map(([label, token]) => (
            <div key={token}>
              <span style={{ background: `var(${token})` }} />
              <strong>{label}</strong>
              <code>{token}</code>
            </div>
          ))}
        </div>
      </Section>

      <Section title="Typography and status">
        <div className="design-guide-type">
          <h1>Display · Archivo</h1>
          <p>Interface copy · IBM Plex Sans with system fallbacks</p>
          <code>Data · SPY 46.0% · 2026-07-28T03:40Z</code>
        </div>
        <div className="design-guide-row">
          <StatusBadge label="Healthy" tone="success" />
          <StatusBadge label="Warning" tone="warning" />
          <StatusBadge label="Critical" tone="critical" />
          <StatusBadge label="Stale" tone="stale" />
          <AuthorityBadge routed />
          <AuthorityBadge routed={false} />
        </div>
      </Section>

      <Section title="Allocation Spine">
        <AllocationSpine
          allocations={{ SPY: 0.46, GLD: 0.38, TLT: 0.16 }}
          regime="neutral"
          updatedAt="11:40 CST"
          killEnabled
          killLevel="warning"
        />
        <AllocationSpine allocations={null} routed={false} />
      </Section>

      <Section title="Metrics and actions">
        <div className="design-guide-metrics">
          <MetricCard label="Portfolio value" value="$100,482" detail="+0.48% since inception" />
          <MetricCard label="Open incidents" value="1" detail="Highest severity P2" tone="attention" />
          <MetricCard label="Fresh signals" value="22 / 23" detail="1 advisory feed stale" />
        </div>
        <ActionCenter incidents={exampleActions} />
        <ActionCenter incidents={[]} />
      </Section>

      <Section title="Overflow and advisory regions">
        <OverflowRegion label="Design guide sample table">
          <table className="positions-table">
            <thead>
              <tr>
                <th>Symbol</th>
                <th>Shares</th>
                <th>Value</th>
                <th>Weight</th>
                <th>Unrealized P&amp;L</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>SPY</td>
                <td>46.00</td>
                <td>$46,000</td>
                <td>46.0%</td>
                <td>$0</td>
              </tr>
            </tbody>
          </table>
        </OverflowRegion>
      </Section>

      <Section title="Navigation and workspace header">
        <div className="design-guide-shell-sample">
          <NavigationRail active="overview" onNavigate={() => undefined} />
          <div>
            <WorkspaceHeader workspace="overview" refreshedAt="11:40:23" />
            <p className="control-empty">Workspace content begins here.</p>
          </div>
        </div>
      </Section>

      <Section title="Shell composition and containment">
        <AppErrorBoundary scope="workspace" name="Design guide shell">
          <div className="design-guide-shell-frame">
            <AppShell
              workspace="overview"
              onNavigate={() => undefined}
              spine={(
                <AllocationSpine
                  allocations={{ SPY: 0.46, GLD: 0.38, TLT: 0.16 }}
                  regime="neutral"
                  updatedAt="11:40 CST"
                />
              )}
              context={(
                <ContextRail
                  incidents={exampleActions}
                  freshness="Core data refreshed at 11:40 CST"
                  openIncidentCount={1}
                  runtimeProvenance={{
                    staticRelease: 'sha=7f3c1d2 · built=2026-07-31T03:40Z',
                    runtimeArtifact: 'signals.json · plane=public · generated=11:40 CST',
                    runtimeStatus: 'generator=full_generate · system=healthy',
                    orderAuthority: 'signals.json.target_allocations → src.broker.order_router',
                  }}
                />
              )}
            >
              <p className="control-empty">The active workspace is isolated by its own error boundary.</p>
            </AppShell>
          </div>
        </AppErrorBoundary>
        <CommandPalette open={false} onClose={() => undefined} onNavigate={() => undefined} />
      </Section>

      <Section title="Loading, empty, and error language">
        <div className="design-guide-states">
          <p role="status" aria-live="polite">Loading current authority…</p>
          <p className="control-empty">No open operator actions.</p>
          <div className="control-error">
            <strong>Workspace unavailable</strong>
            <p>Reload the dashboard. If the failure persists, inspect the latest tasker logs.</p>
          </div>
        </div>
      </Section>
    </main>
  );
}
