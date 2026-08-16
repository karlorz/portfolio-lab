import React, { useState } from 'react';
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
  // Interactive UAT Playground State
  const [routed, setRouted] = useState<boolean | undefined>(true);
  const [killEnabled, setKillEnabled] = useState(false);
  const [killLevel, setKillLevel] = useState<'clear' | 'warning' | 'halt'>('clear');
  const [regime, setRegime] = useState<string>('neutral');
  const [spyAlloc, setSpyAlloc] = useState(46);
  const [gldAlloc, setGldAlloc] = useState(38);
  const [tltAlloc, setTltAlloc] = useState(16);
  const [badgeTone, setBadgeTone] = useState<'success' | 'warning' | 'critical' | 'info' | 'stale' | 'neutral'>('success');
  const [paletteOpen, setPaletteOpen] = useState(false);

  const totalAlloc = spyAlloc + gldAlloc + tltAlloc;
  const normalizedAlloc = totalAlloc > 0 ? {
    SPY: spyAlloc / totalAlloc,
    GLD: gldAlloc / totalAlloc,
    TLT: tltAlloc / totalAlloc,
  } : null;

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

      <Section title="Interactive Operator UAT Playground">
        <div style={{
          padding: '16px',
          background: 'var(--surface-panel)',
          border: '1px solid var(--border-strong)',
          borderRadius: 'var(--radius-lg)',
          display: 'flex',
          flexDirection: 'column',
          gap: '16px',
        }}>
          <p className="control-eyebrow">Interactive Controls &amp; Real-time Component Reactions</p>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '16px' }}>
            <div>
              <label style={{ display: 'block', fontSize: '13px', fontWeight: 600, marginBottom: '6px' }}>
                Routing Authority
              </label>
              <select
                value={routed === undefined ? 'undefined' : String(routed)}
                onChange={(e) => {
                  const val = e.target.value;
                  setRouted(val === 'undefined' ? undefined : val === 'true');
                }}
                style={{
                  width: '100%',
                  padding: '6px 8px',
                  borderRadius: 'var(--radius-sm)',
                  border: '1px solid var(--border-subtle)',
                  background: 'var(--surface-canvas)',
                  color: 'inherit',
                }}
              >
                <option value="true">Live Routed (Active)</option>
                <option value="false">Unrouted (Advisory)</option>
                <option value="undefined">Unavailable / Undefined</option>
              </select>
            </div>

            <div>
              <label style={{ display: 'block', fontSize: '13px', fontWeight: 600, marginBottom: '6px' }}>
                Kill Switch Policy
              </label>
              <select
                value={killEnabled ? killLevel : 'clear'}
                onChange={(e) => {
                  const val = e.target.value;
                  if (val === 'clear') {
                    setKillEnabled(false);
                    setKillLevel('clear');
                  } else {
                    setKillEnabled(true);
                    setKillLevel(val as 'warning' | 'halt');
                  }
                }}
                style={{
                  width: '100%',
                  padding: '6px 8px',
                  borderRadius: 'var(--radius-sm)',
                  border: '1px solid var(--border-subtle)',
                  background: 'var(--surface-canvas)',
                  color: 'inherit',
                }}
              >
                <option value="clear">Disarmed / Clear</option>
                <option value="warning">Armed (Warning level)</option>
                <option value="halt">Armed (Halt level - Hard stop)</option>
              </select>
            </div>

            <div>
              <label style={{ display: 'block', fontSize: '13px', fontWeight: 600, marginBottom: '6px' }}>
                Market Regime
              </label>
              <select
                value={regime}
                onChange={(e) => setRegime(e.target.value)}
                style={{
                  width: '100%',
                  padding: '6px 8px',
                  borderRadius: 'var(--radius-sm)',
                  border: '1px solid var(--border-subtle)',
                  background: 'var(--surface-canvas)',
                  color: 'inherit',
                }}
              >
                <option value="neutral">Neutral / Normal</option>
                <option value="risk_on">Risk On</option>
                <option value="risk_off">Risk Off</option>
                <option value="high_vol">High Volatility</option>
                <option value="stagflation">Stagflation</option>
              </select>
            </div>

            <div>
              <label style={{ display: 'block', fontSize: '13px', fontWeight: 600, marginBottom: '6px' }}>
                Status Badge Tone
              </label>
              <select
                value={badgeTone}
                onChange={(e) => setBadgeTone(e.target.value as any)}
                style={{
                  width: '100%',
                  padding: '6px 8px',
                  borderRadius: 'var(--radius-sm)',
                  border: '1px solid var(--border-subtle)',
                  background: 'var(--surface-canvas)',
                  color: 'inherit',
                }}
              >
                <option value="success">Success / Healthy</option>
                <option value="warning">Warning / Advisory</option>
                <option value="critical">Critical / Halting</option>
                <option value="info">Info</option>
                <option value="stale">Stale</option>
                <option value="neutral">Neutral</option>
              </select>
            </div>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: '12px' }}>
            <div>
              <label style={{ display: 'block', fontSize: '12px', marginBottom: '4px' }}>SPY Weight: {spyAlloc}%</label>
              <input
                type="range"
                min="0"
                max="100"
                value={spyAlloc}
                onChange={(e) => setSpyAlloc(Number(e.target.value))}
                style={{ width: '100%' }}
              />
            </div>
            <div>
              <label style={{ display: 'block', fontSize: '12px', marginBottom: '4px' }}>GLD Weight: {gldAlloc}%</label>
              <input
                type="range"
                min="0"
                max="100"
                value={gldAlloc}
                onChange={(e) => setGldAlloc(Number(e.target.value))}
                style={{ width: '100%' }}
              />
            </div>
            <div>
              <label style={{ display: 'block', fontSize: '12px', marginBottom: '4px' }}>TLT Weight: {tltAlloc}%</label>
              <input
                type="range"
                min="0"
                max="100"
                value={tltAlloc}
                onChange={(e) => setTltAlloc(Number(e.target.value))}
                style={{ width: '100%' }}
              />
            </div>
          </div>

          <div style={{
            marginTop: '8px',
            paddingTop: '16px',
            borderTop: '1px solid var(--border-subtle)',
            display: 'flex',
            flexDirection: 'column',
            gap: '12px',
          }}>
            <p className="control-eyebrow">Playground Live Preview</p>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '12px', alignItems: 'center' }}>
              <StatusBadge label={`Status: ${badgeTone.toUpperCase()}`} tone={badgeTone} detail="Interactive tone probe" />
              <AuthorityBadge
                routed={routed}
                blocked={killEnabled && killLevel === 'halt'}
                source="signals.json.target_allocations"
              />
              <button
                type="button"
                onClick={() => setPaletteOpen(true)}
                style={{
                  padding: '6px 12px',
                  borderRadius: 'var(--radius-sm)',
                  border: '1px solid var(--border-strong)',
                  background: 'var(--surface-panel)',
                  cursor: 'pointer',
                  fontSize: '13px',
                }}
              >
                Open Command Palette (Interactive)
              </button>
            </div>

            <AllocationSpine
              allocations={normalizedAlloc}
              regime={regime}
              updatedAt="Just now (UAT)"
              routed={routed}
              killEnabled={killEnabled}
              killLevel={killLevel}
            />
          </div>
        </div>
        <CommandPalette open={paletteOpen} onClose={() => setPaletteOpen(false)} onNavigate={() => setPaletteOpen(false)} />
      </Section>

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
          <StatusBadge label="Healthy" tone="success" detail="All subsystems operational" />
          <StatusBadge label="Warning" tone="warning" detail="Advisory degraded state" />
          <StatusBadge label="Critical" tone="critical" detail="Requires immediate action" />
          <StatusBadge label="Info" tone="info" detail="Informational status" />
          <StatusBadge label="Stale" tone="stale" detail="Data freshness overdue" />
          <StatusBadge label="Neutral" tone="neutral" detail="Default neutral tone" />
        </div>
        <div className="design-guide-row" style={{ marginTop: '12px' }}>
          <AuthorityBadge routed source="signals.json.target_allocations" />
          <AuthorityBadge routed={false} source="signals.json.target_allocations" />
          <AuthorityBadge blocked source="kill_switch: active halt" />
          <AuthorityBadge routed={undefined} source="signals.json.target_allocations" />
        </div>
      </Section>

      <Section title="Allocation Spine">
        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          <div>
            <p className="control-eyebrow" style={{ marginBottom: '8px' }}>Active routed allocation (Champion 46/38/16)</p>
            <AllocationSpine
              allocations={{ SPY: 0.46, GLD: 0.38, TLT: 0.16 }}
              regime="neutral"
              updatedAt="11:40 CST"
              routed
            />
          </div>
          <div>
            <p className="control-eyebrow" style={{ marginBottom: '8px' }}>Kill switch active (level = halt)</p>
            <AllocationSpine
              allocations={{ SPY: 0.46, GLD: 0.38, TLT: 0.16 }}
              regime="high_vol"
              updatedAt="11:40 CST"
              killEnabled
              killLevel="halt"
              routed
            />
          </div>
          <div>
            <p className="control-eyebrow" style={{ marginBottom: '8px' }}>Advisory unrouted allocation</p>
            <AllocationSpine allocations={{ SPY: 0.50, GLD: 0.30, TLT: 0.20 }} routed={false} />
          </div>
          <div>
            <p className="control-eyebrow" style={{ marginBottom: '8px' }}>Unavailable / Loading state</p>
            <AllocationSpine allocations={null} routed={false} />
          </div>
        </div>
      </Section>

      <Section title="Metrics and actions">
        <div className="design-guide-metrics">
          <MetricCard label="Portfolio value" value="$100,482" detail="+0.48% since inception" />
          <MetricCard label="Open incidents" value="1" detail="Highest severity P2" tone="attention" />
          <MetricCard label="Fresh signals" value="22 / 23" detail="1 advisory feed stale" />
          <MetricCard label="System status" value="Warning" detail="1 degraded source" tone="attention" />
        </div>
        <ActionCenter incidents={exampleActions} />
        <ActionCenter incidents={[]} />
      </Section>

      <Section title="Quality evidence brief">
        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          <div>
            <p className="control-eyebrow" style={{ marginBottom: '8px' }}>Critical signal quality (IC decay active)</p>
            <section className="operator-brief operator-brief-critical" aria-labelledby="guide-operator-brief-title-crit">
              <div className="operator-brief-header">
                <div>
                  <p className="control-eyebrow">Current control state</p>
                  <h2 id="guide-operator-brief-title-crit">Operator brief</h2>
                </div>
                <span className="operator-brief-status operator-brief-status-critical">Signal quality: Critical</span>
              </div>
              <div className="operator-brief-grid">
                <div className="operator-brief-item">
                  <span>Market regime</span>
                  <strong>Normal</strong>
                </div>
                <div className="operator-brief-item">
                  <span>Signal quality</span>
                  <strong>2 critical signals</strong>
                  <small>ensemble_duration · ensemble_gold · n=60/20</small>
                </div>
                <div className="operator-brief-item">
                  <span>Execution control</span>
                  <strong>Routing available</strong>
                  <small>Routing authority: advisory_only</small>
                </div>
              </div>
              <div className="operator-brief-evidence">
                <p><strong>Affected signals</strong> ensemble_duration IC -0.0563 (60/20) · ensemble_gold IC -0.1197 (60/20)</p>
                <p>2 staged pending labels · 0 historical unlabeled rows</p>
                <p>Captured runtime snapshot · paper_warning control effect</p>
                <button type="button" className="operator-brief-action">Review IC evidence</button>
              </div>
            </section>
          </div>

          <div>
            <p className="control-eyebrow" style={{ marginBottom: '8px' }}>Healthy signal quality</p>
            <section className="operator-brief operator-brief-healthy" aria-labelledby="guide-operator-brief-title-ok">
              <div className="operator-brief-header">
                <div>
                  <p className="control-eyebrow">Current control state</p>
                  <h2 id="guide-operator-brief-title-ok">Operator brief</h2>
                </div>
                <span className="operator-brief-status operator-brief-status-healthy">Signal quality: Healthy</span>
              </div>
              <div className="operator-brief-grid">
                <div className="operator-brief-item">
                  <span>Market regime</span>
                  <strong>Normal</strong>
                </div>
                <div className="operator-brief-item">
                  <span>Signal quality</span>
                  <strong>All signals qualified</strong>
                  <small>0 affected signals · 6 qualified</small>
                </div>
                <div className="operator-brief-item">
                  <span>Execution control</span>
                  <strong>Routing available</strong>
                  <small>Routing authority: signals.json.target_allocations</small>
                </div>
              </div>
            </section>
          </div>
        </div>
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
