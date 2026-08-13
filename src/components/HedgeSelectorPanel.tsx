import type { HedgeSelectorData } from '../types/live';

interface HedgeSelectorPanelProps {
  data?: HedgeSelectorData | null;
}

export interface HedgeSelectorMetric {
  label: string;
  value: string;
  detail?: string;
}

export type HedgeSelectorDisplay =
  | {
      available: false;
      title: string;
      emptyMessage: string;
      emptyDetail: string;
    }
  | {
      available: true;
      title: string;
      gateLabel: string;
      gateOpen: boolean;
      regimeLabel: string;
      primaryHedge: HedgeSelectorMetric;
      secondaryHedge: HedgeSelectorMetric;
      regimeConfidence: HedgeSelectorMetric;
      expectedBenefit: HedgeSelectorMetric;
      expectedCost: HedgeSelectorMetric;
      netBenefit: HedgeSelectorMetric;
      kellyFraction: HedgeSelectorMetric;
      confidenceScaledSize: HedgeSelectorMetric;
      minimumHold: HedgeSelectorMetric;
      lastUpdated: string;
      netBenefitBps: number;
    };

export function formatHedgeBps(value: number): string {
  return `${value.toFixed(1)} bps`;
}

export function formatHedgeSizePct(value: number): string {
  return `${value.toFixed(2)}%`;
}

export function formatHedgeRatioPct(value: number): string {
  return `${(value * 100).toFixed(2)}%`;
}

function formatMinimumHold(days: number): string {
  if (days <= 0) return 'Not reported';
  return `${days} trading ${days === 1 ? 'day' : 'days'}`;
}

function formatGeneratedAt(value: string): string {
  if (!value) return 'Unknown';
  const timestamp = new Date(value);
  if (Number.isNaN(timestamp.getTime())) return 'Unknown';
  return timestamp.toLocaleString();
}

export function buildHedgeSelectorDisplay(data?: HedgeSelectorData | null): HedgeSelectorDisplay {
  if (!data || !data.available) {
    return {
      available: false,
      title: 'Hedge Selector',
      emptyMessage: 'Hedge selector data not available',
      emptyDetail: 'Run hedge selector exporter to populate',
    };
  }

  return {
    available: true,
    title: 'Hedge Selector',
    gateLabel: data.cost_benefit_gate ? 'GATE OPEN' : 'GATE CLOSED',
    gateOpen: data.cost_benefit_gate,
    regimeLabel: data.regime,
    primaryHedge: {
      label: 'Primary Hedge',
      value: data.primary_hedge,
      detail: formatHedgeSizePct(data.primary_size_pct),
    },
    secondaryHedge: {
      label: 'Secondary Hedge',
      value: data.secondary_hedge || 'None',
      detail: data.secondary_hedge ? formatHedgeSizePct(data.secondary_size_pct) : '-',
    },
    regimeConfidence: {
      label: 'Regime Confidence',
      value: formatHedgeRatioPct(data.regime_confidence),
    },
    expectedBenefit: {
      label: 'Expected Benefit',
      value: formatHedgeBps(data.expected_benefit_bps),
    },
    expectedCost: {
      label: 'Expected Cost',
      value: formatHedgeBps(data.expected_cost_bps),
    },
    netBenefit: {
      label: 'Net Benefit',
      value: formatHedgeBps(data.net_benefit_bps),
    },
    kellyFraction: {
      label: 'Kelly Fraction',
      value: formatHedgeRatioPct(data.kelly_fraction),
    },
    confidenceScaledSize: {
      label: 'Confidence Scaled Size',
      value: formatHedgeSizePct(data.confidence_scaled_size),
    },
    minimumHold: {
      label: 'Minimum Hold',
      value: formatMinimumHold(data.min_hold_days),
    },
    lastUpdated: formatGeneratedAt(data.generated_at),
    netBenefitBps: data.net_benefit_bps,
  };
}

function MetricCard({
  metric,
  valueClassName = '',
}: {
  metric: HedgeSelectorMetric;
  valueClassName?: string;
}) {
  return (
    <div className="alc-card">
      <p className="alc-label">{metric.label}</p>
      <p className={`alc-value-lg ${valueClassName}`}>{metric.value}</p>
      {metric.detail !== undefined && (
        <p className="alc-small">{metric.detail}</p>
      )}
    </div>
  );
}

export function HedgeSelectorPanel({ data }: HedgeSelectorPanelProps) {
  const display = buildHedgeSelectorDisplay(data);

  if (!display.available) {
    return (
      <div className="alc-panel alc-panel-muted">
        <div className="alc-header">
          <h3 className="alc-title">{display.title}</h3>
        </div>
        <p className="alc-muted">{display.emptyMessage}</p>
        <p className="alc-small">{display.emptyDetail}</p>
      </div>
    );
  }

  const netBenefitColor = display.netBenefitBps > 0 ? 'alc-text-success' : 'alc-text-danger';
  const gaugeWidth = Math.min(Math.max((display.netBenefitBps + 50) / 100, 0), 1) * 100;
  const gaugeColor =
    display.netBenefitBps > 10 ? '#10b981' :
    display.netBenefitBps > 0 ? '#3b82f6' :
    display.netBenefitBps > -10 ? '#f59e0b' :
    '#ef4444';

  return (
    <div className="alc-panel alc-panel-muted">
      <div className="alc-header">
        <h3 className="alc-title">{display.title}</h3>
        <div className="alc-header-actions">
          <span className={`alc-chip ${
            display.gateOpen ? 'alc-chip-success' : 'alc-chip-danger'
          }`}>
            {display.gateLabel}
          </span>
          <span className="alc-chip alc-chip-info">
            {display.regimeLabel}
          </span>
        </div>
      </div>

      <div className="alc-section">
        <h4 className="alc-section-title">Current Hedge Allocation</h4>
        <div className="alc-grid alc-grid-three">
          <MetricCard metric={display.primaryHedge} />
          <MetricCard metric={display.secondaryHedge} />
          <MetricCard metric={display.regimeConfidence} />
        </div>
      </div>

      <div className="alc-section">
        <h4 className="alc-section-title">Cost-Benefit Analysis</h4>
        <div className="alc-grid alc-grid-three">
          <MetricCard metric={display.expectedBenefit} valueClassName="alc-text-success" />
          <MetricCard metric={display.expectedCost} valueClassName="alc-text-warning" />
          <MetricCard metric={display.netBenefit} valueClassName={netBenefitColor} />
        </div>

        <div className="alc-card">
          <div className="alc-row">
            <span>Net Benefit (bps)</span>
            <span>{display.netBenefit.value}</span>
          </div>
          <div className="alc-progress">
            <div
              className="alc-progress-fill"
              style={{ width: `${gaugeWidth}%`, backgroundColor: gaugeColor }}
            />
          </div>
          <div className="alc-scale">
            <span>-50 bps</span>
            <span>0</span>
            <span>+50 bps</span>
          </div>
        </div>
      </div>

      <div className="alc-section">
        <h4 className="alc-section-title">Sizing Parameters</h4>
        <div className="alc-grid alc-grid-three">
          <MetricCard metric={display.kellyFraction} />
          <MetricCard metric={display.confidenceScaledSize} />
          <MetricCard metric={display.minimumHold} />
        </div>
      </div>

      <div className="alc-small">
        <p>Last updated: {display.lastUpdated}</p>
      </div>
    </div>
  );
}
