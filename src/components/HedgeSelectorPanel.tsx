import React from 'react';
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
  valueClassName = 'text-gray-100',
}: {
  metric: HedgeSelectorMetric;
  valueClassName?: string;
}) {
  return (
    <div className="bg-gray-800/60 rounded-lg p-3 border border-gray-700/40">
      <p className="text-xs text-gray-500 mb-1">{metric.label}</p>
      <p className={`text-lg font-mono font-bold ${valueClassName}`}>{metric.value}</p>
      {metric.detail !== undefined && (
        <p className="text-xs text-gray-500 mt-1">{metric.detail}</p>
      )}
    </div>
  );
}

export function HedgeSelectorPanel({ data }: HedgeSelectorPanelProps) {
  const display = buildHedgeSelectorDisplay(data);

  if (!display.available) {
    return (
      <div className="bg-gray-900/50 border border-gray-700/50 rounded-lg p-4">
        <div className="flex items-center justify-between mb-2">
          <h3 className="text-base font-semibold text-gray-100">{display.title}</h3>
        </div>
        <p className="text-sm text-gray-500">{display.emptyMessage}</p>
        <p className="text-xs text-gray-600 mt-1">{display.emptyDetail}</p>
      </div>
    );
  }

  const netBenefitColor = display.netBenefitBps > 0 ? 'text-emerald-400' : 'text-red-400';
  const gaugeWidth = Math.min(Math.max((display.netBenefitBps + 50) / 100, 0), 1) * 100;
  const gaugeColor =
    display.netBenefitBps > 10 ? '#10b981' :
    display.netBenefitBps > 0 ? '#3b82f6' :
    display.netBenefitBps > -10 ? '#f59e0b' :
    '#ef4444';

  return (
    <div className="bg-gray-900/50 border border-gray-700/50 rounded-lg p-4 space-y-5 text-gray-100">
      <div className="flex items-center justify-between gap-3">
        <h3 className="text-base font-semibold">{display.title}</h3>
        <div className="flex items-center gap-2">
          <span className={`px-2.5 py-0.5 rounded text-xs font-semibold ${
            display.gateOpen ? 'bg-emerald-600/20 text-emerald-400' : 'bg-red-600/20 text-red-400'
          }`}>
            {display.gateLabel}
          </span>
          <span className="px-2.5 py-0.5 rounded text-xs font-semibold bg-blue-600/20 text-blue-400">
            {display.regimeLabel}
          </span>
        </div>
      </div>

      <div>
        <h4 className="text-sm font-medium text-gray-400 mb-3">Current Hedge Allocation</h4>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          <MetricCard metric={display.primaryHedge} />
          <MetricCard metric={display.secondaryHedge} />
          <MetricCard metric={display.regimeConfidence} />
        </div>
      </div>

      <div>
        <h4 className="text-sm font-medium text-gray-400 mb-3">Cost-Benefit Analysis</h4>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mb-3">
          <MetricCard metric={display.expectedBenefit} valueClassName="text-emerald-400" />
          <MetricCard metric={display.expectedCost} valueClassName="text-amber-400" />
          <MetricCard metric={display.netBenefit} valueClassName={netBenefitColor} />
        </div>

        <div className="bg-gray-800/60 rounded-lg p-3 border border-gray-700/40">
          <div className="flex items-center justify-between text-xs text-gray-500 mb-2">
            <span>Net Benefit (bps)</span>
            <span>{display.netBenefit.value}</span>
          </div>
          <div className="relative h-2 bg-gray-800 rounded-full overflow-hidden">
            <div
              className="absolute top-0 left-0 h-full rounded-full transition-all duration-500"
              style={{ width: `${gaugeWidth}%`, backgroundColor: gaugeColor }}
            />
          </div>
          <div className="flex justify-between text-[10px] text-gray-600 mt-1">
            <span>-50 bps</span>
            <span>0</span>
            <span>+50 bps</span>
          </div>
        </div>
      </div>

      <div>
        <h4 className="text-sm font-medium text-gray-400 mb-3">Sizing Parameters</h4>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          <MetricCard metric={display.kellyFraction} />
          <MetricCard metric={display.confidenceScaledSize} />
          <MetricCard metric={display.minimumHold} />
        </div>
      </div>

      <div className="text-xs text-gray-500 mt-4">
        <p>Last updated: {display.lastUpdated}</p>
      </div>
    </div>
  );
}
