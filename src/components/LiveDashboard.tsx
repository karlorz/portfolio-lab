import React, { lazy, Suspense, useState, useEffect, useMemo, useRef } from 'react';
import { YieldCurveIndicator } from './YieldCurveIndicator';
import { BondAllocationPanel } from './BondAllocationPanel';
import DurationOverlayPanel from './DurationOverlayPanel';
import type { SignalsData, PerformanceEntry, Alert, AssetStat, DashboardData, HealthData, StatsData, AnalyticsData, GarchCvarData, EntropyData, HedgeSelectorData, IncidentLifecycleSummary } from '../types/live';
import type { RebalanceHealthData } from './RebalanceHealthPanel';
import type { ExplainabilityData } from './PortfolioExplainabilityPanel';
import { BehavioralSentimentPanel } from './BehavioralSentimentPanel';
import { CalendarSeasonalityPanel } from './CalendarSeasonalityPanel';
import { CryptoAllocationPanel } from './CryptoAllocationPanel';
import type { GraduationChecklistData } from './GraduationChecklistPanel';
import type { AdaptiveSizingData } from './AdaptiveSizingPanel';
import type { VixyHedgeSizingData } from './VixyHedgeSizingPanel';
import type { BlackLittermanMapperData } from './BlackLittermanMapperPanel';
import type { TurnoverValidatorData } from './TurnoverValidatorPanel';
import type { RegimeGateData } from './RegimeGatePanel';
import type { TSMOMData } from './TSMOMPanel';
import type { CrossAssetRVData } from './CrossAssetRVPanel';
import { PanelErrorBoundary } from './PanelErrorBoundary';
import type { BehavioralSentimentData } from './BehavioralSentimentPanel';
import type { CryptoData } from './CryptoAllocationPanel';
import type { CalendarData } from './CalendarSeasonalityPanel';
import type { AllocationSurfaceRole, EnsembleVotingData } from './EnsembleVotingPanel';
import type { AlternativeData } from './AlternativeDataPanel';
import type { FactorRotationData } from './FactorRotationPanel';
import type { StackingEnsembleData } from './StackingEnsemblePanel';
import type { ConvexityHarvestData } from './ConvexityHarvestPanel';
import type { LLMSentimentData } from './LLMSentimentPanel';
import type { SectorRotationData } from './SectorRotationPanel';
import type { FactorRotationDashboardData } from './FactorRotationDashboardPanel';
import type { CollarData } from './CollarPanel';
import type { KurtosisData } from './KurtosisRegimePanel';
import type { VolatilityParityData } from './VolatilityParityPanel';
import { summarizeHealthOperations } from './healthOperations';
import { IncidentSummary } from './IncidentSummary';
import {
  buildDashboardIncidents,
  getIncidentsForTab,
  getTabIncidentBadge,
  type IncidentTab,
  type TabIncidentBadge,
} from './dashboardIncidents';
import {
  validateSignalsData,
  validateFetchData,
  DashboardDataSchema,
  AlertsDataSchema,
  StatsDataSchema,
  HealthDataSchema,
  IncidentLifecycleSummarySchema,
  AnalyticsDataSchema,
  RebalanceHealthSchema,
  GraduationDataSchema,
  AdaptiveSizingSchema,
  BlackLittermanSchema,
} from '../schemas/signals';
import { z } from 'zod';

// Catch-all schema for panel endpoints without dedicated schemas
const PassthroughSchema = z.object({}).passthrough();

const HealthPanel = lazy(() => import('./HealthPanel').then((module) => ({ default: module.HealthPanel })));
const RegimeTimeline = lazy(() => import('./RegimeTimeline').then((module) => ({ default: module.RegimeTimeline })));
const SPYComparisonChart = lazy(() => import('./SPYComparisonChart').then((module) => ({ default: module.SPYComparisonChart })));
const RebalancePanel = lazy(() => import('./RebalancePanel').then((module) => ({ default: module.RebalancePanel })));
const SmartRebalancePanel = lazy(() => import('./SmartRebalancePanel').then((module) => ({ default: module.SmartRebalancePanel })));
const BrokerPanel = lazy(() => import('./BrokerPanel').then((module) => ({ default: module.BrokerPanel })));
const RebalanceHealthPanel = lazy(() => import('./RebalanceHealthPanel').then((module) => ({ default: module.RebalanceHealthPanel })));
const UnderwaterChart = lazy(() => import('./AnalyticsCharts').then((module) => ({ default: module.UnderwaterChart })));
const RollingMetricsChart = lazy(() => import('./AnalyticsCharts').then((module) => ({ default: module.RollingMetricsChart })));
const CrisisOverlay = lazy(() => import('./AnalyticsCharts').then((module) => ({ default: module.CrisisOverlay })));
const PortfolioExplainabilityPanel = lazy(() => import('./PortfolioExplainabilityPanel').then((module) => ({ default: module.PortfolioExplainabilityPanel })));
const EnsembleVotingPanel = lazy(() => import('./EnsembleVotingPanel').then((module) => ({ default: module.EnsembleVotingPanel })));
const AlternativeDataPanel = lazy(() => import('./AlternativeDataPanel').then((module) => ({ default: module.AlternativeDataPanel })));
const FactorRotationPanel = lazy(() => import('./FactorRotationPanel').then((module) => ({ default: module.FactorRotationPanel })));
const StackingEnsemblePanel = lazy(() => import('./StackingEnsemblePanel').then((module) => ({ default: module.StackingEnsemblePanel })));
const ConvexityHarvestPanel = lazy(() => import('./ConvexityHarvestPanel').then((module) => ({ default: module.ConvexityHarvestPanel })));
const LLMSentimentPanel = lazy(() => import('./LLMSentimentPanel').then((module) => ({ default: module.LLMSentimentPanel })));
const SectorRotationPanel = lazy(() => import('./SectorRotationPanel').then((module) => ({ default: module.SectorRotationPanel })));
const MLSignalsPanel = lazy(() => import('./MLSignalsPanel').then((module) => ({ default: module.MLSignalsPanel })));
const MarlRuntimeStatusPanel = lazy(() => import('./MarlRuntimeStatusPanel').then((module) => ({ default: module.MarlRuntimeStatusPanel })));
const FactorRotationDashboardPanel = lazy(() => import('./FactorRotationDashboardPanel').then((module) => ({ default: module.FactorRotationDashboardPanel })));
const GraduationChecklistPanel = lazy(() => import('./GraduationChecklistPanel').then((module) => ({ default: module.GraduationChecklistPanel })));
const AdaptiveSizingPanel = lazy(() => import('./AdaptiveSizingPanel').then((module) => ({ default: module.AdaptiveSizingPanel })));
const VixyHedgeSizingPanel = lazy(() => import('./VixyHedgeSizingPanel').then((module) => ({ default: module.VixyHedgeSizingPanel })));
const HedgeSelectorPanel = lazy(() => import('./HedgeSelectorPanel').then((module) => ({ default: module.HedgeSelectorPanel })));
const BlackLittermanMapperPanel = lazy(() => import('./BlackLittermanMapperPanel').then((module) => ({ default: module.BlackLittermanMapperPanel })));
const TurnoverValidatorPanel = lazy(() => import('./TurnoverValidatorPanel').then((module) => ({ default: module.TurnoverValidatorPanel })));
const RegimeGatePanel = lazy(() => import('./RegimeGatePanel').then((module) => ({ default: module.RegimeGatePanel })));
const TSMOMPanel = lazy(() => import('./TSMOMPanel').then((module) => ({ default: module.TSMOMPanel })));
const CrossAssetRVPanel = lazy(() => import('./CrossAssetRVPanel').then((module) => ({ default: module.CrossAssetRVPanel })));
const ModelValidationPanel = lazy(() => import('./ModelValidationPanel').then((module) => ({ default: module.ModelValidationPanel })));
const ZeroDTEPanel = lazy(() => import('./ZeroDTEPanel').then((module) => ({ default: module.ZeroDTEPanel })));
const CollarPanel = lazy(() => import('./CollarPanel').then((module) => ({ default: module.CollarPanel })));
const ClosingAuctionPanel = lazy(() => import('./ClosingAuctionPanel').then((module) => ({ default: module.ClosingAuctionPanel })));
const GarchCvarPanel = lazy(() => import('./GarchCvarPanel').then((module) => ({ default: module.GarchCvarPanel })));
const EntropyPanel = lazy(() => import('./EntropyPanel').then((module) => ({ default: module.EntropyPanel })));
const VIXTermStructurePanel = lazy(() => import('./VIXTermStructurePanel').then((module) => ({ default: module.VIXTermStructurePanel })));
const BondMomentumPanel = lazy(() => import('./BondMomentumPanel').then((module) => ({ default: module.BondMomentumPanel })));
const KurtosisRegimePanel = lazy(() => import('./KurtosisRegimePanel').then((module) => ({ default: module.KurtosisRegimePanel })));
const VolatilityParityPanel = lazy(() => import('./VolatilityParityPanel').then((module) => ({ default: module.VolatilityParityPanel })));
const LabsPanel = lazy(() => import('./LabsPanel').then((module) => ({ default: module.LabsPanel })));
const DecisionReplayPanel = lazy(() =>
  import('./DecisionReplayPanel').then((module) => ({ default: module.DecisionReplayPanel })),
);
const ChatPanel = lazy(() => import('./ChatPanel').then((module) => ({ default: module.ChatPanel })));
const TasksPanel = lazy(() => import('./TasksPanel').then((module) => ({ default: module.TasksPanel })));

function tabLoadingFallback(name: string) {
  return (
    <div className="tab-panel lazy-tab-loading" role="status" aria-live="polite">
      Loading {name}...
    </div>
  );
}

interface LiveDashboardProps {
  refreshInterval?: number; // seconds
}

function formatAllocationSurfaceRoute(role: AllocationSurfaceRole): string {
  const status = role.routed ? 'Order-routed' : 'Not order-routed';
  return role.routed_by ? `${status} via ${role.routed_by}` : status;
}

function isBehavioralSentimentData(value: unknown): value is BehavioralSentimentData {
  if (!value || typeof value !== 'object') return false;
  const data = value as Record<string, unknown>;

  return (
    typeof data.active === 'boolean' &&
    typeof data.composite_score === 'number' &&
    typeof data.signal_type === 'string' &&
    typeof data.confidence === 'number' &&
    typeof data.equity_shift_pct === 'number' &&
    typeof data.z_score === 'number' &&
    typeof data.vix === 'number' &&
    typeof data.regime_suppressed === 'boolean' &&
    typeof data.signal_count_5d === 'number'
  );
}

async function safeParseJson(response: Response, endpoint: string): Promise<unknown | null> {
  if (!response.ok) return null;

  const contentType = response.headers.get('content-type')?.toLowerCase() || '';
  if (!contentType.includes('application/json')) {
    // Static preview servers may return index.html (text/html) for missing JSON assets.
    // Skip parsing so one bad endpoint does not fail the whole dashboard refresh.
    return null;
  }

  try {
    return await response.json();
  } catch {
    if (import.meta.env.DEV) {
      console.warn(`[${endpoint}] Failed to parse JSON response`);
    }
    return null;
  }
}

type TabType = IncidentTab;

export function LiveDashboard({ refreshInterval = 60 }: LiveDashboardProps) {
  const [activeTab, setActiveTab] = useState<TabType>('overview');
  const [signals, setSignals] = useState<SignalsData | null>(null);
  const [performance, setPerformance] = useState<PerformanceEntry[]>([]);
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [stats, setStats] = useState<StatsData | null>(null);
  const [dashboard, setDashboard] = useState<DashboardData | null>(null);
  const [health, setHealth] = useState<HealthData | null>(null);
  const [incidentSummary, setIncidentSummary] = useState<IncidentLifecycleSummary | null>(null);
  const [analytics, setAnalytics] = useState<AnalyticsData | null>(null);
  const [rebalanceHealth, setRebalanceHealth] = useState<RebalanceHealthData | null>(null);
  const [explainability, setExplainability] = useState<ExplainabilityData | null>(null);
  const [lastUpdate, setLastUpdate] = useState<string>('');
  const [error, setError] = useState<string | null>(null);
  const [expandedHealth, setExpandedHealth] = useState(true);
  const [graduationData, setGraduationData] = useState<GraduationChecklistData | null>(null);
  const [adaptiveSizingData, setAdaptiveSizingData] = useState<AdaptiveSizingData | null>(null);
  const [vixyHedgeData, setVixyHedgeData] = useState<VixyHedgeSizingData | null>(null);
  const [hedgeSelectorData, setHedgeSelectorData] = useState<HedgeSelectorData | null>(null);
  const [blMapperData, setBLMapperData] = useState<BlackLittermanMapperData | null>(null);
  const [turnoverData, setTurnoverData] = useState<TurnoverValidatorData | null>(null);
  const [regimeGateData, setRegimeGateData] = useState<RegimeGateData | null>(null);
  const [tsmomData, setTsmomData] = useState<TSMOMData | null>(null);
  const [crossAssetRVData, setCrossAssetRVData] = useState<CrossAssetRVData | null>(null);
  const optionalFetchTimestamps = useRef<Partial<Record<TabType, number>>>({});
  const coreFetchGeneration = useRef(0);
  const optionalFetchGenerations = useRef<Partial<Record<TabType, number>>>({});

  const fetchCoreData = async () => {
    const requestGeneration = ++coreFetchGeneration.current;
    try {
      const [signalsRes, dashboardRes, alertsRes, statsRes, healthRes, incidentsRes] = await Promise.all([
        fetch('/data/signals.json'),
        fetch('/data/dashboard.json'),
        fetch('/data/alerts.json'),
        fetch('/data/stats.json'),
        fetch('/data/health.json'),
        fetch('/data/incidents.json'),
      ]);
      if (requestGeneration !== coreFetchGeneration.current) return;

      const signalsRaw = await safeParseJson(signalsRes, 'signals');
      if (requestGeneration !== coreFetchGeneration.current) return;
      if (signalsRaw) {
        const raw = signalsRaw;
        const validated = validateSignalsData(raw);
        if (validated) {
          setSignals(validated);
          setLastUpdate(new Date(validated.timestamp).toLocaleTimeString());
          setHedgeSelectorData(validated.hedge_selector ?? null);
        }
      }
      const dashboardRaw = await safeParseJson(dashboardRes, 'dashboard');
      if (requestGeneration !== coreFetchGeneration.current) return;
      if (dashboardRaw) {
        const raw = dashboardRaw;
        const validated = validateFetchData(raw, DashboardDataSchema, 'dashboard');
        if (validated) {
          setPerformance(validated.paper_portfolio || []);
          setDashboard(validated as DashboardData);
        }
      }
      const alertsRaw = await safeParseJson(alertsRes, 'alerts');
      if (requestGeneration !== coreFetchGeneration.current) return;
      if (alertsRaw) {
        const raw = alertsRaw;
        const validated = validateFetchData(raw, AlertsDataSchema, 'alerts');
        if (validated) {
          setAlerts(validated.alerts || []);
        }
      }
      const statsRaw = await safeParseJson(statsRes, 'stats');
      if (requestGeneration !== coreFetchGeneration.current) return;
      if (statsRaw) {
        const raw = statsRaw;
        const validated = validateFetchData(raw, StatsDataSchema, 'stats');
        if (validated) setStats(validated as StatsData);
      }
      const healthRaw = await safeParseJson(healthRes, 'health');
      if (requestGeneration !== coreFetchGeneration.current) return;
      if (healthRaw) {
        const raw = healthRaw;
        const validated = validateFetchData(raw, HealthDataSchema, 'health');
        if (validated) setHealth(validated as HealthData);
      }
      const incidentsRaw = await safeParseJson(incidentsRes, 'incidents');
      if (requestGeneration !== coreFetchGeneration.current) return;
      if (incidentsRaw) {
        const raw = incidentsRaw;
        const validated = validateFetchData(raw, IncidentLifecycleSummarySchema, 'incidents');
        if (validated) setIncidentSummary(validated as IncidentLifecycleSummary);
      }

      if (requestGeneration === coreFetchGeneration.current) setError(null);
    } catch (err) {
      if (requestGeneration === coreFetchGeneration.current) setError('Failed to load live data');
    }
  };

  const shouldFetchOptionalTab = (tab: TabType, force: boolean) => {
    if (tab !== 'rebalance' && tab !== 'analytics') {
      return false;
    }
    if (force) {
      return true;
    }
    const lastFetchedAt = optionalFetchTimestamps.current[tab] ?? 0;
    return Date.now() - lastFetchedAt >= refreshInterval * 1000;
  };

  const fetchOptionalDataForTab = async (tab: TabType, force = false) => {
    if (!shouldFetchOptionalTab(tab, force)) {
      return;
    }
    const requestGeneration = (optionalFetchGenerations.current[tab] ?? 0) + 1;
    optionalFetchGenerations.current[tab] = requestGeneration;

    if (tab === 'rebalance') {
      try {
        const rebalanceHealthRes = await fetch('/data/rebalance_health.json');
        const rebalanceHealthRaw = await safeParseJson(rebalanceHealthRes, 'rebalance_health');
        if (optionalFetchGenerations.current[tab] !== requestGeneration) return;
        if (rebalanceHealthRaw) {
          const raw = rebalanceHealthRaw;
          const validated = validateFetchData(raw, RebalanceHealthSchema, 'rebalance_health');
          if (validated) setRebalanceHealth(validated as unknown as RebalanceHealthData);
        }
        if (optionalFetchGenerations.current[tab] !== requestGeneration) return;
        optionalFetchTimestamps.current[tab] = Date.now();
      } catch { /* panels render gracefully with null data */ }
    }

    if (tab === 'analytics') {
      try {
        const [analyticsRes, exRes, gradRes, sizingRes, vixyRes, blRes, turnoverRes] = await Promise.all([
          fetch('/data/analytics.json'),
          fetch('/data/explainability/explainability_latest.json'),
          fetch('/data/graduation.json'),
          fetch('/data/adaptive_sizing.json'),
          fetch('/data/vixy_hedge.json'),
          fetch('/data/black_litterman.json'),
          fetch('/data/turnover_validator.json'),
        ]);
        if (optionalFetchGenerations.current[tab] !== requestGeneration) return;
        const analyticsRaw = await safeParseJson(analyticsRes, 'analytics');
        if (optionalFetchGenerations.current[tab] !== requestGeneration) return;
        if (analyticsRaw) {
          const raw = analyticsRaw;
          const validated = validateFetchData(raw, AnalyticsDataSchema, 'analytics');
          if (validated) setAnalytics(validated as AnalyticsData);
        }
        const explainabilityRaw = await safeParseJson(exRes, 'explainability');
        if (optionalFetchGenerations.current[tab] !== requestGeneration) return;
        if (explainabilityRaw) {
          const raw = explainabilityRaw;
          const validated = validateFetchData(raw, PassthroughSchema, 'explainability');
          if (validated) setExplainability(validated as unknown as ExplainabilityData);
        }
        const graduationRaw = await safeParseJson(gradRes, 'graduation');
        if (optionalFetchGenerations.current[tab] !== requestGeneration) return;
        if (graduationRaw) {
          const raw = graduationRaw;
          const validated = validateFetchData(raw, GraduationDataSchema, 'graduation');
          if (validated) setGraduationData(validated as unknown as GraduationChecklistData);
        }
        const adaptiveSizingRaw = await safeParseJson(sizingRes, 'adaptive_sizing');
        if (optionalFetchGenerations.current[tab] !== requestGeneration) return;
        if (adaptiveSizingRaw) {
          const raw = adaptiveSizingRaw;
          const validated = validateFetchData(raw, AdaptiveSizingSchema, 'adaptive_sizing');
          if (validated) setAdaptiveSizingData(validated as unknown as AdaptiveSizingData);
        }
        const vixyRaw = await safeParseJson(vixyRes, 'vixy_hedge');
        if (optionalFetchGenerations.current[tab] !== requestGeneration) return;
        if (vixyRaw) {
          const raw = vixyRaw;
          const validated = validateFetchData(raw, PassthroughSchema, 'vixy_hedge');
          if (validated) setVixyHedgeData(validated as unknown as VixyHedgeSizingData);
        }
        const blackLittermanRaw = await safeParseJson(blRes, 'black_litterman');
        if (optionalFetchGenerations.current[tab] !== requestGeneration) return;
        if (blackLittermanRaw) {
          const raw = blackLittermanRaw;
          const validated = validateFetchData(raw, BlackLittermanSchema, 'black_litterman');
          if (validated) setBLMapperData(validated as unknown as BlackLittermanMapperData);
        }
        const turnoverRaw = await safeParseJson(turnoverRes, 'turnover_validator');
        if (optionalFetchGenerations.current[tab] !== requestGeneration) return;
        if (turnoverRaw) {
          const raw = turnoverRaw;
          const validated = validateFetchData(raw, PassthroughSchema, 'turnover_validator');
          if (validated) setTurnoverData(validated as unknown as TurnoverValidatorData);
        }
      } catch { /* panels render gracefully with null data */ }

      try {
        const [rgRes, tsmomRes, rvRes] = await Promise.all([
          fetch('/data/regime_gate.json'),
          fetch('/data/tsmom.json'),
          fetch('/data/cross_asset_rv.json'),
        ]);
        if (optionalFetchGenerations.current[tab] !== requestGeneration) return;
        const regimeGateRaw = await safeParseJson(rgRes, 'regime_gate');
        if (optionalFetchGenerations.current[tab] !== requestGeneration) return;
        if (regimeGateRaw) {
          const raw = regimeGateRaw;
          const validated = validateFetchData(raw, PassthroughSchema, 'regime_gate');
          if (validated) setRegimeGateData(validated as unknown as RegimeGateData);
        }
        const tsmomRaw = await safeParseJson(tsmomRes, 'tsmom');
        if (optionalFetchGenerations.current[tab] !== requestGeneration) return;
        if (tsmomRaw) {
          const raw = tsmomRaw;
          const validated = validateFetchData(raw, PassthroughSchema, 'tsmom');
          if (validated) setTsmomData(validated as unknown as TSMOMData);
        }
        const crossAssetRVRaw = await safeParseJson(rvRes, 'cross_asset_rv');
        if (optionalFetchGenerations.current[tab] !== requestGeneration) return;
        if (crossAssetRVRaw) {
          const raw = crossAssetRVRaw;
          const validated = validateFetchData(raw, PassthroughSchema, 'cross_asset_rv');
          if (validated) setCrossAssetRVData(validated as unknown as CrossAssetRVData);
        }
      } catch { /* panels render gracefully with null data */ }

      if (optionalFetchGenerations.current[tab] !== requestGeneration) return;
      optionalFetchTimestamps.current[tab] = Date.now();
    }
  };

  useEffect(() => {
    fetchCoreData();
    const interval = setInterval(fetchCoreData, refreshInterval * 1000);
    return () => clearInterval(interval);
  }, [refreshInterval]);

  useEffect(() => {
    fetchOptionalDataForTab(activeTab);
    const interval = setInterval(() => fetchOptionalDataForTab(activeTab, true), refreshInterval * 1000);
    return () => clearInterval(interval);
  }, [activeTab, refreshInterval]);

  const portfolioValue = useMemo(() => {
    return signals?.total_value || 100000;
  }, [signals]);

  const regimeColor = useMemo(() => {
    const r = signals?.regime?.regime;
    switch (r) {
      case 'crisis': return '#ef4444';
      case 'vol_spike': return '#f59e0b';
      case 'low_vol': return '#10b981';
      default: return '#3b82f6';
    }
  }, [signals]);

  const formatCurrency = (v: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0
    }).format(v);
  };

  const formatPct = (v: number) => `${(v * 100).toFixed(1)}%`;
  const behavioralSentimentData = isBehavioralSentimentData(signals?.behavioral_sentiment)
    ? signals.behavioral_sentiment
    : null;

  const healthOperationsSummary = health ? summarizeHealthOperations(health) : null;
  const dashboardIncidents = useMemo(
    () => buildDashboardIncidents({ alerts, signals, health, incidentSummary }),
    [alerts, signals, health, incidentSummary],
  );
  const overviewIncidents = useMemo(
    () => getIncidentsForTab(dashboardIncidents, 'overview'),
    [dashboardIncidents],
  );
  const healthIncidents = useMemo(
    () => getIncidentsForTab(dashboardIncidents, 'health'),
    [dashboardIncidents],
  );
  const riskIncidents = useMemo(
    () => getIncidentsForTab(dashboardIncidents, 'risk'),
    [dashboardIncidents],
  );
  const targetAllocationRole = signals?.allocation_surface_roles?.surfaces.target_allocations;
  const ensembleVotingRole = signals?.allocation_surface_roles?.surfaces.ensemble_voting;

  const countBadge = (count: number | undefined, severity: TabIncidentBadge['severity']): TabIncidentBadge | undefined => {
    return count && count > 0 ? { count, severity } : undefined;
  };

  const tabs: { id: TabType; label: string; badge?: TabIncidentBadge }[] = [
    { id: 'overview', label: 'Overview', badge: getTabIncidentBadge(dashboardIncidents, 'overview') },
    { id: 'health', label: 'Health', badge: getTabIncidentBadge(dashboardIncidents, 'health') },
    { id: 'risk', label: 'Risk', badge: getTabIncidentBadge(dashboardIncidents, 'risk') },
    { id: 'history', label: 'History' },
    { id: 'performance', label: 'Performance' },
    { id: 'rebalance', label: 'Rebalance' },
    { id: 'analytics', label: 'Analytics' },
    { id: 'options', label: '0DTE', badge: countBadge(signals?.zero_dte?.positions?.length, 'info') },
    { id: 'auction', label: 'Auction', badge: countBadge(signals?.closing_auction?.signals?.filter(s => s.should_trade).length, 'warning') },
    { id: 'labs', label: 'Labs' },
    { id: 'decisions', label: 'Decisions', badge: getTabIncidentBadge(dashboardIncidents, 'decisions') },
    { id: 'tasks', label: 'Tasks' },
    { id: 'chat', label: 'Chat' }
  ];

  return (
    <div className="live-dashboard">
      {/* Header */}
      <div className="dashboard-header">
        <div className="header-main">
          <h2>Live Paper Trading</h2>
          <div className="status-bar">
            <span
              className="regime-badge"
              style={{ backgroundColor: regimeColor }}
            >
              {signals?.regime?.regime?.toUpperCase() || 'LOADING'}
            </span>
            <span className="last-update">Updated: {lastUpdate || 'Never'}</span>
            {error && <span className="error">{error}</span>}
          </div>
        </div>

        {/* Health Summary (always visible) */}
        {health && (
          <div 
            className={`health-summary-bar status-${health.system_status}`}
            onClick={() => setActiveTab('health')}
          >
            <span className="health-indicator"></span>
            <span className="health-text">
              {healthOperationsSummary?.headerText}
            </span>
          </div>
        )}
      </div>

      {/* Tabs */}
      <div className="dashboard-tabs">
        {tabs.map(tab => (
          <button
            key={tab.id}
            className={`tab ${activeTab === tab.id ? 'active' : ''}`}
            onClick={() => setActiveTab(tab.id)}
          >
            {tab.label}
            {tab.badge !== undefined && tab.badge.count > 0 && (
              <span className={`tab-badge tab-badge-${tab.badge.severity}`}>{tab.badge.count}</span>
            )}
          </button>
        ))}
      </div>

      {/* Tab Content */}
      <div className="tab-content">
        {/* Overview Tab */}
        {activeTab === 'overview' && (
        <PanelErrorBoundary name="Overview">
          <div className="tab-panel overview-panel">
            <IncidentSummary
              title="Action Center"
              incidents={overviewIncidents}
              showTab
              onIncidentSelect={(incident) => setActiveTab(incident.tab)}
            />

            {/* Portfolio Summary */}
            <div className="metrics-grid">
              <div className="metric-card primary">
                <label>Portfolio Value</label>
                <span className="value-display">{formatCurrency(portfolioValue)}</span>
                {signals?.cash && (
                  <small>Cash: {formatCurrency(signals.cash)}</small>
                )}
              </div>

              <div className="metric-card">
                <label>Regime</label>
                <span className="value-display" style={{ color: regimeColor }}>
                  {signals?.regime?.regime?.toUpperCase()}
                </span>
                {signals?.regime?.vix && (
                  <small>VIX: {signals.regime.vix.toFixed(1)}</small>
                )}
              </div>

              <div className="metric-card">
                <label>Target Allocation</label>
                <div className="alloc-preview">
                  {signals?.target_allocations && Object.entries(signals.target_allocations)
                    .map(([sym, weight]) => (
                      <span key={sym} className="alloc-tag">
                        {sym}: {formatPct(weight as number)}
                      </span>
                    ))
                  }
                </div>
                {targetAllocationRole && (
                  <small style={{ overflowWrap: 'anywhere' }}>
                    {formatAllocationSurfaceRoute(targetAllocationRole)}
                  </small>
                )}
              </div>
            </div>

            {/* Positions & Orders */}
            <div className="positions-orders-row">
              {signals?.current_positions && signals.current_positions.length > 0 && (
                <div className="positions-section">
                  <h3>Current Positions</h3>
                  <table className="positions-table">
                    <thead>
                      <tr>
                        <th>Symbol</th>
                        <th>Shares</th>
                        <th>Value</th>
                        <th>Weight</th>
                        <th>P&L</th>
                      </tr>
                    </thead>
                    <tbody>
                      {signals.current_positions.map((pos) => (
                        <tr key={pos.symbol}>
                          <td><strong>{pos.symbol}</strong></td>
                          <td>{pos.shares.toFixed(2)}</td>
                          <td>{formatCurrency(pos.value)}</td>
                          <td>{formatPct(pos.weight)}</td>
                          <td className={pos.unrealized >= 0 ? 'positive' : 'negative'}>
                            {formatCurrency(pos.unrealized)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}

              {signals?.recent_orders && signals.recent_orders.length > 0 && (
                <div className="orders-section">
                  <h3>Recent Orders</h3>
                  <table className="orders-table">
                    <thead>
                      <tr>
                        <th>Symbol</th>
                        <th>Side</th>
                        <th>Shares</th>
                        <th>Value</th>
                      </tr>
                    </thead>
                    <tbody>
                      {signals.recent_orders.map((order, i) => (
                        <tr key={i}>
                          <td><strong>{order.sym}</strong></td>
                          <td className={order.side === 'buy' ? 'positive' : 'negative'}>
                            {order.side.toUpperCase()}
                          </td>
                          <td>{order.shares.toFixed(2)}</td>
                          <td>{formatCurrency(order.value)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>

            {/* Yield Curve & Bond Allocation */}
            <div className="yield-bond-row">
              <YieldCurveIndicator 
                spread2s10s={signals?.yield_curve?.spread2s10s ?? null}
                regime={signals?.yield_curve?.duration_regime ?? null}
                spreadHistory={signals?.yield_curve?.spread_history}
                lastUpdate={lastUpdate}
                sourceMode={signals?.yield_curve?.source_mode}
                sourceStatus={signals?.yield_curve?.source_status}
                sourceReason={signals?.yield_curve?.source_reason}
              />
              <BondAllocationPanel 
                currentAllocation={signals?.duration_allocation ?? null}
                targetAllocation={signals?.duration_allocation ?? null}
                regime={signals?.yield_curve?.duration_regime ?? null}
                portfolioValue={portfolioValue}
                bondSlicePct={0.16}
              />
            </div>

            {/* Duration Overlay Panel */}
            <DurationOverlayPanel
              yieldCurve={signals?.yield_curve ?? null}
              durationAllocation={signals?.duration_allocation ?? null}
            />

            {/* Signal Panels */}
            <div className="signal-panels-row">
              {behavioralSentimentData && (
                <div className="signal-panel-slot">
                  <BehavioralSentimentPanel data={behavioralSentimentData} />
                </div>
              )}
              <div className="signal-panel-slot">
                <CryptoAllocationPanel
                  data={signals?.crypto_allocation ?? null}
                  portfolioValue={portfolioValue}
                />
              </div>
              <div className="signal-panel-slot">
                <CalendarSeasonalityPanel data={signals?.calendar_seasonality ?? null} />
              </div>
            </div>
          </div>
        </PanelErrorBoundary>
        )}

        {/* Health Tab */}
        {activeTab === 'health' && (
        <PanelErrorBoundary name="Health">
          <Suspense fallback={tabLoadingFallback('Health')}>
          <div className="tab-panel health-panel-container">
            <IncidentSummary
              title="Health Incidents"
              incidents={healthIncidents}
            />
            <HealthPanel
              health={health}
              expanded={expandedHealth}
              onToggleExpand={() => setExpandedHealth(!expandedHealth)}
            />
          </div>
          </Suspense>
        </PanelErrorBoundary>
        )}

        {/* History Tab */}
        {activeTab === 'history' && (
        <PanelErrorBoundary name="History">
          <Suspense fallback={tabLoadingFallback('History')}>
          <div className="tab-panel history-panel">
            <RegimeTimeline history={dashboard?.regimes || []} />
          </div>
          </Suspense>
        </PanelErrorBoundary>
        )}

        {/* Performance Tab */}
        {activeTab === 'performance' && (
        <PanelErrorBoundary name="Performance">
          <Suspense fallback={tabLoadingFallback('Performance')}>
          <div className="tab-panel performance-panel">
            {/* SPY Comparison Chart */}
            <SPYComparisonChart stats={stats} performance={performance} />

            {/* Paper Portfolio Stats */}
            {performance.length > 0 && (
              <div className="performance-summary">
                <div className="perf-card">
                  <label>Current Value</label>
                  <span className="value-display">
                    {formatCurrency(performance[performance.length - 1]?.v || 100000)}
                  </span>
                </div>
                <div className="perf-card">
                  <label>Start Value</label>
                  <span className="value-display">
                    {formatCurrency(performance[0]?.v || 100000)}
                  </span>
                </div>
                <div className="perf-card">
                  <label>Days Tracked</label>
                  <span className="value-display">{performance.length}</span>
                </div>
                <div className="perf-card">
                  <label>Total Return</label>
                  <span className={`value-display ${
                    ((performance[performance.length - 1]?.v || 100000) - 100000) >= 0 ? 'positive' : 'negative'
                  }`}>
                    {formatCurrency((performance[performance.length - 1]?.v || 100000) - 100000)}
                  </span>
                </div>
              </div>
            )}

            {/* Asset Stats */}
            {stats?.asset_stats && Object.keys(stats.asset_stats).length > 0 && (
              <div className="stats-section">
                <h3>Market Overview (30d)</h3>
                <div className="stats-grid">
                  {Object.entries(stats.asset_stats).map(([sym, stat]) => (
                    <div key={sym} className="stat-card">
                      <h4>{sym}</h4>
                      <div className="stat-value">
                        <span className={stat['30d_return'] >= 0 ? 'positive' : 'negative'}>
                          {stat['30d_return'] >= 0 ? '+' : ''}{stat['30d_return'].toFixed(1)}%
                        </span>
                        <small>30d return</small>
                      </div>
                      <div className="stat-vol">
                        <span>Vol: {stat.volatility.toFixed(1)}%</span>
                        <small>${stat.current.toFixed(2)}</small>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
          </Suspense>
        </PanelErrorBoundary>
        )}

        {/* Rebalance Tab */}
        {activeTab === 'rebalance' && (
        <PanelErrorBoundary name="Rebalance">
          <Suspense fallback={tabLoadingFallback('Rebalance')}>
          <div className="tab-panel rebalance-panel-container">
            <BrokerPanel data={signals?.broker} />
            <SmartRebalancePanel data={signals?.smart_rebalance} />
            <RebalanceHealthPanel
              rebalanceData={signals?.smart_rebalance}
              healthData={rebalanceHealth}
            />
            <RebalancePanel
              signals={signals}
              readOnly={true}
              onRebalanceRequest={() => {
                // In paper mode, rebalancing is automatic via cron
                // This would trigger manual rebalance in future live mode
                console.log('Manual rebalance requested');
              }}
            />
          </div>
          </Suspense>
        </PanelErrorBoundary>
        )}

        {/* Analytics Tab */}
        {activeTab === 'analytics' && (
        <PanelErrorBoundary name="Analytics">
          <Suspense fallback={tabLoadingFallback('Analytics')}>
          <div className="tab-panel analytics-panel">
            {analytics?.status === 'success' ? (
              <>
                {/* Underwater Chart - Drawdown */}
                {analytics.drawdown?.series?.length > 0 && (
                  <UnderwaterChart 
                    series={analytics.drawdown.series}
                    maxDrawdown={analytics.drawdown.max_drawdown}
                  />
                )}

                {/* Rolling Metrics */}
                {(analytics.rolling_metrics?.sharpe_63d?.length > 0 || 
                  analytics.rolling_metrics?.sharpe_126d?.length > 0 ||
                  analytics.rolling_metrics?.sharpe_252d?.length > 0) && (
                  <RollingMetricsChart
                    sharpe63d={analytics.rolling_metrics.sharpe_63d}
                    sharpe126d={analytics.rolling_metrics.sharpe_126d}
                    sharpe252d={analytics.rolling_metrics.sharpe_252d}
                  />
                )}

                {/* Crisis Periods */}
                {analytics.crisis_periods?.length > 0 && (
                  <CrisisOverlay periods={analytics.crisis_periods} />
                )}

                {/* Data Summary */}
                <div className="analytics-summary">
                  <div className="analytics-card">
                    <label>Data Points</label>
                    <span>{analytics.data_points}</span>
                  </div>
                  <div className="analytics-card">
                    <label>Date Range</label>
                    <span>{analytics.date_range.start} to {analytics.date_range.end}</span>
                  </div>
                  <div className="analytics-card">
                    <label>Max Drawdown</label>
                    <span className={analytics.drawdown?.max_drawdown?.max_drawdown < -15 ? 'negative' : ''}>
                      {analytics.drawdown?.max_drawdown?.max_drawdown?.toFixed(2)}%
                    </span>
                  </div>
                  {analytics.drawdown?.max_drawdown?.recovery_date ? (
                    <div className="analytics-card">
                      <label>Recovered</label>
                      <span className="positive">Yes</span>
                    </div>
                  ) : (
                    <div className="analytics-card">
                      <label>Underwater Days</label>
                      <span className="warning">{analytics.drawdown?.max_drawdown?.underwater_days}</span>
                    </div>
                  )}
                </div>
              </>
            ) : (
              <div className="analytics-empty">
                <p>{analytics?.message || 'Analytics data not available'}</p>
                <small>Data points: {analytics?.data_points || 0}</small>
              </div>
            )}
            <PanelErrorBoundary name="Analytics/Explainability">
              <PortfolioExplainabilityPanel data={explainability} />
            </PanelErrorBoundary>

            <div className="dashboard-section-stack">
              {/* Model & Ensemble Panels */}
              <div className="dashboard-grid dashboard-grid-two analytics-panel-group">
                <PanelErrorBoundary name="Analytics/Ensemble Voting">
                  <EnsembleVotingPanel
                    data={signals?.ensemble_voting ?? null}
                    allocationSurfaceRole={ensembleVotingRole}
                  />
                </PanelErrorBoundary>
                <PanelErrorBoundary name="Analytics/Alternative Data">
                  <AlternativeDataPanel data={signals?.alternative_data ?? null} />
                </PanelErrorBoundary>
              </div>
              <div className="dashboard-grid dashboard-grid-two analytics-panel-group">
                <PanelErrorBoundary name="Analytics/Factor Rotation">
                  <FactorRotationPanel data={(signals?.factor_rotation ?? null) as unknown as FactorRotationData | null} />
                </PanelErrorBoundary>
                <PanelErrorBoundary name="Analytics/Stacking Ensemble">
                  <StackingEnsemblePanel data={signals?.stacking_ensemble ?? null} />
                </PanelErrorBoundary>
              </div>
              <div className="dashboard-grid dashboard-grid-two analytics-panel-group">
                <PanelErrorBoundary name="Analytics/Convexity Harvest">
                  <ConvexityHarvestPanel data={signals?.convexity_harvest ?? null} />
                </PanelErrorBoundary>
                <PanelErrorBoundary name="Analytics/LLM Sentiment">
                  <LLMSentimentPanel data={signals?.llm_sentiment ?? null} />
                </PanelErrorBoundary>
              </div>
              <div className="analytics-panel-group">
                <PanelErrorBoundary name="Analytics/Sector Rotation">
                  <SectorRotationPanel data={signals?.sector_rotation ?? null} />
                </PanelErrorBoundary>
              </div>
              <div className="dashboard-grid dashboard-grid-three analytics-panel-group">
                <PanelErrorBoundary name="Analytics/ML Signals">
                  <MLSignalsPanel data={signals?.ml_signals ?? null} />
                </PanelErrorBoundary>
                <PanelErrorBoundary name="Analytics/MARL Runtime Status">
                  <MarlRuntimeStatusPanel data={signals?.marl_status ?? null} />
                </PanelErrorBoundary>
                <PanelErrorBoundary name="Analytics/Factor Rotation Dashboard">
                  <FactorRotationDashboardPanel data={signals?.factor_rotation_dashboard ?? null} />
                </PanelErrorBoundary>
              </div>
              <div className="dashboard-grid dashboard-grid-three analytics-panel-group">
                <PanelErrorBoundary name="Analytics/Graduation Checklist">
                  <GraduationChecklistPanel data={graduationData} />
                </PanelErrorBoundary>
                <PanelErrorBoundary name="Analytics/Adaptive Sizing">
                  <AdaptiveSizingPanel data={adaptiveSizingData} />
                </PanelErrorBoundary>
                <PanelErrorBoundary name="Analytics/VIXY Hedge Sizing">
                  <VixyHedgeSizingPanel data={vixyHedgeData} />
                </PanelErrorBoundary>
                <PanelErrorBoundary name="Analytics/Hedge Selector">
                  <HedgeSelectorPanel data={hedgeSelectorData} />
                </PanelErrorBoundary>
              </div>
              <div className="dashboard-grid dashboard-grid-two analytics-panel-group">
                <PanelErrorBoundary name="Analytics/Black-Litterman">
                  <BlackLittermanMapperPanel data={blMapperData} />
                </PanelErrorBoundary>
                <PanelErrorBoundary name="Analytics/Turnover Validator">
                  <TurnoverValidatorPanel data={turnoverData} />
                </PanelErrorBoundary>
              </div>
              <div className="dashboard-grid dashboard-grid-three analytics-panel-group">
                <PanelErrorBoundary name="Analytics/Regime Gate">
                  <RegimeGatePanel data={regimeGateData} />
                </PanelErrorBoundary>
                <PanelErrorBoundary name="Analytics/TSMOM">
                  <TSMOMPanel data={tsmomData} />
                </PanelErrorBoundary>
                <PanelErrorBoundary name="Analytics/Cross Asset RV">
                  <CrossAssetRVPanel data={crossAssetRVData} />
                </PanelErrorBoundary>
              </div>
              <div className="analytics-panel-group">
                <PanelErrorBoundary name="Analytics/Model Validation">
                  <ModelValidationPanel
                    dsr={null}
                    championSharpe={stats?.paper_portfolio?.sharpe ?? null}
                    blWeights={null}
                    overlayWeights={signals?.target_allocations ?? null}
                  />
                </PanelErrorBoundary>
              </div>
            </div>
          </div>
          </Suspense>
        </PanelErrorBoundary>
        )}

        {/* 0DTE Options Tab */}
        {activeTab === 'options' && (
        <PanelErrorBoundary name="0DTE Options">
          <Suspense fallback={tabLoadingFallback('0DTE Options')}>
          <div className="tab-panel options-panel">
            <ZeroDTEPanel
              positions={signals?.zero_dte?.positions || []}
              config={signals?.zero_dte?.config || null}
              portfolioValue={portfolioValue}
              vix={signals?.regime?.vix || null}
              weeklyLimitRemaining={2 - (signals?.zero_dte?.weekly_trades_used || 0)}
            />
            <div className="options-panel-section">
              <CollarPanel
                data={signals?.collar ?? null}
                spyPrice={signals?.latest_prices?.SPY}
              />
            </div>
          </div>
          </Suspense>
        </PanelErrorBoundary>
        )}

        {/* Closing Auction Tab */}
        {activeTab === 'auction' && (
        <PanelErrorBoundary name="Closing Auction">
          <Suspense fallback={tabLoadingFallback('Closing Auction')}>
          <div className="tab-panel auction-panel">
            <ClosingAuctionPanel
              signals={signals?.closing_auction?.signals || []}
              isMarketOpen={signals?.closing_auction?.market_open || false}
            />
          </div>
          </Suspense>
        </PanelErrorBoundary>
        )}

        {/* Risk Tab (GARCH-CVaR + Entropy + VIX Term Structure + Bond Momentum) */}
        {activeTab === 'risk' && (
        <PanelErrorBoundary name="Risk">
          <Suspense fallback={tabLoadingFallback('Risk')}>
          <div className="tab-panel risk-panel">
            <IncidentSummary
              title="Risk Incidents"
              incidents={riskIncidents}
            />
            <div className="risk-primary-grid">
              <GarchCvarPanel data={signals?.garch_cvar as GarchCvarData | null | undefined} />
              <EntropyPanel data={signals?.entropy as EntropyData | null | undefined} />
            </div>
            <div className="risk-panel-section">
              <VIXTermStructurePanel 
                data={signals?.vix_term_structure ?? null}
                overlayState={signals?.vix_overlay ?? null}
              />
            </div>
            <div className="risk-panel-section">
              <BondMomentumPanel
                signals={signals?.bond_momentum?.signals || []}
                timestamp={signals?.bond_momentum?.timestamp}
                ensembleRecommendation={signals?.bond_momentum?.ensemble}
              />
            </div>
            <div className="risk-panel-section">
              <KurtosisRegimePanel data={signals?.kurtosis_regime ?? null} />
            </div>
            <div className="risk-panel-section">
              <VolatilityParityPanel data={(signals?.volatility_parity ?? null) as unknown as VolatilityParityData | null} />
            </div>
          </div>
          </Suspense>
        </PanelErrorBoundary>
        )}

        {/* Labs Tab */}
        {activeTab === 'labs' && (
        <PanelErrorBoundary name="Labs">
          <Suspense fallback={tabLoadingFallback('Labs')}>
          <div className="tab-panel labs-panel-container">
            <LabsPanel />
          </div>
          </Suspense>
        </PanelErrorBoundary>
        )}

        {/* Decision Replay Tab */}
        {activeTab === 'decisions' && (
        <PanelErrorBoundary name="Decisions">
          <Suspense fallback={tabLoadingFallback('Decisions')}>
          <div className="tab-panel decisions-panel-container">
            <DecisionReplayPanel />
          </div>
          </Suspense>
        </PanelErrorBoundary>
        )}

        {/* Tasks Tab */}
        {activeTab === 'tasks' && (
        <PanelErrorBoundary name="Tasks">
          <Suspense fallback={tabLoadingFallback('Tasks')}>
          <div className="tab-panel tasks-panel-container">
            <TasksPanel />
          </div>
          </Suspense>
        </PanelErrorBoundary>
        )}

        {/* Chat Tab */}
        {activeTab === 'chat' && (
        <PanelErrorBoundary name="Chat">
          <Suspense fallback={tabLoadingFallback('Chat')}>
          <div className="tab-panel chat-panel-container">
            <ChatPanel />
          </div>
          </Suspense>
        </PanelErrorBoundary>
        )}
      </div>
    </div>
  );
}
