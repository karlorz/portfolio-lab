import { useEffect, useMemo, useState } from 'react';
import { fetchLabsDashboardDataFromIndex } from '../data/labs';
import type {
  LabsDashboardData,
  LabsEndpointKey,
  LabsEndpointStatus,
  LabsExperimentDiffData,
  LabsRegistryRow,
  LabsReplayData,
  LabsScorecardData,
  LabsValidationResult,
} from '../schemas/labs';

export type LabsSortKey =
  | 'sharpe'
  | 'max_drawdown'
  | 'wfe'
  | 'dsr'
  | 'status'
  | 'provenance_status'
  | 'replay_status'
  | 'validation_status';

type SortDirection = 'asc' | 'desc';
type ValidationStatus = 'valid' | 'invalid' | 'missing';
type OptionalFilter = string | 'all';
type EndpointBadgeTone = 'success' | 'info' | 'warning' | 'danger';

const MAX_REPLAY_DIAGNOSTIC_LENGTH = 240;

export interface LabsPanelFilters {
  status?: OptionalFilter;
  provenanceStatus?: OptionalFilter;
  replayStatus?: OptionalFilter;
  validationStatus?: ValidationStatus | 'all';
  searchText?: string;
  minSharpe?: number | null;
  maxDrawdownPct?: number | null;
  sortBy?: LabsSortKey;
  sortDirection?: SortDirection;
}

interface LabsSortOption {
  value: LabsSortKey;
  label: string;
}

export interface LabsPanelReplayDiagnostics {
  failureReason: LabsReplayData['failure_reason'] | null;
  errorType: string | null;
  errorMessage: string | null;
  durationSeconds: number | null;
  details: string[];
}

export interface LabsPanelRow {
  experimentId: string;
  artifactPath: string;
  status: LabsRegistryRow['status'];
  provenanceStatus: LabsRegistryRow['provenance_status'];
  governanceState: NonNullable<LabsRegistryRow['governance_state']>;
  governanceReasons: string[];
  scorecardStatus: LabsScorecardData['status'] | 'missing';
  replayStatus: LabsReplayData['status'] | 'missing';
  replayDiagnostics: LabsPanelReplayDiagnostics | null;
  validationStatus: ValidationStatus;
  sharpe: number | null;
  maxDrawdownPct: number | null;
  cagrPct: number | null;
  wfe: number | null;
  dsr: number | null;
  positiveOosRatio: number | null;
  regimeCoverage: number | null;
  validationErrors: string[];
  validationErrorCount: number;
  omittedValidationErrorCount: number;
}

export interface LabsPanelEndpointBadge {
  endpoint: LabsEndpointKey;
  label: string;
  tone: EndpointBadgeTone;
  details: string[];
}

export interface LabsPanelPaginationControlPage {
  page: number;
  path: string;
  rowCount: number | null;
}

export interface LabsPanelPaginationControl {
  endpoint: LabsEndpointKey;
  selectedPage: number;
  pages: LabsPanelPaginationControlPage[];
}

export interface LabsPanelValidationTruncation {
  totalResultCount: number;
  returnedResultCount: number;
  omittedResultCount: number;
  omittedErrorCount: number;
  maxErrorsPerResult: number;
}

export interface LabsPanelDiffMetricDelta {
  metric: string;
  left: number;
  right: number;
  delta: number;
}

export interface LabsPanelMissingMetric {
  metric: string;
  missingFrom: Array<'left' | 'right'>;
}

export interface LabsPanelConfigChange {
  key: string;
  left: unknown;
  right: unknown;
}

export interface LabsPanelDiff {
  title: string;
  metricDeltas: LabsPanelDiffMetricDelta[];
  missingMetrics: LabsPanelMissingMetric[];
  configChanges: LabsPanelConfigChange[];
  provenanceChange: {
    left: string;
    right: string;
    changed: boolean;
  };
}

export interface LabsPanelViewModel {
  disabled: boolean;
  emptyMessage: string;
  missingEndpoints: LabsEndpointKey[];
  errors: string[];
  endpointStatus: LabsEndpointStatus[];
  endpointBadges: LabsPanelEndpointBadge[];
  summaryLimitedEndpoints: LabsEndpointStatus[];
  paginationControls: LabsPanelPaginationControl[];
  validationTruncation: LabsPanelValidationTruncation | null;
  rows: LabsPanelRow[];
  diffs: LabsPanelDiff[];
  sortOptions: LabsSortOption[];
  filterOptions: {
    status: string[];
    provenanceStatus: string[];
    replayStatus: string[];
    validationStatus: Array<ValidationStatus | 'all'>;
  };
  summary: {
    available: boolean;
    experiments: number;
    scorecards: number;
    replays: number;
    invalidArtifacts: number;
    missingProvenanceCandidates: number;
    summaryLimitedEndpoints: number;
    indexedRows: number;
  };
}

export const LABS_SORT_OPTIONS: LabsSortOption[] = [
  { value: 'sharpe', label: 'Sharpe' },
  { value: 'max_drawdown', label: 'Max Drawdown' },
  { value: 'wfe', label: 'WFE' },
  { value: 'dsr', label: 'DSR' },
  { value: 'status', label: 'Status' },
  { value: 'provenance_status', label: 'Provenance' },
  { value: 'replay_status', label: 'Replay' },
  { value: 'validation_status', label: 'Validation' },
];

const DEFAULT_FILTERS: Required<LabsPanelFilters> = {
  status: 'all',
  provenanceStatus: 'all',
  replayStatus: 'all',
  validationStatus: 'all',
  searchText: '',
  minSharpe: null,
  maxDrawdownPct: null,
  sortBy: 'sharpe',
  sortDirection: 'desc',
};

function uniqueSorted(values: string[]): string[] {
  return ['all', ...Array.from(new Set(values)).sort()];
}

function metricValue(row: LabsRegistryRow, key: string): number | null {
  const value = row.metrics[key];
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function validationForRow(
  row: LabsRegistryRow,
  validationResults: LabsValidationResult[],
): LabsValidationResult | null {
  const byExperimentId = validationResults.find((result) => result.experiment_id === row.experiment_id);
  if (byExperimentId) return byExperimentId;

  const byArtifactPath = validationResults.find((result) => {
    if (!result.artifact_path) return false;
    return (
      result.artifact_path === row.artifact_path ||
      result.artifact_path.endsWith(row.artifact_path) ||
      row.artifact_path.endsWith(result.artifact_path)
    );
  });
  if (byArtifactPath) return byArtifactPath;

  return validationResults.find((result) => {
    if (result.experiment_id || result.artifact_path) return false;
    if (!result.path) return false;
    return (
      result.path === row.artifact_path ||
      result.path.endsWith(row.artifact_path) ||
      result.path.includes(row.experiment_id)
    );
  }) ?? null;
}

function boundedDiagnosticText(value: string | undefined): string | null {
  if (!value) {
    return null;
  }
  const trimmed = value.trim();
  if (!trimmed) {
    return null;
  }
  const redacted = trimmed
    .replace(
      /\b[A-Z][A-Z0-9_]*(?:SECRET|TOKEN|PASSWORD|KEY|CREDENTIAL)[A-Z0-9_]*\s*=\s*[^\s,;]+/gi,
      '[redacted]',
    )
    .replace(
      /\b(?:token|secret|password|api[_-]?key|credential)\s*=\s*[^\s,;]+/gi,
      '[redacted]',
    );

  return redacted.length <= MAX_REPLAY_DIAGNOSTIC_LENGTH
    ? redacted
    : `${redacted.slice(0, MAX_REPLAY_DIAGNOSTIC_LENGTH - 3).trimEnd()}...`;
}

function replayDurationSeconds(replay: LabsReplayData): number | null {
  return typeof replay.duration_seconds === 'number' && Number.isFinite(replay.duration_seconds)
    ? replay.duration_seconds
    : null;
}

function buildReplayDiagnostics(replay: LabsReplayData | undefined): LabsPanelReplayDiagnostics | null {
  if (!replay || replay.status === 'passed') {
    return null;
  }

  const failureReason = replay.failure_reason ?? null;
  const errorType = boundedDiagnosticText(replay.error_type);
  const errorMessage = boundedDiagnosticText(replay.error_message);
  const durationSeconds = replayDurationSeconds(replay);
  const details: string[] = [];

  if (failureReason) {
    details.push(`reason: ${failureReason}`);
  }
  if (errorType) {
    details.push(`error: ${errorType}`);
  }
  if (errorMessage) {
    details.push(`message: ${errorMessage}`);
  }
  if (durationSeconds !== null) {
    details.push(`duration: ${durationSeconds.toFixed(2)}s`);
  }

  return details.length > 0
    ? {
      failureReason,
      errorType,
      errorMessage,
      durationSeconds,
      details,
    }
    : null;
}

function compareNullableNumbers(left: number | null, right: number | null): number {
  if (left === null && right === null) return 0;
  if (left === null) return -1;
  if (right === null) return 1;
  return left - right;
}

function compareRows(left: LabsPanelRow, right: LabsPanelRow, sortBy: LabsSortKey): number {
  switch (sortBy) {
    case 'sharpe':
      return compareNullableNumbers(left.sharpe, right.sharpe);
    case 'max_drawdown':
      return compareNullableNumbers(left.maxDrawdownPct, right.maxDrawdownPct);
    case 'wfe':
      return compareNullableNumbers(left.wfe, right.wfe);
    case 'dsr':
      return compareNullableNumbers(left.dsr, right.dsr);
    case 'status':
      return left.status.localeCompare(right.status);
    case 'provenance_status':
      return left.provenanceStatus.localeCompare(right.provenanceStatus);
    case 'replay_status':
      return left.replayStatus.localeCompare(right.replayStatus);
    case 'validation_status':
      return left.validationStatus.localeCompare(right.validationStatus);
  }
}

function matchesFilter(value: string, filter: OptionalFilter): boolean {
  return filter === 'all' || value === filter;
}

function matchesSearch(row: LabsPanelRow, searchText: string): boolean {
  const query = searchText.trim().toLowerCase();
  if (!query) {
    return true;
  }
  return (
    row.experimentId.toLowerCase().includes(query) ||
    row.artifactPath.toLowerCase().includes(query)
  );
}

function matchesMinimumMetric(value: number | null, threshold: number | null): boolean {
  return threshold === null || (value !== null && value >= threshold);
}

function matchesMaximumDrawdown(value: number | null, threshold: number | null): boolean {
  return threshold === null || (value !== null && value >= threshold);
}

function firstByExperimentId<T extends { experiment_id: string }>(rows: T[]): Map<string, T> {
  const byId = new Map<string, T>();
  for (const row of rows) {
    if (!byId.has(row.experiment_id)) {
      byId.set(row.experiment_id, row);
    }
  }
  return byId;
}

function diffSideLabel(side: LabsExperimentDiffData['left']): string {
  return side.experiment_id ?? side.label ?? side.artifact_path ?? 'unknown';
}

function buildDiffView(diff: LabsExperimentDiffData): LabsPanelDiff {
  return {
    title: `${diffSideLabel(diff.left)} -> ${diffSideLabel(diff.right)}`,
    metricDeltas: Object.entries(diff.metric_deltas).map(([metric, delta]) => ({
      metric,
      left: delta.left,
      right: delta.right,
      delta: delta.delta,
    })),
    missingMetrics: diff.missing_metrics.map((metric) => ({
      metric: metric.metric,
      missingFrom: metric.missing_from,
    })),
    configChanges: Object.entries(diff.config_diffs).map(([key, change]) => ({
      key,
      left: change.left,
      right: change.right,
    })),
    provenanceChange: {
      left: diff.provenance.left,
      right: diff.provenance.right,
      changed: diff.provenance.changed,
    },
  };
}

function isSummaryLimitedEndpoint(status: LabsEndpointStatus): boolean {
  return status.summary_limited === true || status.render_strategy === 'summarize';
}

function endpointRowCount(status: LabsEndpointStatus): number {
  return typeof status.row_count === 'number' && Number.isFinite(status.row_count)
    ? status.row_count
    : 0;
}

function buildEndpointBadge(status: LabsEndpointStatus): LabsPanelEndpointBadge {
  const details: string[] = [];
  if (typeof status.row_count === 'number') {
    details.push(`${status.row_count.toLocaleString()} rows`);
  }
  if (typeof status.size_bytes === 'number') {
    details.push(`${status.size_bytes.toLocaleString()} bytes`);
  }
  if (typeof status.selected_page === 'number') {
    details.push(`page ${status.selected_page}`);
  }
  details.push(...status.validation_errors);

  if (status.status === 'missing' || status.render_strategy === 'missing') {
    return { endpoint: status.endpoint, label: 'missing', tone: 'warning', details };
  }
  if (status.validation_status === 'invalid') {
    return { endpoint: status.endpoint, label: 'invalid', tone: 'danger', details };
  }
  if (isSummaryLimitedEndpoint(status)) {
    return { endpoint: status.endpoint, label: 'summarized', tone: 'warning', details };
  }
  if (status.render_strategy === 'paginate') {
    return { endpoint: status.endpoint, label: 'paginated', tone: 'info', details };
  }
  return { endpoint: status.endpoint, label: 'direct', tone: 'success', details };
}

function buildPaginationControl(status: LabsEndpointStatus): LabsPanelPaginationControl | null {
  if (status.render_strategy !== 'paginate' || !status.pagination || status.pagination.pages.length === 0) {
    return null;
  }

  const firstPage = status.pagination.pages[0];
  return {
    endpoint: status.endpoint,
    selectedPage: status.selected_page ?? firstPage.page,
    pages: status.pagination.pages.map((page) => ({
      page: page.page,
      path: page.path,
      rowCount: page.row_count ?? null,
    })),
  };
}

function buildValidationTruncation(
  data: LabsDashboardData | null,
): LabsPanelValidationTruncation | null {
  const truncation = data?.validation?.truncation;
  if (!truncation) {
    return null;
  }
  return {
    totalResultCount: truncation.total_result_count,
    returnedResultCount: truncation.returned_result_count,
    omittedResultCount: truncation.omitted_result_count,
    omittedErrorCount: truncation.omitted_error_count,
    maxErrorsPerResult: truncation.max_errors_per_result,
  };
}

export function buildLabsPanelViewModel(
  data: LabsDashboardData | null,
  filters: LabsPanelFilters = {},
): LabsPanelViewModel {
  const effectiveFilters = { ...DEFAULT_FILTERS, ...filters };
  const registryRows = data?.registry?.experiments ?? [];
  const scorecardsById = firstByExperimentId(data?.scorecards ?? []);
  const replaysById = firstByExperimentId(data?.replays ?? []);
  const validationResults = data?.validation?.results ?? [];
  const diffs = (data?.diffs ?? []).map(buildDiffView);

  const rows = registryRows.map((row): LabsPanelRow => {
    const scorecard = scorecardsById.get(row.experiment_id);
    const replay = replaysById.get(row.experiment_id);
    const validation = validationForRow(row, validationResults);
    const validationStatus: ValidationStatus = validation
      ? (validation.valid ? 'valid' : 'invalid')
      : 'missing';
    const validationErrors = validation?.errors ?? [];
    const omittedValidationErrorCount = validation?.omitted_error_count ?? 0;

    return {
      experimentId: row.experiment_id,
      artifactPath: row.artifact_path,
      status: row.status,
      provenanceStatus: row.provenance_status,
      governanceState: row.governance_state ?? scorecard?.governance_state ?? 'clear',
      governanceReasons: row.governance_reasons ?? scorecard?.governance_reasons ?? [],
      scorecardStatus: scorecard?.status ?? 'missing',
      replayStatus: replay?.status ?? 'missing',
      replayDiagnostics: buildReplayDiagnostics(replay),
      validationStatus,
      sharpe: metricValue(row, 'sharpe'),
      maxDrawdownPct: metricValue(row, 'max_drawdown_pct'),
      cagrPct: metricValue(row, 'cagr_pct'),
      wfe: metricValue(row, 'wfe'),
      dsr: metricValue(row, 'dsr'),
      positiveOosRatio: metricValue(row, 'positive_oos_ratio'),
      regimeCoverage: metricValue(row, 'regime_coverage'),
      validationErrors,
      validationErrorCount: validationErrors.length + omittedValidationErrorCount,
      omittedValidationErrorCount,
    };
  });

  const filteredRows = rows
    .filter((row) => matchesFilter(row.status, effectiveFilters.status))
    .filter((row) => matchesFilter(row.provenanceStatus, effectiveFilters.provenanceStatus))
    .filter((row) => matchesFilter(row.replayStatus, effectiveFilters.replayStatus))
    .filter((row) => matchesFilter(row.validationStatus, effectiveFilters.validationStatus))
    .filter((row) => matchesSearch(row, effectiveFilters.searchText))
    .filter((row) => matchesMinimumMetric(row.sharpe, effectiveFilters.minSharpe))
    .filter((row) => matchesMaximumDrawdown(row.maxDrawdownPct, effectiveFilters.maxDrawdownPct))
    .sort((left, right) => {
      const comparison = compareRows(left, right, effectiveFilters.sortBy);
      return effectiveFilters.sortDirection === 'asc' ? comparison : -comparison;
    });

  const missingEndpoints = data?.missing ?? [];
  const errors = data?.errors ?? [];
  const hasPublishedData = registryRows.length > 0 || (data?.scorecards.length ?? 0) > 0 || (data?.replays.length ?? 0) > 0;
  const endpointStatus = data?.endpoint_status ?? [];
  const endpointBadges = endpointStatus.map(buildEndpointBadge);
  const paginationControls = endpointStatus
    .map(buildPaginationControl)
    .filter((control): control is LabsPanelPaginationControl => control !== null);
  const summaryLimitedEndpoints = endpointStatus.filter(isSummaryLimitedEndpoint);
  const summaryLimitedRowCount = summaryLimitedEndpoints.reduce(
    (total, status) => total + endpointRowCount(status),
    0,
  );
  const missingProvenanceCandidates = rows.filter((row) => (
    row.provenanceStatus === 'missing' &&
    (
      row.status === 'candidate' ||
      row.status === 'validated' ||
      row.scorecardStatus === 'watch' ||
      row.scorecardStatus === 'promote'
    )
  )).length;
  const hasSummaryLimitedData = summaryLimitedEndpoints.length > 0;
  const disabled = !data || (!data.available && !hasPublishedData && !hasSummaryLimitedData);
  const emptyMessage = disabled
    ? 'Labs artifacts are not published yet'
    : hasSummaryLimitedData && registryRows.length === 0
      ? 'Labs summary metadata is available; full artifacts were not downloaded'
      : 'No Labs experiments match the selected filters';

  return {
    disabled,
    emptyMessage,
    missingEndpoints,
    errors,
    endpointStatus,
    endpointBadges,
    summaryLimitedEndpoints,
    paginationControls,
    validationTruncation: buildValidationTruncation(data),
    rows: filteredRows,
    diffs,
    sortOptions: LABS_SORT_OPTIONS,
    filterOptions: {
      status: uniqueSorted(rows.map((row) => row.status)),
      provenanceStatus: uniqueSorted(rows.map((row) => row.provenanceStatus)),
      replayStatus: uniqueSorted(rows.map((row) => row.replayStatus)),
      validationStatus: ['all', 'valid', 'invalid', 'missing'],
    },
    summary: {
      available: Boolean(data?.available),
      experiments: registryRows.length,
      scorecards: data?.scorecards.length ?? 0,
      replays: data?.replays.length ?? 0,
      invalidArtifacts: rows.filter((row) => row.validationStatus === 'invalid').length,
      missingProvenanceCandidates,
      summaryLimitedEndpoints: summaryLimitedEndpoints.length,
      indexedRows: summaryLimitedRowCount,
    },
  };
}

function formatMetric(value: number | null, suffix = ''): string {
  return value === null ? 'n/a' : `${value.toFixed(2)}${suffix}`;
}

function formatRatioPercent(value: number | null): string {
  return value === null ? 'n/a' : `${(value * 100).toFixed(1)}%`;
}

function formatDiffValue(value: unknown): string {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return value.toFixed(2);
  }
  if (typeof value === 'string' || typeof value === 'boolean') {
    return String(value);
  }
  if (value === null || value === undefined) {
    return 'n/a';
  }
  return JSON.stringify(value);
}

function filterValue(value: OptionalFilter | ValidationStatus | undefined): string {
  return value ?? 'all';
}

function numberInputValue(value: number | null | undefined): string {
  return value === null || value === undefined ? '' : String(value);
}

function parseNumberInput(value: string): number | null {
  if (!value.trim()) {
    return null;
  }
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function formatEndpointStatus(status: LabsEndpointStatus): string {
  const details: string[] = [];
  if (isSummaryLimitedEndpoint(status)) {
    details.push('summary limited');
  }
  if (typeof status.row_count === 'number') {
    details.push(`${status.row_count.toLocaleString()} rows`);
  }
  if (status.validation_status !== 'not_applicable') {
    details.push(`validation ${status.validation_status}`);
  }
  if (typeof status.size_bytes === 'number') {
    details.push(`${status.size_bytes.toLocaleString()} bytes`);
  }

  return details.length > 0
    ? `${status.endpoint}: ${status.render_strategy} (${details.join(', ')})`
    : `${status.endpoint}: ${status.render_strategy}`;
}

function isAbortError(error: unknown): boolean {
  return (
    typeof error === 'object' &&
    error !== null &&
    'name' in error &&
    (error as { name?: unknown }).name === 'AbortError'
  );
}

export function LabsPanel() {
  const [data, setData] = useState<LabsDashboardData | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [filters, setFilters] = useState<LabsPanelFilters>(DEFAULT_FILTERS);
  const [selectedPages, setSelectedPages] = useState<Partial<Record<LabsEndpointKey, number>>>({});

  useEffect(() => {
    let active = true;
    const controller = new AbortController();
    setLoading(true);

    fetchLabsDashboardDataFromIndex(fetch, { selectedPages, signal: controller.signal })
      .then((nextData) => {
        if (!active) return;
        setData(nextData);
        setLoadError(null);
      })
      .catch((error) => {
        if (!active) return;
        if (isAbortError(error)) return;
        setLoadError(`Labs data unavailable: ${String(error)}`);
      })
      .finally(() => {
        if (active) setLoading(false);
      });

    return () => {
      active = false;
      controller.abort();
    };
  }, [selectedPages]);

  const view = useMemo(() => buildLabsPanelViewModel(data, filters), [data, filters]);

  if (loading) {
    return (
      <div className="labs-panel">
        <h3>Labs</h3>
        <div className="analytics-empty" role="status">Loading Labs artifacts...</div>
      </div>
    );
  }

  if (loadError) {
    return (
      <div className="labs-panel">
        <h3>Labs</h3>
        <div className="analytics-empty">
          <p>{loadError}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="labs-panel">
      <div className="panel-header">
        <h3>Labs</h3>
        <span className={`status-pill ${view.summary.available ? 'status-healthy' : 'status-warning'}`}>
          {view.summary.available ? 'available' : 'disabled'}
        </span>
      </div>

      <div className="analytics-summary">
        <div className="analytics-card">
          <label>Experiments</label>
          <span>{view.summary.experiments}</span>
        </div>
        <div className="analytics-card">
          <label>Scorecards</label>
          <span>{view.summary.scorecards}</span>
        </div>
        <div className="analytics-card">
          <label>Replays</label>
          <span>{view.summary.replays}</span>
        </div>
        <div className="analytics-card">
          <label>Invalid</label>
          <span className={view.summary.invalidArtifacts > 0 ? 'negative' : 'positive'}>
            {view.summary.invalidArtifacts}
          </span>
        </div>
        <div className="analytics-card">
          <label>Missing Provenance</label>
          <span className={view.summary.missingProvenanceCandidates > 0 ? 'negative' : 'positive'}>
            {view.summary.missingProvenanceCandidates}
          </span>
        </div>
      </div>

      {(view.disabled || view.missingEndpoints.length > 0 || view.errors.length > 0) && (
        <div className="analytics-empty">
          <p>{view.disabled ? view.emptyMessage : 'Some Labs endpoints are not available'}</p>
          {view.missingEndpoints.length > 0 && (
            <small>Missing: {view.missingEndpoints.join(', ')}</small>
          )}
          {view.errors.map((error) => (
            <small key={error} className="negative">{error}</small>
          ))}
        </div>
      )}

      {view.endpointBadges.length > 0 && (
        <div className="endpoint-badge-row">
          {view.endpointBadges.map((badge) => (
            <span key={badge.endpoint} className={`status-pill status-${badge.tone}`}>
              {badge.endpoint}: {badge.label}
              {badge.details.length > 0 && <small>{badge.details.join(', ')}</small>}
            </span>
          ))}
        </div>
      )}

      {view.endpointStatus.some((status) => status.render_strategy !== 'direct') && (
        <div className="analytics-empty">
          {view.endpointStatus
            .filter((status) => status.render_strategy !== 'direct')
            .map((status) => (
              <small key={status.endpoint}>
                {formatEndpointStatus(status)}
              </small>
            ))}
        </div>
      )}

      {view.validationTruncation && (
        <div className="analytics-empty">
          <small>
            Validation report: {view.validationTruncation.returnedResultCount} of{' '}
            {view.validationTruncation.totalResultCount} results retained;
            {' '}{view.validationTruncation.omittedResultCount} rows and{' '}
            {view.validationTruncation.omittedErrorCount} errors omitted.
          </small>
        </div>
      )}

      {view.diffs.length > 0 && (
        <div className="stats-section">
          <h4>Experiment Diffs</h4>
          {view.diffs.map((diff) => (
            <div className="analytics-empty" key={diff.title}>
              <strong>{diff.title}</strong>
              {diff.metricDeltas.map((metric) => (
                <small key={metric.metric}>
                  {metric.metric}: {formatDiffValue(metric.left)}{' -> '}{formatDiffValue(metric.right)}
                  {' '}({formatDiffValue(metric.delta)})
                </small>
              ))}
              {diff.missingMetrics.map((metric) => (
                <small key={metric.metric}>
                  {metric.metric}: missing from {metric.missingFrom.join(', ')}
                </small>
              ))}
              {diff.configChanges.map((change) => (
                <small key={change.key}>
                  {change.key}: {formatDiffValue(change.left)}{' -> '}{formatDiffValue(change.right)}
                </small>
              ))}
              {diff.provenanceChange.changed && (
                <small>
                  provenance: {diff.provenanceChange.left}{' -> '}{diff.provenanceChange.right}
                </small>
              )}
            </div>
          ))}
        </div>
      )}

      {!view.disabled && (
        <>
          <div className="controls-row">
            <label>
              Search
              <input
                type="search"
                value={filters.searchText ?? ''}
                onChange={(event) => setFilters((current) => ({
                  ...current,
                  searchText: event.target.value,
                }))}
              />
            </label>
            <label>
              Status
              <select
                value={filterValue(filters.status)}
                onChange={(event) => setFilters((current) => ({ ...current, status: event.target.value }))}
              >
                {view.filterOptions.status.map((option) => (
                  <option key={option} value={option}>{option}</option>
                ))}
              </select>
            </label>
            <label>
              Provenance
              <select
                value={filterValue(filters.provenanceStatus)}
                onChange={(event) => setFilters((current) => ({ ...current, provenanceStatus: event.target.value }))}
              >
                {view.filterOptions.provenanceStatus.map((option) => (
                  <option key={option} value={option}>{option}</option>
                ))}
              </select>
            </label>
            <label>
              Replay
              <select
                value={filterValue(filters.replayStatus)}
                onChange={(event) => setFilters((current) => ({ ...current, replayStatus: event.target.value }))}
              >
                {view.filterOptions.replayStatus.map((option) => (
                  <option key={option} value={option}>{option}</option>
                ))}
              </select>
            </label>
            <label>
              Validation
              <select
                value={filterValue(filters.validationStatus)}
                onChange={(event) => setFilters((current) => ({
                  ...current,
                  validationStatus: event.target.value as LabsPanelFilters['validationStatus'],
                }))}
              >
                {view.filterOptions.validationStatus.map((option) => (
                  <option key={option} value={option}>{option}</option>
                ))}
              </select>
            </label>
            <label>
              Sort
              <select
                value={filters.sortBy ?? DEFAULT_FILTERS.sortBy}
                onChange={(event) => setFilters((current) => ({
                  ...current,
                  sortBy: event.target.value as LabsSortKey,
                }))}
              >
                {view.sortOptions.map((option) => (
                  <option key={option.value} value={option.value}>{option.label}</option>
                ))}
              </select>
            </label>
            <label>
              Direction
              <select
                value={filters.sortDirection ?? DEFAULT_FILTERS.sortDirection}
                onChange={(event) => setFilters((current) => ({
                  ...current,
                  sortDirection: event.target.value as SortDirection,
                }))}
              >
                <option value="desc">desc</option>
                <option value="asc">asc</option>
              </select>
            </label>
            <label>
              Min Sharpe
              <input
                type="number"
                step="0.01"
                value={numberInputValue(filters.minSharpe)}
                onChange={(event) => setFilters((current) => ({
                  ...current,
                  minSharpe: parseNumberInput(event.target.value),
                }))}
              />
            </label>
            <label>
              Max DD
              <input
                type="number"
                step="0.1"
                value={numberInputValue(filters.maxDrawdownPct)}
                onChange={(event) => setFilters((current) => ({
                  ...current,
                  maxDrawdownPct: parseNumberInput(event.target.value),
                }))}
              />
            </label>
            {view.paginationControls.map((control) => (
              <label key={control.endpoint}>
                {control.endpoint} Page
                <select
                  value={selectedPages[control.endpoint] ?? control.selectedPage}
                  onChange={(event) => setSelectedPages((current) => ({
                    ...current,
                    [control.endpoint]: Number(event.target.value),
                  }))}
                >
                  {control.pages.map((page) => (
                    <option key={page.page} value={page.page}>
                      {page.page}{page.rowCount === null ? '' : ` (${page.rowCount})`}
                    </option>
                  ))}
                </select>
              </label>
            ))}
          </div>

          {view.rows.length === 0 ? (
            <div className="analytics-empty">
              <p>{view.emptyMessage}</p>
            </div>
          ) : (
            <div className="stats-section">
              <div className="labs-table-scroll">
                <table className="positions-table">
                  <thead>
                    <tr>
                      <th>Experiment</th>
                      <th>Status</th>
                      <th>Sharpe</th>
                      <th>Max DD</th>
                      <th>WFE</th>
                      <th>DSR</th>
                      <th>Scorecard</th>
                      <th>Replay</th>
                      <th>Validation</th>
                    </tr>
                  </thead>
                  <tbody>
                    {view.rows.map((row) => (
                      <tr key={row.experimentId}>
                        <td>
                          <strong>{row.experimentId}</strong>
                          <small>{row.artifactPath}</small>
                        </td>
                        <td>
                          {row.status}
                          {row.governanceState !== 'clear' && (
                            <small>
                              {row.governanceState}
                              {row.governanceReasons.length > 0
                                ? `: ${row.governanceReasons.join(', ')}`
                                : ''}
                            </small>
                          )}
                        </td>
                        <td>{formatMetric(row.sharpe)}</td>
                        <td className={(row.maxDrawdownPct ?? 0) < -20 ? 'negative' : ''}>
                          {formatMetric(row.maxDrawdownPct, '%')}
                        </td>
                        <td>
                          {formatMetric(row.wfe)}
                          {row.positiveOosRatio !== null && (
                            <small>OOS {formatRatioPercent(row.positiveOosRatio)}</small>
                          )}
                        </td>
                        <td>
                          {formatMetric(row.dsr)}
                          {row.regimeCoverage !== null && (
                            <small>Regime {formatRatioPercent(row.regimeCoverage)}</small>
                          )}
                        </td>
                        <td>{row.scorecardStatus}</td>
                        <td>
                          {row.replayStatus}
                          {row.replayDiagnostics && (
                            <small>{row.replayDiagnostics.details.join('; ')}</small>
                          )}
                        </td>
                        <td className={row.validationStatus === 'invalid' ? 'negative' : ''}>
                          {row.validationStatus}
                          {row.validationErrorCount > 0 && (
                            <small>
                              {row.validationErrors.join('; ')}
                              {row.omittedValidationErrorCount > 0
                                ? ` (${row.omittedValidationErrorCount} more omitted)`
                                : ''}
                            </small>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}
