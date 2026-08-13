
export interface CalendarData {
  active: boolean;
  modifier: number;
  active_windows: string[];
  next_window: string;
  days_to_next: number;
  recommendation: string;
  effect: string;
}

interface CalendarSeasonalityPanelProps {
  data: CalendarData | null;
}

const EFFECT_COLORS: Record<string, string> = {
  positive: '#10b981',
  neutral: '#6b7280',
  negative: '#f59e0b',
  avoid: '#ef4444',
};

const WINDOW_LABELS: Record<string, string> = {
  tom_window: 'Turn-of-Month',
  pre_holiday: 'Pre-Holiday',
  post_holiday: 'Post-Holiday',
  quarter_end: 'Quarter-End',
  monday: 'Monday',
  pre_fomc: 'Pre-FOMC',
  december: 'December',
  options_expiry: 'OPEX',
};

export function formatCalendarWindow(nextWindow: string, daysToNext: number): string {
  const label = WINDOW_LABELS[nextWindow] || nextWindow;
  return `${label} in ${daysToNext}d`;
}

export function CalendarSeasonalityPanel({ data }: CalendarSeasonalityPanelProps) {
  if (!data || !data.active) {
    return (
      <div className="panel signal-card calendar-seasonality-card">
        <h3>Calendar Seasonality (v3.50)</h3>
        <p className="muted">Market closed or no data</p>
      </div>
    );
  }

  const effectColor = EFFECT_COLORS[data.effect] || '#6b7280';
  const modifierPct = (data.modifier * 100).toFixed(0);

  return (
    <div className="panel signal-card calendar-seasonality-card">
      <div className="signal-card-header">
        <h3>Calendar Seasonality (v3.50)</h3>
        <span className="signal-status-pill signal-status-info">Execution Timing</span>
      </div>

      <div className="signal-card-hero compact">
        <div className="signal-hero-summary">
          <span className="label">Recommendation</span>
          <span className="value hero-value" style={{ color: effectColor }}>
            {data.recommendation.toUpperCase()}
          </span>
          <span className="subtext">{data.effect.toLowerCase()} execution effect</span>
        </div>
      </div>

      <div className="panel-grid signal-kpi-grid">
        <div className="metric">
          <span className="label">Urgency</span>
          <span className="value" style={{ color: effectColor }}>
            {data.modifier.toFixed(2)}x ({modifierPct}%)
          </span>
        </div>
        <div className="metric">
          <span className="label">Effect</span>
          <span className="value" style={{ color: effectColor }}>
            {data.effect.toUpperCase()}
          </span>
        </div>
        <div className="metric">
          <span className="label">Execution Window</span>
          <span className="value">
            {formatCalendarWindow(data.next_window, data.days_to_next)}
          </span>
        </div>
      </div>
      {data.active_windows.length > 0 && (
        <div className="windows-list">
          <span className="label">Active Windows:</span>
          <div className="window-tags">
            {data.active_windows.map(w => (
              <span key={w} className="window-tag">
                {WINDOW_LABELS[w] || w}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
