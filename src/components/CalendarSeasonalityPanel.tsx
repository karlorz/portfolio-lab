import React from 'react';

interface CalendarData {
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

export function CalendarSeasonalityPanel({ data }: CalendarSeasonalityPanelProps) {
  if (!data || !data.active) {
    return (
      <div className="panel">
        <h3>Calendar Seasonality (v3.50)</h3>
        <p className="muted">Market closed or no data</p>
      </div>
    );
  }

  const effectColor = EFFECT_COLORS[data.effect] || '#6b7280';
  const modifierPct = (data.modifier * 100).toFixed(0);

  return (
    <div className="panel">
      <h3>Calendar Seasonality (v3.50)</h3>
      <div className="panel-grid">
        <div className="metric">
          <span className="label">Urgency Modifier</span>
          <span className="value" style={{ color: effectColor }}>
            {data.modifier.toFixed(2)}x ({modifierPct}%)
          </span>
        </div>
        <div className="metric">
          <span className="label">Recommendation</span>
          <span className="value" style={{ color: effectColor }}>
            {data.recommendation.toUpperCase()}
          </span>
        </div>
        <div className="metric">
          <span className="label">Effect</span>
          <span className="value" style={{ color: effectColor }}>
            {data.effect.toUpperCase()}
          </span>
        </div>
        <div className="metric">
          <span className="label">Next Window</span>
          <span className="value">
            {WINDOW_LABELS[data.next_window] || data.next_window} ({data.days_to_next}d)
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
