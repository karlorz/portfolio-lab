export interface RollingMetricPoint {
  date: string;
  sharpe: number;
  volatility: number;
  mean_return: number;
  window_days: number;
}

export interface RollingMetricsRow {
  date: string;
  dateFormatted: string;
  sharpe63: number | null;
  sharpe126: number | null;
  sharpe252: number | null;
  vol63: number | null;
}

export function formatShortDate(dateStr: string): string {
  const d = new Date(dateStr);
  return `${d.getMonth() + 1}/${d.getDate()}`;
}

function byDate(series: RollingMetricPoint[]): Map<string, RollingMetricPoint> {
  return new Map(series.map((point) => [point.date, point]));
}

export function mergeRollingMetrics(
  sharpe63d: RollingMetricPoint[],
  sharpe126d: RollingMetricPoint[],
  sharpe252d: RollingMetricPoint[],
): RollingMetricsRow[] {
  const by63 = byDate(sharpe63d);
  const by126 = byDate(sharpe126d);
  const by252 = byDate(sharpe252d);
  const allDates = new Set<string>([
    ...by63.keys(),
    ...by126.keys(),
    ...by252.keys(),
  ]);

  return Array.from(allDates)
    .sort()
    .map((date) => {
      const row63 = by63.get(date);
      return {
        date,
        dateFormatted: formatShortDate(date),
        sharpe63: row63?.sharpe ?? null,
        sharpe126: by126.get(date)?.sharpe ?? null,
        sharpe252: by252.get(date)?.sharpe ?? null,
        vol63: row63?.volatility ?? null,
      };
    });
}
