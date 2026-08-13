
export type StatusTone = 'neutral' | 'info' | 'success' | 'warning' | 'critical' | 'stale';

interface StatusBadgeProps {
  label: string;
  tone?: StatusTone;
  detail?: string;
}
export function StatusBadge({ label, tone = 'neutral', detail }: StatusBadgeProps) {
  return (
    <span className={`control-status control-status-${tone}`} title={detail}>
      <span className="control-status-shape" aria-hidden="true" />
      <span>{label}</span>
    </span>
  );
}
