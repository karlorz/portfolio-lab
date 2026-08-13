import { useId } from 'react';
import { AuthorityBadge } from './AuthorityBadge';
import { StatusBadge } from './StatusBadge';

interface AllocationSpineProps {
  allocations?: Record<string, number> | null;
  regime?: string | null;
  updatedAt?: string | null;
  killEnabled?: boolean;
  killLevel?: string | null;
  routed?: boolean;
  marlAuthoritative?: boolean;
}

const percentFormatter = new Intl.NumberFormat('en-US', {
  style: 'percent',
  maximumFractionDigits: 1,
});

function normalizedAllocations(allocations?: Record<string, number> | null) {
  const rows = Object.entries(allocations ?? {})
    .filter(([, value]) => Number.isFinite(value) && value >= 0);
  const total = rows.reduce((sum, [, value]) => sum + value, 0);
  if (total <= 0) return [];
  return rows.map(([symbol, value]) => ({
    symbol,
    value,
    percentage: (value / total) * 100,
  }));
}

export function AllocationSpine({
  allocations,
  regime,
  updatedAt,
  killEnabled = false,
  killLevel,
  routed,
  marlAuthoritative = false,
}: AllocationSpineProps) {
  const headingId = useId();
  const segments = normalizedAllocations(allocations);

  return (
    <section className="allocation-spine" aria-labelledby={headingId}>
      <div className="allocation-spine-heading">
        <div>
          <p className="control-eyebrow">Live authority</p>
          <h2 id={headingId}>Allocation Spine</h2>
        </div>
        <div className="allocation-spine-status">
          <AuthorityBadge routed={routed} blocked={killEnabled} />
          <StatusBadge
            label={killEnabled ? `Kill ${killLevel || 'enabled'}` : 'Order flow permitted'}
            tone={killEnabled ? 'warning' : 'success'}
          />
        </div>
      </div>

      {segments.length > 0 ? (
        <>
          <div className="allocation-spine-track" aria-label="Authoritative target allocation">
            {segments.map((segment, index) => (
              <div
                key={segment.symbol}
                className={`allocation-spine-segment allocation-spine-segment-${(index % 5) + 1}`}
                style={{ flexBasis: `${segment.percentage}%` }}
              >
                <strong translate="no">{segment.symbol}</strong>
                <span>{percentFormatter.format(segment.value)}</span>
              </div>
            ))}
          </div>
          <div className="allocation-spine-meta">
            <span>Regime: <strong>{regime || 'Unknown'}</strong></span>
            <span>Updated: <strong>{updatedAt || 'Awaiting data'}</strong></span>
            <span>MARL: <strong>{marlAuthoritative ? 'Authoritative' : 'Research shadow · non-routed'}</strong></span>
          </div>
        </>
      ) : (
        <p className="control-empty" role="status">Authoritative allocation is unavailable. Verify signals.json before acting.</p>
      )}
    </section>
  );
}
