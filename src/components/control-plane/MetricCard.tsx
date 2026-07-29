import React from 'react';

interface MetricCardProps {
  label: string;
  value: React.ReactNode;
  detail?: React.ReactNode;
  tone?: 'default' | 'attention';
}
export function MetricCard({ label, value, detail, tone = 'default' }: MetricCardProps) {
  return (
    <section className={`control-metric control-metric-${tone}`}>
      <h3>{label}</h3>
      <div className="control-metric-value">{value}</div>
      {detail && <div className="control-metric-detail">{detail}</div>}
    </section>
  );
}
