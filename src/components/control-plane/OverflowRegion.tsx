import React from 'react';

interface OverflowRegionProps {
  label: string;
  children: React.ReactNode;
  className?: string;
}

export function OverflowRegion({ label, children, className }: OverflowRegionProps) {
  return (
    <div
      className={className ? `overflow-region ${className}` : 'overflow-region'}
      role="region"
      aria-label={label}
      tabIndex={0}
    >
      {children}
    </div>
  );
}
