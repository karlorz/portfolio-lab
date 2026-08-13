import { Component, type ReactNode, type ErrorInfo } from 'react';

interface ErrorBoundaryProps {
  children: ReactNode;
  fallback?: ReactNode;
  name?: string;
}

interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
}

/**
 * Lightweight error boundary that catches rendering errors in child components.
 * Each dashboard panel is wrapped with this so a single panel crash doesn't
 * take down the entire dashboard.
 */
export class PanelErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error(
      `[PanelErrorBoundary${this.props.name ? `:${this.props.name}` : ''}]`,
      error,
      info.componentStack,
    );
  }

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback;
      }
      return (
        <div className="panel-error-boundary" style={{
          padding: '16px',
          border: '1px dashed #ef4444',
          borderRadius: '8px',
          background: '#fef2f2',
          color: '#991b1b',
          fontSize: '14px',
        }}>
          <strong>Panel Error</strong>
          {this.props.name && <span> — {this.props.name}</span>}
          <p style={{ margin: '8px 0 0', fontSize: '12px', opacity: 0.8 }}>
            {this.state.error?.message || 'Unknown rendering error'}
          </p>
          <button
            onClick={() => this.setState({ hasError: false, error: null })}
            style={{
              marginTop: '8px',
              padding: '4px 12px',
              fontSize: '12px',
              cursor: 'pointer',
              border: '1px solid #991b1b',
              borderRadius: '4px',
              background: 'transparent',
              color: '#991b1b',
            }}
          >
            Retry
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}
