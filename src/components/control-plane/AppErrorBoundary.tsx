import { Component, type ErrorInfo, type ReactNode } from 'react';

export type AppErrorBoundaryScope = 'shell' | 'workspace';

export interface AppErrorBoundaryProps {
  children: ReactNode;
  scope?: AppErrorBoundaryScope;
  name?: string;
  fallback?: ReactNode;
  /** Clears a captured workspace error when the parent changes workspace. */
  resetKey?: string | number;
  diagnosticReference?: string;
  onError?: (error: Error, info: ErrorInfo) => void;
}

interface AppErrorBoundaryState {
  error: Error | null;
}

/**
 * Crash containment for application-shell and workspace render failures.
 * The fallback deliberately reports authority as unavailable rather than
 * inferring a healthy or safe-to-trade state.
 */
export class AppErrorBoundary extends Component<
  AppErrorBoundaryProps,
  AppErrorBoundaryState
> {
  state: AppErrorBoundaryState = { error: null };

  static getDerivedStateFromError(error: Error): AppErrorBoundaryState {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    console.error(
      `[AppErrorBoundary:${this.props.scope ?? 'shell'}${this.props.name ? `:${this.props.name}` : ''}]`,
      error,
      info.componentStack,
    );
    this.props.onError?.(error, info);
  }

  componentDidUpdate(previousProps: AppErrorBoundaryProps): void {
    if (
      this.state.error
      && previousProps.resetKey !== this.props.resetKey
    ) {
      this.setState({ error: null });
    }
  }

  private retryWorkspace = (): void => {
    this.setState({ error: null });
  };

  private reloadApplication = (): void => {
    window.location.reload();
  };

  render(): ReactNode {
    if (!this.state.error) return this.props.children;
    if (this.props.fallback) return this.props.fallback;

    const scope = this.props.scope ?? 'shell';
    const isWorkspace = scope === 'workspace';
    const diagnosticReference = this.props.diagnosticReference
      ?? `${scope}:${this.state.error.name || 'render-error'}`;

    return (
      <section
        className={`app-error-boundary app-error-boundary--${scope}`}
        role="alert"
        aria-labelledby={`${scope}-error-title`}
      >
        <p className="app-error-boundary__brand">Portfolio Lab</p>
        <h1 id={`${scope}-error-title`}>
          {isWorkspace ? 'This workspace could not be displayed' : 'The application could not be displayed'}
        </h1>
        <p>
          Live authority and system health cannot be confirmed from this view.
          Do not place new orders until validated data is available.
        </p>
        <p>
          {isWorkspace
            ? 'Retry this workspace. Reload the application if the failure continues.'
            : 'Reload the application. If the failure continues, inspect the latest frontend logs.'}
        </p>
        <p className="app-error-boundary__diagnostic">
          Diagnostic reference: <code>{diagnosticReference}</code>
        </p>
        <button
          type="button"
          onClick={isWorkspace ? this.retryWorkspace : this.reloadApplication}
        >
          {isWorkspace ? 'Retry workspace' : 'Reload application'}
        </button>
      </section>
    );
  }
}

export default AppErrorBoundary;
