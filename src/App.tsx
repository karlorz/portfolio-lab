import { useEffect, lazy, Suspense } from 'react';
import { LiveDashboard } from './components/LiveDashboard';
import { AppShell } from './components/control-plane/AppShell';
import { AppErrorBoundary } from './components/control-plane/AppErrorBoundary';
import { workspaceDestination } from './components/control-plane/navigation';
import { useWorkspaceLocation } from './hooks/useWorkspaceLocation';
import './styles/tokens.css';
import './styles/base.css';
import './styles/control-plane.css';
import './App.css';

const BacktestsWorkspace = lazy(() => import('./components/control-plane/BacktestsWorkspace'));
const DesignGuidePage = lazy(() => import('./components/control-plane/DesignGuidePage').then(m => ({ default: m.DesignGuidePage })));

function App() {
  const { workspace, navigate } = useWorkspaceLocation();

  useEffect(() => {
    document.title = window.location.pathname === '/design-guide'
      ? 'Design Guide · Portfolio Lab'
      : `${workspaceDestination(workspace).label} · Portfolio Lab`;
  }, [workspace]);

  if (window.location.pathname === '/design-guide') {
    return (
      <Suspense fallback={<div className="control-loading" role="status">Loading design guide…</div>}>
        <DesignGuidePage />
      </Suspense>
    );
  }

  return (
    <AppErrorBoundary scope="shell">
      <AppShell workspace={workspace} onNavigate={navigate}>
        <AppErrorBoundary scope="workspace" resetKey={workspace}>
          <Suspense
            fallback={<div className="control-loading" role="status">Loading workspace…</div>}
          >
            {workspace === 'backtests' ? <BacktestsWorkspace /> : <LiveDashboard
              refreshInterval={60}
              activeView={workspace}
              onViewChange={navigate}
            />}
          </Suspense>
        </AppErrorBoundary>
      </AppShell>
    </AppErrorBoundary>
  );
}

export default App;
