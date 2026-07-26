"""Batch BA / M11: dual-write provenance badge contracts (source-level)."""

from pathlib import Path


def test_health_operations_exports_dual_write_summary():
    src = Path("src/components/healthOperations.ts").read_text(encoding="utf-8")
    assert "summarizeDualWriteProvenance" in src
    assert "dualWrite" in src
    assert "Dual-write: FAIL" in src
    assert "dual_write_lag_stale" in src


def test_health_panel_renders_dual_write_badge():
    src = Path("src/components/HealthPanel.tsx").read_text(encoding="utf-8")
    assert "dualWrite" in src
    assert "dual-write-badge" in src
    assert "Dual-Write" in src


def test_live_dashboard_fetches_health_ops_for_provenance():
    src = Path("src/components/LiveDashboard.tsx").read_text(encoding="utf-8")
    assert "health_ops.json" in src
    assert "healthOpsProvenance" in src or "setHealthOpsProvenance" in src
    assert "dualWriteProvenance" in src


def test_live_types_include_provenance_completeness():
    src = Path("src/types/live.ts").read_text(encoding="utf-8")
    assert "ProvenanceCompleteness" in src
    assert "provenance_completeness" in src
    assert "dual_write_ok" in src
