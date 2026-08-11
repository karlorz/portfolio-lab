"""Item 8 — health-projections cluster extraction tests.

Covers A1 (import smoke: every moved name resolves via BOTH
``src.dashboard.generator`` and ``src.dashboard.health_projections``) and A2
(behavior-equality of the re-exported projections on a canned fixture).
"""

import pytest

import src.dashboard.generator as generator
import src.dashboard.health_projections as health_projections

MOVED_NAMES = [
    "_parse_rebalance_clock",
    "project_smart_rebalance_budget_onto_health",
    "project_execution_timeline_onto_health",
    "project_repo_public_mirror_lag_onto_health",
    "project_pending_artifact_cron_onto_health",
    "project_reentry_eligibility_onto_health",
    "project_voting_mass_quality_onto_health",
    "project_paper_return_ssot_onto_health",
    "_apply_kill_to_smart_rebalance",
    "_remaining_budget_ratio",
    "_remaining_budget_display_pct",
    "_load_canonical_health_report",
]


@pytest.mark.parametrize("name", MOVED_NAMES)
def test_name_resolves_via_both_modules(name):
    gen_attr = getattr(generator, name)
    hp_attr = getattr(health_projections, name)
    assert gen_attr is hp_attr


def _fixture():
    health: dict = {}
    smart_rebalance = {
        "status": {
            "ytd_cost_bps": 45.2,
            "remaining_budget_pct": 0.6,
            "annual_cost_limit_pct": 0.5,
            "is_over_budget": False,
            "is_warning": True,
            "last_rebalance": "2026-08-10T12:00:00Z",
            "config": {"annual_cost_limit": "0.5%"},
        }
    }
    rebalance_health = {
        "next_rebalance": {
            "last_execution_at": "2026-08-10T14:30:00Z",
            "last_execution_clock": "order_event",
        },
        "execution_history": [{"date": "2026-07-30"}, {"date": "2026-07-11"}],
        "raw_history_entries": 53,
        "snapshot_rewrite_files": 13,
        "canonical_execution_days": 5,
        "rebalance_execution_timeline_status": "ok",
    }
    return health, smart_rebalance, rebalance_health


def test_budget_projection_behavior_stable():
    """A2: canned fixture → same output shape/values as recorded pre-move."""
    health, sr, rb = _fixture()
    out = generator.project_smart_rebalance_budget_onto_health(dict(health), sr, rb)
    assert out["rebalance_ytd_cost_bps"] == 45.2
    assert out["rebalance_remaining_budget_pct"] == 0.6
    assert out["rebalance_annual_cost_limit_pct"] == 0.5
    assert out["rebalance_is_over_budget"] is False
    assert out["rebalance_is_warning"] is True
    assert out["rebalance_budget_status"] == "warning"
    assert out["rebalance_controller_last_rebalance"] == "2026-08-10T12:00:00Z"
    assert out["rebalance_last_execution_at"] == "2026-08-10T14:30:00Z"
    assert out["rebalance_last_execution_clock"] == "order_event"
    assert out["rebalance_controller_clock_lag_days"] == 0.1  # 2.5h ≈ 0.10d
    assert out["rebalance_controller_clock_lagging"] is False


def test_budget_projection_clear_path():
    out = generator.project_smart_rebalance_budget_onto_health(None, None)
    assert out == {"rebalance_budget_status": "unknown"}


def test_execution_timeline_projection_behavior_stable():
    """A2: canned fixture → same output values as recorded pre-move."""
    health, _sr, rb = _fixture()
    out = generator.project_execution_timeline_onto_health(dict(health), rb)
    assert out["rebalance_execution_timeline_status"] == "ok"
    assert out["rebalance_execution_timeline_badge"] == "unique=5 raw=53"
    assert out["rebalance_raw_history_entries"] == 53
    assert out["rebalance_snapshot_rewrite_files"] == 13
    assert out["rebalance_unique_execution_days"] == 5


def test_execution_timeline_unknown_path():
    out = generator.project_execution_timeline_onto_health(None, None)
    assert out == {"rebalance_execution_timeline_status": "unknown"}
