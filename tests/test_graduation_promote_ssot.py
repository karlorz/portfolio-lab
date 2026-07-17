"""GraduationChecklist is sole writer of .promote_to_live candidacy."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.strategy.graduation_checklist import CheckResult, GraduationChecklist


def _results(*, ready: bool) -> dict[str, CheckResult]:
    """Build minimal checklist results that pass or fail is_graduation_ready."""
    names = [
        "min_trading_days",
        "min_sharpe",
        "max_drawdown",
        "min_win_rate",
        "health_checks",
        "min_tca_orders",
        "circuit_breaker_confidence",
        "min_dsr",
        "regime_coverage",
        "signal_diversity",
        "sharpe_ci_lower",
        "manual_approval",
    ]
    out = {}
    for name in names:
        if name == "manual_approval":
            out[name] = CheckResult(name, False, 0.0, 1.0, "manual")
        else:
            out[name] = CheckResult(
                name,
                ready,
                1.0 if ready else 0.0,
                1.0,
                name,
            )
    return out


def test_write_promote_when_checklist_ready(tmp_path: Path) -> None:
    checklist = GraduationChecklist()
    results = _results(ready=True)
    path = checklist.write_promote_to_live_if_ready(results, data_dir=tmp_path)
    assert path is not None
    assert path.exists()
    payload = json.loads(path.read_text())
    assert payload["action"] == "promote_to_live"
    assert payload["source"] == "graduation_checklist"
    assert payload["graduation_conflict"] is False
    assert payload["requires_approval"] is True


def test_no_promote_when_checklist_not_ready(tmp_path: Path) -> None:
    checklist = GraduationChecklist()
    results = _results(ready=False)
    # seed a stale metric-only promote marker
    stale = tmp_path / ".promote_to_live"
    stale.write_text(json.dumps({"action": "promote_to_live", "source": "evaluator"}))
    path = checklist.write_promote_to_live_if_ready(results, data_dir=tmp_path)
    assert path is None
    conflict = tmp_path / ".graduation_conflict.json"
    assert conflict.exists()
    body = json.loads(conflict.read_text())
    assert body["graduation_conflict"] is True
    # Stale candidacy must be tombstoned — no live promote_to_live action
    promote = json.loads(stale.read_text())
    assert promote["action"] == "promote_blocked_checklist"
    assert promote["graduation_conflict"] is True
    assert promote["action"] != "promote_to_live"


def test_no_promote_under_kill_halt(tmp_path: Path) -> None:
    (tmp_path / "kill_switch.json").write_text(
        json.dumps(
            {
                "enabled": True,
                "level": "halt",
                "reason": "unresolved_incident:signal_staleness",
                "source": "incident_lifecycle",
            }
        )
    )
    checklist = GraduationChecklist()
    results = _results(ready=True)
    path = checklist.write_promote_to_live_if_ready(results, data_dir=tmp_path)
    assert path is None
    assert not (tmp_path / ".promote_to_live").exists()


def test_stale_promote_invalidated_under_kill_halt(tmp_path: Path) -> None:
    """Existing candidacy must not survive kill halt as action promote_to_live."""
    (tmp_path / "kill_switch.json").write_text(
        json.dumps(
            {
                "enabled": True,
                "level": "halt",
                "reason": "unresolved_incident:signal_staleness",
                "source": "incident_lifecycle",
                "incident_id": "inc-1",
                "mode": "paper",
            }
        )
    )
    stale = tmp_path / ".promote_to_live"
    stale.write_text(
        json.dumps(
            {
                "action": "promote_to_live",
                "source": "graduation_checklist",
                "requires_approval": True,
                "metrics": {"sharpe": 0.86},
                "timestamp": "2026-07-15T02:31:12",
            }
        )
    )
    checklist = GraduationChecklist()
    path = checklist.write_promote_to_live_if_ready(_results(ready=True), data_dir=tmp_path)
    assert path is None
    promote = json.loads(stale.read_text())
    assert promote["action"] == "promote_blocked_kill"
    assert promote["graduation_conflict"] is True
    assert "halt" in str(promote.get("kill_level") or promote.get("reason") or "").lower() or (
        promote.get("reason") == "kill_authority"
    )
    conflict = tmp_path / ".graduation_conflict.json"
    assert conflict.exists()
    assert json.loads(conflict.read_text())["action"] == "promote_blocked_kill"


def test_evaluator_does_not_write_promote_from_metric_only_gates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Even with good advisory metrics, promote only via checklist SSOT."""
    from src.strategy.evaluator import check_graduation_criteria, Portfolio
    import src.strategy.evaluator as ev

    monkeypatch.setattr(ev, "DATA_DIR", tmp_path)

    p = Portfolio(tmp_path / "portfolio.json", mode="paper")
    p.history = []
    import numpy as np

    rng = np.random.RandomState(12345)
    val = 100000.0
    for i in range(63):
        ret = rng.normal(0.0008, 0.01)
        val *= 1 + ret
        p.history.append(
            {
                "timestamp": f"2026-01-{(i % 28) + 1:02d}T23:00:00",
                "total_value": round(val, 2),
                "daily_return": ret,
            }
        )

    # Checklist not ready on live data → no promote file from evaluator path
    with patch.object(ev, "DATA_DIR", tmp_path):
        check_graduation_criteria(p)

    # Metric-only path must not create promote without checklist readiness
    # (live checklist almost certainly not ready; ensure evaluator didn't invent metrics-only file)
    promote = tmp_path / ".promote_to_live"
    if promote.exists():
        payload = json.loads(promote.read_text())
        assert payload.get("source") == "graduation_checklist"


def test_wiki_sync_dict_fail_closed_without_claiming_sharpe_met(tmp_path: Path) -> None:
    from src.research.wiki_sync import WikiSync

    # Force checklist not ready
    with patch(
        "src.strategy.graduation_checklist.GraduationChecklist.is_graduation_ready",
        return_value=False,
    ), patch(
        "src.strategy.graduation_checklist.GraduationChecklist.check",
        return_value=_results(ready=False),
    ), patch(
        "src.strategy.graduation_checklist.GraduationChecklist.readiness_score",
        return_value=18.2,
    ):
        # WikiSync may need construction args — use minimal mock if needed
        sync = object.__new__(WikiSync)
        out = WikiSync._graduation_status_dict(sync, 0.1, 0.9, 0.05, 100)
    assert out["status"] == "tracking"
    assert out["sharpe_met"] is False
    assert out["graduation_conflict"] is True
    assert out["advisory_sharpe_met"] is True
