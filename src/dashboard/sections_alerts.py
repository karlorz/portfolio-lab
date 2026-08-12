"""Alerts / incidents / promotion mixin extracted from ``src.dashboard.generator``.

Class-level cluster C7 (6 state-free static helpers) moved here by Item 22
(2026-08-12). ``DashboardGenerator`` inherits ``_AlertsSectionsMixin``.
Helpers are param-driven (data_dir/public_dir passed in); the 8
``DashboardGenerator.`` refs (incl. ``_load_json_file`` — C6, stays) use a
call-time lazy import (circular-import safe).
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils import safe_get


class _AlertsSectionsMixin:
    @staticmethod
    def _has_open_blocking_incident(data_dir: Path) -> bool:
        from src.dashboard.generator import DashboardGenerator  # lazy (stays in generator)
        for filename in ("incidents.json", "incident_state.json"):
            payload = DashboardGenerator._load_json_file(data_dir / filename)
            if not payload:
                continue
            raw_incidents = payload.get("incidents", payload.get("open_incidents", []))
            incidents = raw_incidents if isinstance(raw_incidents, list) else []
            for incident in incidents:
                if not isinstance(incident, dict):
                    continue
                status = str(incident.get("status", "open")).lower()
                blocking = bool(incident.get("blocking") or incident.get("blocks_promotion"))
                if blocking and status not in {"closed", "resolved", "pass"}:
                    return True
        return False

    @staticmethod
    def _promotion_gate_status(data_dir: Path) -> tuple[bool, list[str]]:
        from src.dashboard.provenance import SIGNAL_EXCEPTIONS  # lazy (mixin staticmethod rule)
        from src.dashboard.generator import DashboardGenerator  # lazy (stays in generator)
        kill_switch = DashboardGenerator._load_json_file(data_dir / "kill_switch.json")
        blockers: list[str] = []
        if kill_switch and kill_switch.get("enabled"):
            blockers.append("kill_switch")
        if DashboardGenerator._has_open_blocking_incident(data_dir):
            blockers.append("blocking_incident")

        try:
            from src.strategy.graduation_checklist import GraduationChecklist

            checklist = GraduationChecklist()
            results = checklist.check()
            manual = results.get("manual_approval")
            if manual is None or not manual.passed:
                blockers.append("manual_approval")
            if not checklist.is_graduation_ready(results):
                blockers.append("graduation_checklist")
        except SIGNAL_EXCEPTIONS:
            blockers.append("graduation_checklist_unavailable")

        return not blockers, blockers

    @staticmethod
    def _is_active_promote_candidacy(data: Dict[str, Any]) -> bool:
        """True only for live promote candidacy, not tombstones.

        GraduationChecklist rewrites ``.promote_to_live`` with
        ``action: promote_blocked_*`` when kill or checklist blocks.
        Those are not candidates — alerts must ignore them.
        """
        action = data.get("action")
        if action is None:
            # Legacy markers omit action; treat as candidacy.
            return True
        if not isinstance(action, str):
            return False
        if action == "promote_to_live":
            return True
        if action.startswith("promote_blocked"):
            return False
        # Unknown action strings are not live candidacy claims.
        return False

    @staticmethod
    def _graduation_candidate_alert(data_dir: Path) -> Optional[Dict[str, Any]]:
        from src.dashboard.generator import DashboardGenerator  # lazy (stays in generator)
        data = DashboardGenerator._load_json_file(data_dir / ".promote_to_live")
        if not data:
            return None
        if not DashboardGenerator._is_active_promote_candidacy(data):
            return None

        allowed, blockers = DashboardGenerator._promotion_gate_status(data_dir)
        if allowed:
            return {
                "level": "success",
                "type": "graduation_candidate",
                "title": "Paper Trading Graduation Ready",
                "message": f"Sharpe: {safe_get(data, 'metrics', 'sharpe')}, ready for live approval",
                "timestamp": data.get("timestamp"),
                "requires_action": True,
            }
        return {
            "level": "warning",
            "type": "graduation_candidate",
            "title": "Paper Trading Graduation Blocked",
            "message": "Promotion marker present but current gates block live approval: "
            + ", ".join(sorted(set(blockers))),
            "timestamp": data.get("timestamp"),
            "requires_action": True,
        }

    @staticmethod
    def _stale_data_alerts_from_quality_report(public_dir: Path) -> Optional[List[Dict[str, Any]]]:
        from src.dashboard.generator import DashboardGenerator  # lazy (stays in generator)
        report = DashboardGenerator._load_json_file(public_dir / "data_quality.json")
        if report is None:
            return None

        issue_counts = report.get("issue_counts")
        stale_count = (
            issue_counts.get("stale_latest_dates", 0)
            if isinstance(issue_counts, dict)
            else 0
        )
        if not isinstance(stale_count, int) or isinstance(stale_count, bool) or stale_count <= 0:
            return []

        alerts: List[Dict[str, Any]] = []
        symbols = report.get("symbols")
        rows = symbols if isinstance(symbols, list) else []
        stale_rows = [
            row for row in rows
            if isinstance(row, dict)
            and (
                row.get("stale_latest_date")
                or str(row.get("status", "")).lower() in {"fail", "failed", "critical"}
                or (isinstance(row.get("issue_counts"), dict)
                    and row["issue_counts"].get("stale_latest_dates", 0) > 0)
            )
        ]

        for row in stale_rows[:stale_count]:
            stale_meta = row.get("stale_latest_date") if isinstance(row.get("stale_latest_date"), dict) else {}
            symbol = row.get("symbol", "unknown")
            latest_date = stale_meta.get("latest_date") or row.get("latest_date", "unknown")
            reference_date = stale_meta.get("reference_date") or report.get("reference_date", "unknown")
            lag_days = stale_meta.get("latest_lag_days")
            lag = f" ({lag_days} trading day lag)" if isinstance(lag_days, int) else ""
            alerts.append({
                "level": "warning",
                "type": "stale_data",
                "title": f"Stale Data: {symbol}",
                "message": f"{symbol} latest date {latest_date} lags reference {reference_date}{lag}",
                "timestamp": report.get("generated_at"),
                "requires_action": False,
            })

        while len(alerts) < stale_count:
            alerts.append({
                "level": "warning",
                "type": "stale_data",
                "title": "Stale Data",
                "message": f"data_quality.json reports {stale_count} stale latest-date issue(s)",
                "timestamp": report.get("generated_at"),
                "requires_action": False,
            })
        return alerts



    @staticmethod
    def _empty_incident_summary() -> Dict[str, Any]:
        from src.dashboard import generator as _generator  # lazy (FakeDateTime patch seam)

        return {
            "schema_version": "incident-lifecycle/v1",
            "generated_at": _generator.datetime.now(_generator.timezone.utc).isoformat(),
            "open_count": 0,
            "incidents": [],
            "metrics": {
                "incident_frequency": 0,
                "open_count": 0,
                "resolved_count": 0,
                "mean_mttr_seconds": None,
            },
        }

