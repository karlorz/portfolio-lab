"""Graduation / explainability / risk-decomposition mixin extracted from
``src.dashboard.generator``.

Class-level cluster C9 (5 helpers: ``_graduation_display_value``,
``_paper_trading_summary_for_dashboard``,
``_latest_stale_explainability_metadata``,
``_build_unavailable_explainability_payload``,
``_load_risk_decomposition_signal_section`` incl. nested ``_stamp``) moved
here by Item 25 (2026-08-12). ``DashboardGenerator`` inherits
``_GraduationExplainabilitySectionsMixin``. The 3 entry methods
(``generate_graduation_json`` / ``generate_explainability_json`` /
``generate_risk_decomposition_json``) stay in the generator and call these
helpers via ``self.`` (bound refs resolve via inheritance). datetime.now
deferred through the generator module (FakeDateTime patch seam, rule
136e2d9); ``decompose_portfolio`` stays a function-local lazy import.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from src.paths import BASE_ALLOCATION, DATA_DIR, PUBLIC_DATA_DIR

PUBLIC_DIR = PUBLIC_DATA_DIR  # same alias as generator.py (src.paths has no PUBLIC_DIR)

logger = logging.getLogger(__name__)


class _GraduationExplainabilitySectionsMixin:
    @staticmethod
    def _graduation_display_value(value: Any) -> str:
        """Format checklist numeric/bool values for the dashboard criterion table."""
        if isinstance(value, bool):
            return "yes" if value else "no"
        if isinstance(value, int) and not isinstance(value, bool):
            return str(value)
        if isinstance(value, float):
            if value == 0.0:
                return "0"
            if abs(value) >= 100:
                return f"{value:.1f}"
            if abs(value) >= 1:
                return f"{value:.2f}"
            return f"{value:.4f}".rstrip("0").rstrip(".")
        return str(value)

    @staticmethod
    def _paper_trading_summary_for_dashboard(
        state: Dict[str, Any],
        *,
        days_elapsed: Any,
        days_required: Any,
    ) -> Dict[str, Any]:
        """Build frontend paper_trading block from checklist-loaded state.

        Dual-shape: dashboard GraduationDataSchema requires start_date /
        initial_capital / current_value / days_elapsed / days_required while
        the producer keeps trading_days / min_trading_days at the top level.
        """
        portfolio = state.get("portfolio") if isinstance(state.get("portfolio"), dict) else {}
        summary = (
            state.get("paper_trading_summary")
            if isinstance(state.get("paper_trading_summary"), dict)
            else {}
        )
        history = portfolio.get("history") if isinstance(portfolio.get("history"), list) else []

        start_date = ""
        if history and isinstance(history[0], dict):
            ts = history[0].get("timestamp")
            if isinstance(ts, str) and ts:
                start_date = ts[:10]
        if not start_date:
            date_hint = summary.get("date")
            if isinstance(date_hint, str) and date_hint:
                start_date = date_hint[:10]

        initial_capital = 100_000.0
        current_value: Optional[float] = None

        if history and isinstance(history[0], dict):
            start_val = history[0].get("total_value")
            if isinstance(start_val, (int, float)):
                initial_capital = float(start_val)
        if history and isinstance(history[-1], dict):
            end_val = history[-1].get("total_value")
            if isinstance(end_val, (int, float)):
                current_value = float(end_val)

        # Prefer authoritative paper-trading-performance metrics when present.
        start_value_files = sorted(DATA_DIR.glob("paper-trading-performance-*.json"))
        if start_value_files:
            try:
                with open(start_value_files[-1]) as f:
                    perf_raw = json.load(f)
                if isinstance(perf_raw, dict):
                    if not start_date and isinstance(perf_raw.get("date"), str):
                        start_date = perf_raw["date"][:10]
                    perf = perf_raw.get("performance")
                    if isinstance(perf, dict):
                        sv = perf.get("start_value")
                        if isinstance(sv, (int, float)):
                            initial_capital = float(sv)
                        cv = perf.get("current_value")
                        if isinstance(cv, (int, float)):
                            current_value = float(cv)
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                pass

        if current_value is None:
            cash = portfolio.get("cash")
            positions = portfolio.get("positions")
            if isinstance(cash, (int, float)) and isinstance(positions, dict):
                pos_sum = 0.0
                for pos in positions.values():
                    if isinstance(pos, dict) and isinstance(pos.get("value"), (int, float)):
                        pos_sum += float(pos["value"])
                current_value = float(cash) + pos_sum
        if current_value is None:
            current_value = initial_capital

        try:
            days_elapsed_n = int(days_elapsed) if days_elapsed is not None else 0
        except (TypeError, ValueError):
            days_elapsed_n = 0
        try:
            days_required_n = int(days_required) if days_required is not None else 0
        except (TypeError, ValueError):
            days_required_n = 0

        if not start_date:
            from src.dashboard import generator as _generator  # lazy (patch seams)
            start_date = _generator.datetime.now().date().isoformat()

        return {
            "start_date": start_date,
            "initial_capital": round(initial_capital, 2),
            "current_value": round(float(current_value), 2),
            "days_elapsed": days_elapsed_n,
            "days_required": days_required_n,
        }

    @staticmethod
    def _latest_stale_explainability_metadata(source_dir: Path) -> Dict[str, Any]:
        """Return metadata for the newest historical explainability file, if any."""
        dated_files = sorted(source_dir.glob("explainability_*.json"), reverse=True)
        if not dated_files:
            return {}

        latest = dated_files[0]
        metadata: Dict[str, Any] = {"stale_source_file": latest.name}
        try:
            payload = json.loads(latest.read_text())
            analysis_date = payload.get("analysis_date")
            if analysis_date:
                metadata["stale_analysis_date"] = str(analysis_date)
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as e:
            metadata["stale_read_error"] = str(e)
        return metadata

    @staticmethod
    def _build_unavailable_explainability_payload(
        *,
        generated_at: str,
        reason: str,
        source_file: Optional[str] = None,
        analysis_date: Optional[str] = None,
        stale_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build an explicit no-current-explainability artifact."""
        return {
            "timestamp": generated_at,
            "analysis_date": analysis_date or generated_at[:10],
            "latest_decision": None,
            "recent_decisions": [],
            "signal_deep_dives": {},
            "top_sources_today": [],
            "decision_quality": {
                "status": "unavailable_current_signals",
                "reason": reason,
            },
            "freshness": {
                "status": "unavailable",
                "generated_at": generated_at,
                "source_file": source_file,
                "reason": reason,
                **(stale_metadata or {}),
            },
        }

    def _load_risk_decomposition_signal_section(self) -> Optional[Dict[str, Any]]:
        """Embed risk decomposition into signals.json for optional staleness TTL.

        Prefer computing live; fall back to the public sidecar when present so
        the section is not left missing (None → optional unavailable forever).
        """
        from src.dashboard import generator as _generator  # lazy (patch seams)
        now_ts = _generator.datetime.now(_generator.timezone.utc).isoformat()

        def _stamp(payload: Dict[str, Any]) -> Dict[str, Any]:
            payload.setdefault("generated_at", now_ts)
            payload.setdefault("timestamp", payload.get("generated_at") or now_ts)
            return payload

        try:
            from src.monitor.risk_decomposition import decompose_portfolio

            result = decompose_portfolio(weights=BASE_ALLOCATION)
            return _stamp(result.to_dict())
        except Exception as exc:  # noqa: BLE001 — optional section
            logger.debug("Live risk_decomposition embed skipped: %s", exc)

        path = PUBLIC_DIR / "risk_decomposition.json"
        if not path.exists():
            return None
        try:
            with open(path) as f:
                data = json.load(f)
            if not isinstance(data, dict):
                return None
            # Explicit unavailable/error sidecars stay unavailable for operators.
            if data.get("status") in {"unavailable", "error"} or "error" in data:
                return _stamp(data)
            # Successful decompose payloads should not look unavailable.
            data.pop("status", None)
            return _stamp(data)
        except (OSError, json.JSONDecodeError, TypeError):
            return None
