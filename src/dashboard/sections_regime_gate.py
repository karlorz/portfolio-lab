"""Regime-gate-state mixin extracted from ``src.dashboard.generator``.

Class-level cluster C10 (3 helpers: ``_normalize_gate_regime_name``,
``_resolve_current_regime_for_gate``, ``_persist_regime_state``) moved here
by Item 26 (2026-08-12). ``DashboardGenerator`` inherits
``_RegimeGateStateMixin``. ``generate_regime_gate_json`` stays in the
generator and calls these helpers via ``self.`` (bound refs resolve via
inheritance).

PATCH-TARGET CONTRACT (critical): ``DATA_DIR`` / ``PUBLIC_DIR`` /
``save_results_json`` / ``datetime`` are resolved via call-time lazy imports
from the generator module so tests that patch
``src.dashboard.generator.DATA_DIR`` / ``PUBLIC_DIR`` / ``datetime`` keep
working (test_generator.py:3305/3349/3397, :3271) — module-top src.paths
imports would bypass those patches and write regime_state.json to the real
DATA_DIR during tests. Lazy imports are not circular (call-time resolution;
generator fully loaded).
"""

import json
import logging
from pathlib import Path

from src.utils import classify_vix_regime

logger = logging.getLogger(__name__)


class _RegimeGateStateMixin:
    @staticmethod
    def _normalize_gate_regime_name(regime: str | None) -> str:
        """Map live lowercase regimes to RegimeGate uppercase labels."""
        if not regime:
            return "NORMAL"
        name = str(regime).strip()
        if not name:
            return "NORMAL"
        # Gate rules use NORMAL/HIGH_VOL/…; live classify uses normal/vol_spike/…
        upper = name.upper()
        aliases = {
            "VOL_SPIKE": "HIGH_VOL",
            "VOLSPIKE": "HIGH_VOL",
            "HIGHVOL": "HIGH_VOL",
            "LOWVOL": "LOW_VOL",
            "LOW_VOL": "LOW_VOL",
            "HIGH_VOL": "HIGH_VOL",
            "CRISIS": "CRISIS",
            "RECOVERY": "RECOVERY",
            "NORMAL": "NORMAL",
        }
        return aliases.get(upper.replace("-", "_"), upper.replace("-", "_"))

    def _resolve_current_regime_for_gate(self) -> tuple[str, float, str]:
        """Resolve current regime + confidence for gate SSOT (not live order authority).

        Preference order:
        1. Live VIX/trend classifier via open DB connection (same as signals path)
        2. ensemble_voting on public/data signals.json
        3. regime_classifier_state.json (adaptive path; may be stale)
        4. Explicit default with disclosed source
        """
        from src.dashboard.generator import DATA_DIR, PUBLIC_DIR  # lazy (patch seams)

        # 1) Live VIX path when generator has a DB connection
        conn = getattr(self, "conn", None)
        if conn is not None:
            try:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT close FROM prices WHERE symbol = '^VIX' ORDER BY date DESC LIMIT 1"
                )
                vix_row = cursor.fetchone()
                vix_level = float(vix_row[0]) if vix_row and vix_row[0] is not None else None
                cursor.execute(
                    "SELECT regime FROM regime_log ORDER BY detected_at DESC LIMIT 1"
                )
                trend_row = cursor.fetchone()
                trend_regime = trend_row[0] if trend_row else "normal"
                live = classify_vix_regime(vix_level, trend_regime)
                conf = 0.7 if vix_level is not None else 0.55
                return self._normalize_gate_regime_name(live), conf, "classify_vix_regime"
            except Exception as exc:  # noqa: BLE001 — fall through to file SSOT
                logger.debug("regime_state: VIX path failed: %s", exc)

        # 2) Ensemble voting block on published signals
        for signals_path in (PUBLIC_DIR / "signals.json", DATA_DIR / "signals.json"):
            try:
                if not signals_path.exists():
                    continue
                with open(signals_path) as f:
                    signals = json.load(f)
                ensemble = signals.get("ensemble_voting") or {}
                if ensemble.get("regime") is not None:
                    conf_raw = ensemble.get("regime_confidence", 0.5)
                    try:
                        conf = float(conf_raw)
                    except (TypeError, ValueError):
                        conf = 0.5
                    return (
                        self._normalize_gate_regime_name(str(ensemble.get("regime"))),
                        conf,
                        "ensemble_voting",
                    )
                regime_block = signals.get("regime") or {}
                if isinstance(regime_block, dict) and regime_block.get("regime"):
                    return (
                        self._normalize_gate_regime_name(str(regime_block.get("regime"))),
                        0.6,
                        "signals.regime",
                    )
            except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
                logger.debug("regime_state: signals read failed (%s): %s", signals_path, exc)

        # 3) Adaptive classifier state (legacy parallel file)
        clf_path = DATA_DIR / "regime_classifier_state.json"
        try:
            if clf_path.exists():
                with open(clf_path) as f:
                    clf = json.load(f)
                regime = clf.get("current_regime") or clf.get("regime")
                if regime:
                    conf_raw = clf.get("confidence", 0.5)
                    if isinstance(clf.get("history"), list) and clf["history"]:
                        last = clf["history"][-1]
                        if isinstance(last, dict) and last.get("confidence") is not None:
                            conf_raw = last.get("confidence")
                    try:
                        conf = float(conf_raw)
                    except (TypeError, ValueError):
                        conf = 0.5
                    return (
                        self._normalize_gate_regime_name(str(regime)),
                        conf,
                        "regime_classifier_state",
                    )
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            logger.debug("regime_state: classifier read failed: %s", exc)

        return "NORMAL", 0.5, "default_missing_state"

    def _persist_regime_state(
        self,
        regime_name: str,
        confidence: float,
        source: str,
    ) -> Path:
        """Write DATA_DIR/regime_state.json SSOT for gate + graduation consumers."""
        from src.dashboard.generator import DATA_DIR, save_results_json, datetime  # lazy (patch seams)
        regime_file = DATA_DIR / "regime_state.json"
        history: list = []
        previous = None
        if regime_file.exists():
            try:
                with open(regime_file) as f:
                    prior = json.load(f)
                previous = prior.get("regime")
                hist = prior.get("history")
                if isinstance(hist, list):
                    history = hist[-49:]  # keep last 50 after append
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                history = []

        now_iso = datetime.now().isoformat()
        history.append(
            {
                "timestamp": now_iso,
                "regime": regime_name,
                "confidence": confidence,
                "source": source,
            }
        )
        payload = {
            "regime": regime_name,
            "confidence": confidence,
            "source": source,
            "previous_regime": previous,
            "updated_at": now_iso,
            "schema_version": "regime-state/v1",
            "note": (
                "SSOT for dashboard regime_gate + graduation regime_coverage; "
                "not live order-routing authority (see regime_authority / target_allocations)."
            ),
            "history": history,
        }
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        save_results_json(payload, output_path=str(regime_file))

        # Append one regime_log line so graduation coverage can accumulate over cycles
        try:
            log_path = DATA_DIR / "regime_log.json"
            with open(log_path, "a", encoding="utf-8") as logf:
                logf.write(
                    json.dumps(
                        {
                            "regime": regime_name,
                            "confidence": confidence,
                            "source": source,
                            "detected_at": now_iso,
                        }
                    )
                    + "\n"
                )
        except OSError as exc:
            logger.debug("regime_state: regime_log append failed: %s", exc)

        return regime_file
