"""Regime / allocation-authority mixin extracted from ``src.dashboard.generator``.

Class-level cluster C3 (9 methods: allocation-surface roles, advisory
artifact roles, regime authority rollups and the regime-authority
availability classmethod) moved here by Item 21 (2026-08-12).
``DashboardGenerator`` inherits ``_RegimeAuthorityMixin``. Zero
class-qualified refs (audit 18:55Z); ``cls._is_unavailable_signal_block``
resolves via MRO (classmethod).
"""

from pathlib import Path
from typing import Any, Dict

from src.paths import BASE_ALLOCATION, DATA_DIR, REGIME_OVERRIDES
from src.strategy.regime_allocation import normalize_allocation_regime
from src.dashboard.kill_authority import (
    allocation_roles_under_kill,
    load_kill_switch_payload,
    project_kill_switch_fields,
)


class _RegimeAuthorityMixin:
    @staticmethod
    def _build_allocation_surface_roles(data_dir: Path | None = None) -> Dict[str, Any]:
        """Describe the current live-routing role of allocation-like signals surfaces.

        When the kill switch is enabled, target_allocations remains the routing
        surface but is disclosed as execution-blocked (not live_authoritative).
        """
        advisory_description = (
            "Published for advisory diagnostics; current order routing uses "
            "target_allocations."
        )
        roles: Dict[str, Any] = {
            "schema_version": "allocation-surface-roles/v1",
            "routed_surface": "target_allocations",
            "routed_by": "src.broker.order_router",
            "surfaces": {
                "target_allocations": {
                    "label": "Target Allocation",
                    "role": "execution_routed",
                    "routed": True,
                    "routed_by": "src.broker.order_router",
                    "live_authoritative": True,
                    "canonical_controller": "signals.json.target_allocations",
                    "description": (
                        "Current order-routing input consumed by src.broker.order_router."
                    ),
                },
                "ensemble_voting": {
                    "label": "Ensemble Voting",
                    "role": "advisory_non_routed",
                    "routed": False,
                    "routed_by": None,
                    "live_authoritative": False,
                    "canonical_controller": "signals.json.target_allocations",
                    "description": advisory_description,
                },
                "adaptive_sizing": {
                    "label": "Adaptive Sizing",
                    "role": "advisory_non_routed",
                    "routed": False,
                    "routed_by": None,
                    "live_authoritative": False,
                    "canonical_controller": "signals.json.target_allocations",
                    "description": advisory_description,
                },
                "black_litterman": {
                    "label": "Black-Litterman",
                    "role": "advisory_non_routed",
                    "routed": False,
                    "routed_by": None,
                    "live_authoritative": False,
                    "canonical_controller": "signals.json.target_allocations",
                    "description": advisory_description,
                },
                "calendar_seasonality": {
                    "label": "Calendar Seasonality",
                    "role": "advisory_non_routed",
                    "routed": False,
                    "routed_by": None,
                    "live_authoritative": False,
                    "applies_to_target_allocations": False,
                    "canonical_controller": "signals.json.target_allocations",
                    "description": (
                        "Urgency/execution timing advisory only. "
                        "modifier does not scale target_allocations "
                        "(paper book stays champion weights)."
                    ),
                },
                "factor_rotation": {
                    "label": "Factor Rotation",
                    "role": "advisory_non_routed",
                    "routed": False,
                    "routed_by": None,
                    "live_authoritative": False,
                    "allocation_field": "allocation",
                    "canonical_controller": "signals.json.target_allocations",
                    "description": (
                        "Advisory factor sleeve weights (e.g. VLUE/VBR). "
                        "Not order-routed; live routing uses target_allocations."
                    ),
                },
            },
        }
        root = Path(data_dir) if data_dir is not None else DATA_DIR
        kill = project_kill_switch_fields(load_kill_switch_payload(root))
        if kill.get("enabled"):
            roles = allocation_roles_under_kill(
                roles,
                kill_enabled=True,
                kill_level=kill.get("level"),
            )
        return roles

    @staticmethod
    def _build_advisory_allocation_artifact_role(
        surface: str,
        allocation_field: str,
    ) -> Dict[str, Any]:
        """Describe a standalone allocation artifact as advisory/non-routed."""
        return {
            "schema_version": "allocation-artifact-role/v1",
            "surface": surface,
            "allocation_field": allocation_field,
            "runtime_role": "advisory_non_routed",
            "live_authoritative": False,
            "routed": False,
            "routed_by": None,
            "canonical_controller": "signals.json.target_allocations",
            "routed_surface": "target_allocations",
            "routed_surface_path": "public/data/signals.json#target_allocations",
            "description": (
                f"{surface} is published for advisory diagnostics; live order routing "
                "continues to consume signals.json.target_allocations."
            ),
        }

    @staticmethod
    def _flatten_advisory_authority(authority: Dict[str, Any]) -> Dict[str, Any]:
        """Top-level authority fields for operator greps (vixy_hedge pattern).

        Nested ``authority`` remains the schema contract for AdaptiveSizingPanel;
        top-level mirrors make ``runtime_role`` / ``routed`` visible without
        digging into the nested block.
        """
        return {
            "runtime_role": authority.get("runtime_role"),
            "live_authoritative": authority.get("live_authoritative"),
            "routed": authority.get("routed"),
            "routed_by": authority.get("routed_by"),
            "canonical_controller": authority.get("canonical_controller"),
            "routed_surface": authority.get("routed_surface"),
        }

    @staticmethod
    def _canonicalize_public_weights(
        weights: Dict[str, Any],
        canonical_assets: tuple[str, ...] = ("SPY", "GLD", "TLT"),
    ) -> Dict[str, Any]:
        """Uppercase public weight keys and preserve zero-weight diagnostics."""
        normalized: Dict[str, float] = {}
        excluded_assets: list[str] = []
        for symbol, raw_weight in (weights or {}).items():
            canonical = str(symbol).upper()
            try:
                normalized[canonical] = float(raw_weight)
            except (TypeError, ValueError):
                excluded_assets.append(canonical)

        public_weights = {
            symbol: normalized.get(symbol, 0.0)
            for symbol in canonical_assets
        }
        zero_weight_assets = [
            symbol for symbol, weight in public_weights.items()
            if abs(weight) < 1e-12
        ]

        return {
            "weights": public_weights,
            "excluded_assets": sorted(set(excluded_assets)),
            "zero_weight_assets": zero_weight_assets,
        }

    @staticmethod
    def _resolve_live_target_allocations_for_regime(
        current_regime: str,
    ) -> Dict[str, float]:
        """Return the only currently approved order-routing target allocation."""
        _ = current_regime
        return dict(BASE_ALLOCATION)

    @staticmethod
    def _build_regime_allocation_diagnostic(current_regime: str) -> Dict[str, Any]:
        """Expose regime-derived allocation candidates without routing them."""
        allocation_regime = normalize_allocation_regime(current_regime) or "normal"
        candidate = REGIME_OVERRIDES.get(current_regime)
        return {
            "schema_version": "regime-allocation-diagnostic/v1",
            "role": "advisory_non_routed",
            "live_authoritative": False,
            "routed": False,
            "allocation_regime": allocation_regime,
            "candidate_target_allocations": dict(candidate or BASE_ALLOCATION),
            "canonical_controller": "signals.json.target_allocations",
            "description": (
                "Regime allocation candidate is diagnostic only; current hard rule "
                "keeps live target_allocations at the champion baseline."
            ),
        }

    @staticmethod
    def _build_regime_authority(
        current_regime: str,
        target_alloc: Dict[str, float],
    ) -> Dict[str, Any]:
        """Document the routed target allocation controller and advisory regimes."""
        return {
            "schema_version": "regime-authority/v1",
            "live_controller": "signals.json.target_allocations",
            "live_controller_module": "src.broker.order_router",
            "live_regime": current_regime,
            "allocation_regime": normalize_allocation_regime(current_regime) or "normal",
            "routed_surface": "target_allocations",
            "target_allocations": target_alloc,
            "regime_controller": "classify_vix_regime",
            "regime_controller_module": "src.utils.classify_vix_regime",
            "regime_routed": False,
            "advanced_regime_signals": {
                "two_stage_regime": {
                    "role": "advisory_shadow",
                    "routed": False,
                    "availability": "unknown",
                    "published": False,
                    "description": "Availability pending staleness check; not live order-routing authority.",
                },
                "bocd_regime": {
                    "role": "advisory_shadow",
                    "routed": False,
                    "availability": "unknown",
                    "published": False,
                    "description": "Availability pending staleness check; not live order-routing authority.",
                },
                "regime_transition": {
                    "role": "advisory_shadow",
                    "routed": False,
                    "availability": "unknown",
                    "published": False,
                    "description": "Availability pending staleness check; not live order-routing authority.",
                },
            },
        }

    @staticmethod
    def _dedupe_preserve_order(values: list[str]) -> list[str]:
        """Return unique string identifiers while preserving first occurrence order."""
        seen: set[str] = set()
        result: list[str] = []
        for value in values:
            key = str(value)
            if key in seen:
                continue
            seen.add(key)
            result.append(key)
        return result

    @classmethod
    def _update_regime_authority_availability(cls, output: Dict[str, Any]) -> None:
        """Update advanced regime authority entries with observed snapshot state."""
        authority = output.get("regime_authority")
        if not isinstance(authority, dict):
            return

        advanced = authority.get("advanced_regime_signals")
        if not isinstance(advanced, dict):
            return

        staleness = output.get("staleness") if isinstance(output.get("staleness"), dict) else {}
        unavailable = set(staleness.get("unavailable_signals") or [])
        stale = set(staleness.get("stale_signals") or [])

        for signal_name, entry in advanced.items():
            if not isinstance(entry, dict):
                continue

            signal_block = output.get(signal_name)
            if signal_name in unavailable or signal_block is None:
                availability = "unavailable"
                published = False
                description = "Unavailable in this snapshot; not live order-routing authority."
            elif signal_name in stale:
                availability = "stale"
                published = False
                description = "Stale in this snapshot; not live order-routing authority."
            elif cls._is_unavailable_signal_block(signal_block):
                availability = "error"
                published = False
                description = "Error or degraded placeholder in this snapshot; not live order-routing authority."
            else:
                availability = "present"
                published = True
                description = "Published for advisory diagnostics; not live order-routing authority."

            entry.update({
                "availability": availability,
                "published": published,
                "description": description,
            })
