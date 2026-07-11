"""Shared offline experiment promotion governance policy."""

from __future__ import annotations

from typing import Any, Mapping

CLEAN_PROVENANCE_STATUSES = {"present", "embedded", "sidecar"}
BAD_PROVENANCE_STATUSES = {"malformed", "stale", "invalid", "lost"}
REJECT_REGISTRY_STATUSES = {"warning", "rejected", "archived"}
GOVERNANCE_CLEAR = "clear"
GOVERNANCE_BLOCKED = "governance_blocked"
GOVERNANCE_REJECTED = "rejected"


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _normalized_str(value: Any, default: str = "unknown") -> str:
    if value is None:
        return default
    text = str(value).strip().lower()
    return text or default


def provenance_status(row: Mapping[str, Any]) -> str:
    artifacts = _mapping(row.get("artifacts"))
    return _normalized_str(row.get("provenance_status", artifacts.get("provenance_status")))


def registry_status(row: Mapping[str, Any]) -> str:
    artifacts = _mapping(row.get("artifacts"))
    promotion = _normalized_str(row.get("promotion_status"), default="")
    if promotion in REJECT_REGISTRY_STATUSES:
        return promotion
    return _normalized_str(
        row.get(
            "registry_status",
            artifacts.get("registry_status", row.get("status", row.get("promotion_status"))),
        ),
        default="candidate",
    )


def is_clean_provenance_status(status: str | None) -> bool:
    return _normalized_str(status) in CLEAN_PROVENANCE_STATUSES


def is_rejecting_provenance_status(status: str | None) -> bool:
    return _normalized_str(status) in BAD_PROVENANCE_STATUSES


def is_rejecting_registry_status(status: str | None) -> bool:
    return _normalized_str(status, default="candidate") in REJECT_REGISTRY_STATUSES


def governance_failures(row: Mapping[str, Any]) -> list[str]:
    """Return governance blockers that prevent promotion-ready status."""
    failures: list[str] = []
    row_status = registry_status(row)
    prov_status = provenance_status(row)

    if is_rejecting_registry_status(row_status):
        failures.append(f"registry_status_{row_status}")
    if not is_clean_provenance_status(prov_status):
        failures.append(f"provenance_{prov_status}")
    return failures


def classify_offline_promotion_governance(
    row: Mapping[str, Any],
    *,
    metric_gate_status: str,
    metric_gate_pass: bool,
    metric_failures: list[str] | None = None,
) -> dict[str, Any]:
    """Classify canonical promotion readiness after metric and governance gates."""
    metric_blockers = list(metric_failures or [])
    if not metric_gate_pass and not metric_blockers:
        metric_blockers.append("metric_gate_failed")

    gov_failures = governance_failures(row)
    failures = metric_blockers + gov_failures
    row_status = registry_status(row)
    prov_status = provenance_status(row)

    if is_rejecting_registry_status(row_status) or is_rejecting_provenance_status(prov_status):
        recommended_status = "rejected"
    elif metric_gate_status == "promoted" and metric_gate_pass and not gov_failures:
        recommended_status = "promoted"
    else:
        recommended_status = "candidate"

    return {
        "recommended_status": recommended_status,
        "pass": recommended_status == "promoted",
        "failures": failures,
        "metric_gate_status": metric_gate_status,
        "metric_gate_pass": metric_gate_pass,
        "governance_status": row_status,
        "provenance_status": prov_status,
    }


def governance_state(row: Mapping[str, Any]) -> str:
    """Return the public governance disclosure state for an offline experiment row."""
    row_status = registry_status(row)
    prov_status = provenance_status(row)
    if is_rejecting_registry_status(row_status) or is_rejecting_provenance_status(prov_status):
        return GOVERNANCE_REJECTED
    if governance_failures(row):
        return GOVERNANCE_BLOCKED
    return GOVERNANCE_CLEAR


def governance_disclosure_fields(
    row: Mapping[str, Any],
    *,
    metric_gate_status: str,
    metric_gate_pass: bool,
    metric_failures: list[str] | None = None,
) -> dict[str, Any]:
    """Build portable governance fields shared by Labs registry and scorecards."""
    promotion_governance = classify_offline_promotion_governance(
        row,
        metric_gate_status=metric_gate_status,
        metric_gate_pass=metric_gate_pass,
        metric_failures=metric_failures,
    )
    return {
        "governance_state": governance_state(row),
        "governance_reasons": governance_failures(row),
        "promotion_governance": promotion_governance,
    }
