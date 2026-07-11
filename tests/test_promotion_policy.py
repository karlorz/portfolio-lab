"""Tests for shared offline experiment promotion governance policy."""

from __future__ import annotations


def test_shared_policy_blocks_metric_promote_when_provenance_missing() -> None:
    from src.research.promotion_policy import classify_offline_promotion_governance

    result = classify_offline_promotion_governance(
        {
            "status": "candidate",
            "provenance_status": "missing",
        },
        metric_gate_status="promoted",
        metric_gate_pass=True,
    )

    assert result["recommended_status"] == "candidate"
    assert result["pass"] is False
    assert result["metric_gate_status"] == "promoted"
    assert result["metric_gate_pass"] is True
    assert result["failures"] == ["provenance_missing"]


def test_shared_policy_rejects_warning_registry_rows() -> None:
    from src.research.promotion_policy import classify_offline_promotion_governance

    result = classify_offline_promotion_governance(
        {
            "status": "warning",
            "provenance_status": "sidecar",
        },
        metric_gate_status="promoted",
        metric_gate_pass=True,
    )

    assert result["recommended_status"] == "rejected"
    assert result["pass"] is False
    assert result["failures"] == ["registry_status_warning"]
