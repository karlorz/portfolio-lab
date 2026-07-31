"""Strict JSON contracts for public/runtime artifact writers and auditors."""

from __future__ import annotations

import json
import math
from pathlib import Path


def _strict_load(text: str):
    def reject_constant(value: str):
        raise AssertionError(f"non-standard JSON constant: {value}")

    return json.loads(text, parse_constant=reject_constant)


def test_generic_serializer_normalizes_nested_non_finite_values() -> None:
    from src.monitor.signal_authority import serialize_json_payload

    body = serialize_json_payload(
        {
            "status": "warning",
            "diagnostics": {
                "nan": math.nan,
                "positive": math.inf,
                "negative": -math.inf,
                "rows": [{"score": math.nan}],
            },
        },
        public=True,
    )

    assert "NaN" not in body
    assert "Infinity" not in body
    assert _strict_load(body)["diagnostics"] == {
        "nan": None,
        "positive": None,
        "negative": None,
        "rows": [{"score": None}],
    }


def test_signals_serializer_and_generic_serializer_share_strict_policy() -> None:
    from src.monitor.signal_authority import serialize_signals_payload

    payload = {
        "target_allocations": {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16},
        "health": {"score": math.nan, "history": [math.inf, -math.inf]},
    }

    generic = _strict_load(
        __import__("src.monitor.signal_authority", fromlist=["serialize_json_payload"])
        .serialize_json_payload(payload, public=True)
    )
    signals = _strict_load(serialize_signals_payload(payload, public=True))

    assert generic["health"] == signals["health"]
    assert generic["target_allocations"] == payload["target_allocations"]


def test_public_private_fanout_preserves_business_values_and_strictness(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from src.dashboard.public_projection import public_business_values_equal
    from src.monitor.signal_authority import write_json_multi_dest

    public_dir = tmp_path / "public" / "data"
    private_dir = tmp_path / "private" / "data"
    monkeypatch.setenv("PUBLIC_DATA_DIR", str(public_dir))
    monkeypatch.setenv("PORTFOLIO_LAB_FORCE_PUBLIC_PROJECTION", "1")
    payload = {
        "status": "warning",
        "artifact_path": str(private_dir / "attribution" / "latest.json"),
        "diagnostics": {"score": math.nan, "history": [math.inf, -math.inf]},
    }

    result = write_json_multi_dest(
        payload,
        public_path=public_dir / "attribution" / "latest.json",
        private_path=private_dir / "attribution" / "latest.json",
        soft_mirror_repo=False,
    )

    assert result.wrote_public is True
    assert result.wrote_private is True
    public_payload = _strict_load(
        (public_dir / "attribution" / "latest.json").read_text(encoding="utf-8")
    )
    private_payload = _strict_load(
        (private_dir / "attribution" / "latest.json").read_text(encoding="utf-8")
    )
    assert public_business_values_equal(private_payload, public_payload)
    assert public_payload["diagnostics"] == {
        "score": None,
        "history": [None, None],
    }
    assert public_payload["artifact_path"] == "internal/latest.json"


def test_public_save_results_json_uses_strict_serializer(tmp_path: Path, monkeypatch) -> None:
    from src.backtest.metrics import save_results_json

    public_dir = tmp_path / "public-data"
    monkeypatch.setenv("PUBLIC_DATA_DIR", str(public_dir))
    output = public_dir / "health.json"

    save_results_json(
        {"status": "warning", "diagnostics": {"score": math.nan}},
        output_path=str(output),
    )

    assert _strict_load(output.read_text(encoding="utf-8"))["diagnostics"]["score"] is None


def test_health_atomic_fallback_uses_strict_serializer(tmp_path: Path) -> None:
    from src.monitor.health_check import _atomic_write_json_path

    output = tmp_path / "public" / "data" / "health.json"
    _atomic_write_json_path(
        output,
        {"status": "warning", "diagnostics": {"score": math.nan}},
    )

    assert _strict_load(output.read_text(encoding="utf-8"))["diagnostics"]["score"] is None


def test_consistency_loader_rejects_python_only_non_standard_constants(tmp_path: Path) -> None:
    from scripts.check_public_data_consistency import _load_json

    path = tmp_path / "health.json"
    path.write_text('{"diagnostics": {"score": Infinity}}', encoding="utf-8")
    errors: list[str] = []

    assert _load_json(path, errors) is None
    assert errors == [f"{path} is not valid JSON: non-standard JSON constant: Infinity"]


def test_consistency_checks_indexed_nested_attribution_shard_strictly(tmp_path: Path) -> None:
    from scripts.check_public_data_consistency import check_public_data_consistency
    from tests.test_generated_public_data_consistency_smoke import (
        _sha256,
        _write_consistent_public_data_set,
    )

    _write_consistent_public_data_set(tmp_path)
    public_data = tmp_path / "public" / "data"
    shard = public_data / "attribution" / "attribution_2026-07-31.json"
    shard.parent.mkdir(parents=True, exist_ok=True)
    shard.write_text('{"diagnostics": {"score": NaN}}', encoding="utf-8")

    index_path = public_data / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["entries"].append(
        {
            "filename": "attribution/attribution_2026-07-31.json",
            "path": "attribution/attribution_2026-07-31.json",
            "status": "present",
            "sha256": _sha256(shard),
        }
    )
    index_path.write_text(json.dumps(index, sort_keys=True), encoding="utf-8")
    result = check_public_data_consistency(tmp_path, env={}, allow_repo_public_data=True)

    assert result.ok is False
    assert any(
        "attribution/attribution_2026-07-31.json is not valid JSON" in error
        for error in result.errors
    )
