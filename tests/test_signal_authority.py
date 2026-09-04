"""Unit tests for src.monitor.signal_authority (Item Q50).

Tests cover:
- validate_authority_payload (valid 46/38/16 payload, missing keys, invalid allocations sum, invalid types)
- is_champion_target_allocations
- is_ephemeral_write_path
- is_production_ssot_path
- _should_skip_production_ssot_write
- write_signals_multi_dest & try_write_signals_multi_dest
- write_json_multi_dest
- normalize_json_value & _atomic_write_text mode permissions
"""

import json
import os
from pathlib import Path

import pytest

from src.monitor import signal_authority as sa


def test_validate_authority_payload_valid_champion() -> None:
    payload = {
        "target_allocations": {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16},
    }
    # Valid payload should not raise
    sa.validate_authority_payload(payload)


def test_validate_authority_payload_missing_keys_or_not_mapping() -> None:
    with pytest.raises(sa.AuthorityValidationError, match="must be a mapping"):
        sa.validate_authority_payload("not a mapping")  # type: ignore

    with pytest.raises(sa.AuthorityValidationError, match="missing non-empty target_allocations"):
        sa.validate_authority_payload({})

    with pytest.raises(sa.AuthorityValidationError, match="missing non-empty target_allocations"):
        sa.validate_authority_payload({"target_allocations": {}})


def test_validate_authority_payload_invalid_symbols_or_weights() -> None:
    # Invalid symbol type
    with pytest.raises(sa.AuthorityValidationError, match="invalid allocation symbol"):
        sa.validate_authority_payload({"target_allocations": {"": 1.0}})

    # Non-numeric weight
    with pytest.raises(sa.AuthorityValidationError, match="not numeric"):
        sa.validate_authority_payload({"target_allocations": {"SPY": "abc"}})

    # Out of range weights
    with pytest.raises(sa.AuthorityValidationError, match="outside \\(0, 1\\]"):
        sa.validate_authority_payload({"target_allocations": {"SPY": 0.0, "GLD": 0.5, "TLT": 0.5}})

    with pytest.raises(sa.AuthorityValidationError, match="outside \\(0, 1\\]"):
        sa.validate_authority_payload({"target_allocations": {"SPY": 1.5}})

    # Sum does not equal 1.0 (exceeds tolerance)
    with pytest.raises(sa.AuthorityValidationError, match="sum=0.800000 not within"):
        sa.validate_authority_payload(
            {"target_allocations": {"SPY": 0.30, "GLD": 0.30, "TLT": 0.20}},
            require_champion_symbols=False,
        )

    # Missing champion required symbols
    with pytest.raises(sa.AuthorityValidationError, match="missing required symbols"):
        sa.validate_authority_payload(
            {"target_allocations": {"SPY": 0.50, "GLD": 0.50}},
            require_champion_symbols=True,
        )


def test_is_champion_target_allocations() -> None:
    valid = {"target_allocations": {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16}}
    assert sa.is_champion_target_allocations(valid) is True

    # Modified allocation (e.g. defensive 44/36/20)
    defensive = {"target_allocations": {"SPY": 0.44, "GLD": 0.36, "TLT": 0.20}}
    assert sa.is_champion_target_allocations(defensive) is False

    # Missing/invalid target allocations
    assert sa.is_champion_target_allocations({}) is False
    assert sa.is_champion_target_allocations({"target_allocations": "bad"}) is False
    assert sa.is_champion_target_allocations(None) is False  # type: ignore


def test_is_ephemeral_write_path() -> None:
    assert sa.is_ephemeral_write_path("/tmp/plab-pytest-public.123/signals.json") is True
    assert sa.is_ephemeral_write_path("/tmp/pytest-of-root/signals.json") is True
    assert sa.is_ephemeral_write_path("/tmp/pytest-123/signals.json") is True
    assert sa.is_ephemeral_write_path("/var/www/portfolio-lab/data/signals.json") is False
    assert sa.is_ephemeral_write_path(None) is False
    assert sa.is_ephemeral_write_path("") is False


def test_is_production_ssot_path() -> None:
    from src.paths import PROJECT_ROOT

    # SSOT prefixes derive from the current checkout (not a foreign /root tree)
    # plus the live operator /var/www tree.
    project = str(Path(PROJECT_ROOT).resolve())
    assert sa.is_production_ssot_path("/var/www/portfolio-lab/data/signals.json") is True
    assert sa.is_production_ssot_path(f"{project}/data/signals.json") is True
    assert sa.is_production_ssot_path(f"{project}/public/data/signals.json") is True
    # Ephemeral paths are never production SSOT
    assert sa.is_production_ssot_path("/tmp/plab-pytest-public-123/signals.json") is False
    assert sa.is_production_ssot_path(None) is False


def test_should_skip_production_ssot_write(monkeypatch: pytest.MonkeyPatch) -> None:
    # Under pytest, production SSOT writes must be skipped by default
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "test_case")
    monkeypatch.delenv("PORTFOLIO_LAB_ALLOW_LIVE_PUBLIC", raising=False)
    assert sa._should_skip_production_ssot_write("/var/www/portfolio-lab/data/signals.json") is True
    assert sa._should_skip_production_ssot_write("/tmp/plab-pytest-123/signals.json") is False
    assert sa._should_skip_production_ssot_write(None) is False

    # Explicit allow live override
    monkeypatch.setenv("PORTFOLIO_LAB_ALLOW_LIVE_PUBLIC", "1")
    assert sa._should_skip_production_ssot_write("/var/www/portfolio-lab/data/signals.json") is False


def test_normalize_json_value() -> None:
    data = {
        "finite": 1.23,
        "nan": float("nan"),
        "inf": float("inf"),
        "nested": [1, float("-inf"), {"x": float("nan")}],
        "text": "hello",
        "flag": True,
    }
    normalized = sa.normalize_json_value(data)
    assert normalized["finite"] == 1.23
    assert normalized["nan"] is None
    assert normalized["inf"] is None
    assert normalized["nested"] == [1, None, {"x": None}]
    assert normalized["text"] == "hello"
    assert normalized["flag"] is True


def test_write_signals_multi_dest(tmp_path: Path) -> None:
    pub_file = tmp_path / "public" / "signals.json"
    priv_file = tmp_path / "private" / "signals.json"
    repo_file = tmp_path / "repo" / "signals.json"

    payload = {
        "target_allocations": {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16},
        "score": 100,
    }

    res = sa.write_signals_multi_dest(
        payload,
        public_path=pub_file,
        private_path=priv_file,
        repo_path=repo_file,
        soft_mirror_repo=True,
    )

    assert res.wrote_public is True
    assert res.wrote_private is True
    assert res.wrote_repo is True

    pub_data = json.loads(pub_file.read_text(encoding="utf-8"))
    priv_data = json.loads(priv_file.read_text(encoding="utf-8"))
    repo_data = json.loads(repo_file.read_text(encoding="utf-8"))

    assert pub_data["target_allocations"] == {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16}
    assert priv_data["target_allocations"] == {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16}
    assert repo_data["target_allocations"] == {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16}


def test_try_write_signals_multi_dest_failure_logs_and_skips(tmp_path: Path) -> None:
    pub_file = tmp_path / "public" / "signals.json"
    # Missing target_allocations -> fails authority validation
    payload = {"score": 100}

    res = sa.try_write_signals_multi_dest(payload, public_path=pub_file)
    assert res.wrote_public is False
    assert res.skipped_reason is not None
    assert "missing non-empty target_allocations" in res.skipped_reason
    assert not pub_file.exists()


def test_write_json_multi_dest(tmp_path: Path) -> None:
    pub_file = tmp_path / "public" / "alerts.json"
    priv_file = tmp_path / "private" / "alerts.json"
    repo_file = tmp_path / "repo" / "alerts.json"

    payload = {"status": "ok", "alerts": []}

    res = sa.write_json_multi_dest(
        payload,
        public_path=pub_file,
        private_path=priv_file,
        repo_path=repo_file,
        soft_mirror_repo=True,
    )

    assert res.wrote_public is True
    assert res.wrote_private is True
    assert res.wrote_repo is True

    pub_data = json.loads(pub_file.read_text(encoding="utf-8"))
    assert pub_data["status"] == "ok"


def test_atomic_write_text_mode(tmp_path: Path) -> None:
    dest = tmp_path / "readable.json"
    sa._atomic_write_text(dest, '{"test": 1}\n', mode=0o644)
    assert dest.is_file()
    # Check permissions (mask low 9 bits)
    mode = os.stat(dest).st_mode & 0o777
    assert mode == 0o644
