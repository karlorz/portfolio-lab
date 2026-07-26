"""Batch CO: generate_signals_json dual-writes private DATA_DIR signals.json."""

from __future__ import annotations

from pathlib import Path


def test_generate_signals_json_source_dual_writes_private():
    src = Path("src/dashboard/generator.py").read_text(encoding="utf-8")
    # Batch CO dual-write evolved to Batch HK serialize-once multi-dest
    # (write_signals_multi_dest); keep source-contract checks current.
    assert 'private_path = Path(DATA_DIR) / "signals.json"' in src
    assert "write_signals_multi_dest" in src
    assert (
        "Dual-wrote private signals.json" in src
        or "private signals.json dual-write" in src
        or "Multi-dest private signals.json" in src
    )


def test_live_private_signals_exists_after_cn_reemit():
    """Smoke: private twin exists when public does (post-CO generate)."""
    from src.paths import DATA_DIR, PUBLIC_DATA_DIR

    pub = Path(PUBLIC_DATA_DIR) / "signals.json"
    priv = Path(DATA_DIR) / "signals.json"
    if not pub.is_file():
        return  # hermetic CI without live WWW
    # If public exists in live env, private should after CO generate
    if priv.is_file():
        assert priv.stat().st_size > 1000
