"""Batch AJ residual honesty: paper Sharpe implausibility + overlay advisory roles."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock



def test_paper_portfolio_marks_implausible_high_sharpe(tmp_path, monkeypatch):
    """stats.paper_portfolio keeps raw Sharpe > 3 and flags implausible (not zero)."""
    from src.dashboard.generator import DashboardGenerator

    # Minimal generator with fake performance.jsonl of near-constant returns
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    public = tmp_path / "public"
    public.mkdir()
    perf = data_dir / "performance.jsonl"
    # Positive drift + small noise → annualized Sharpe >> 3 (implausible short sample)
    lines = []
    val = 100000.0
    for i in range(25):
        r = 0.002 + (0.0001 if i % 2 == 0 else -0.00005)
        val *= 1 + r
        day = i + 1
        month = 6 if day <= 30 else 7
        d = day if day <= 30 else day - 30
        lines.append(
            json.dumps(
                {
                    "timestamp": f"2026-{month:02d}-{d:02d}T16:00:00+00:00",
                    "date": f"2026-{month:02d}-{d:02d}",
                    "total_value": val,
                    "daily_return": r,
                }
            )
        )
    perf.write_text("\n".join(lines) + "\n")

    monkeypatch.setattr("src.dashboard.generator.DATA_DIR", data_dir)
    monkeypatch.setattr("src.dashboard.generator.PUBLIC_DIR", public)

    gen = DashboardGenerator.__new__(DashboardGenerator)
    # Fake sqlite cursor path used only for SPY comparison — empty is fine
    gen.conn = MagicMock()
    gen.conn.cursor.return_value.fetchall.return_value = []
    gen.conn.cursor.return_value.execute = MagicMock()

    out = DashboardGenerator.generate_stats_json(gen)
    payload = json.loads(Path(out).read_text())
    paper = payload["paper_portfolio"]
    assert paper["sharpe"] > 3.0
    assert paper["sharpe_implausible"] is True
    assert paper["sharpe_plausibility_status"] == "implausible_short_sample"
    assert paper["sharpe"] != 0
    assert "implausible" in (paper.get("sharpe_note") or "").lower()


def test_overlay_crypto_bond_collar_kurtosis_carry_advisory_roles():
    """Overlay sleeves disclose non-routed authority."""
    src = Path("src/dashboard/overlay_dashboard.py").read_text(encoding="utf-8")
    assert src.count('"role": "advisory_non_routed"') >= 2
    assert src.count('"role": "advisory_overlay"') >= 1
    assert src.count('"live_authoritative": False') >= 4
    assert "weight_unit" in src
