"""Action Center: VIX sleeve producers must not be partial-unavailable kill fuel.

Live 2026-07-21: market.db VIX3M proxy rows in vix_term_structure.json lacked
contango_* fields → VIXDataManager skipped every bar → empty cache →
convexity_harvest status=unavailable + volatility_parity null → signal_staleness
kill warning → eval exit 2 → scheduler degraded → rebalance blocked_kill_switch.
"""

from __future__ import annotations

import json

import pytest


def test_vix_from_dict_hydrates_market_db_proxy_rows_without_contango_fields():
    """Shipped from_dict must load market.db-style rows (no contango_* keys)."""
    from src.data.vix_futures import VIXTermStructure

    row = {
        "date": "2026-07-20",
        "vix_spot": 25.0,
        "front_month": 20.4,
        "third_month": 21.0,
        "source": "market.db",
        "as_of": "2026-07-20",
    }
    ts = VIXTermStructure.from_dict(row)
    assert ts.vix_spot == pytest.approx(25.0)
    assert ts.front_month == pytest.approx(20.4)
    assert ts.second_month == pytest.approx(21.0)  # falls back to third_month
    assert ts.third_month == pytest.approx(21.0)
    # contango_spot_1m = (20.4/25 - 1)*100 = -18.4 (backwardation)
    assert ts.contango_spot_1m == pytest.approx((20.4 / 25.0 - 1.0) * 100.0)
    assert ts.contango_1m_2m == pytest.approx((21.0 / 20.4 - 1.0) * 100.0)
    assert ts.is_contango is False


def test_vix_manager_loads_live_shaped_cache_file(tmp_path):
    """_load_cached_data must not skip every market.db proxy bar."""
    from src.data.vix_futures import VIXDataManager

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    vix_file = data_dir / "vix_term_structure.json"
    vix_file.write_text(
        json.dumps(
            {
                "2026-05-12": {
                    "date": "2026-05-12",
                    "vix_spot": 25.0,
                    "front_month": 22.0,
                    "third_month": 21.0,
                },
                "2026-07-20": {
                    "date": "2026-07-20",
                    "vix_spot": 25.0,
                    "front_month": 20.4,
                    "third_month": 21.0,
                    "source": "market.db",
                    "as_of": "2026-07-20",
                },
            }
        )
    )
    mgr = VIXDataManager.__new__(VIXDataManager)
    mgr.DATA_DIR = data_dir
    mgr.VIX_FILE = vix_file
    mgr.data = {}
    mgr._load_cached_data()
    assert len(mgr.data) == 2
    assert "2026-07-20" in mgr.data
    signal = mgr.get_contango_signal("2026-07-20")
    assert signal is not None
    assert signal["vix_level"] == pytest.approx(25.0)


def test_convexity_get_current_signal_uses_last_cache_bar_not_unavailable(tmp_path, monkeypatch):
    """With only lagging cache bars, sleeve is degraded/stale — not unavailable."""
    from src.data.vix_futures import VIXDataManager
    from src.strategy.convexity_harvest import ConvexityHarvestStrategy

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    vix_file = data_dir / "vix_term_structure.json"
    vix_file.write_text(
        json.dumps(
            {
                "2026-07-20": {
                    "date": "2026-07-20",
                    "vix_spot": 16.76,
                    "front_month": 18.0,
                    "second_month": 19.0,
                    "third_month": 19.5,
                    "contango_1m_2m": 5.55,
                    "contango_spot_1m": 7.4,
                    "is_contango": True,
                    "days_to_expiry_front": 10,
                }
            }
        )
    )
    mgr = VIXDataManager.__new__(VIXDataManager)
    mgr.DATA_DIR = data_dir
    mgr.VIX_FILE = vix_file
    mgr.data = {}
    mgr._load_cached_data()
    assert len(mgr.data) == 1

    strategy = ConvexityHarvestStrategy(vix_data_manager=mgr)
    payload = strategy.get_current_signal()
    assert payload["status"] != "unavailable", payload
    assert payload.get("vix_level") is not None
    assert payload.get("generated_at"), "staleness TTL requires generated_at"
    # Today != 2026-07-20 → last-available degraded path
    assert payload["status"] in {"ok", "degraded"}
    if payload["status"] == "degraded":
        assert payload.get("runtime_status") == "stale_futures_cache"


def test_vol_parity_null_vix_does_not_crash_and_publishes_generated_at(tmp_path):
    """Missing VIX must not TypeError; published allocation has TTL stamp."""
    from src.strategy.convexity_harvest import ConvexityHarvestStrategy
    from src.strategy.vol_parity_allocator import VolatilityParityAllocator

    class _EmptyVix:
        data = {}

        def get_contango_signal(self, date):
            return None

        def get_data_range(self):
            return ("", "")

    cx = ConvexityHarvestStrategy(vix_data_manager=_EmptyVix())
    # generate_signal returns unavailable flat with vix_level=None
    pos = cx.generate_signal("2026-07-21")
    assert pos.vix_level is None
    alloc = VolatilityParityAllocator(vix_strategy=cx)
    # Must not raise
    out = alloc.get_current_allocation()
    assert isinstance(out, dict)
    body = out["allocation"]
    assert isinstance(body, dict)
    assert body.get("generated_at")
    assert body.get("weight_unit") == "percent_of_portfolio_0_100"
    assert body.get("live_authoritative") is False


def test_staleness_classifier_does_not_flag_degraded_convexity_with_generated_at():
    """_is_unavailable_signal_block must not treat status=degraded as unavailable."""
    from src.dashboard.generator import DashboardGenerator

    block = {
        "status": "degraded",
        "runtime_status": "stale_futures_cache",
        "freshness_status": "stale",
        "generated_at": "2026-07-21T12:00:00+00:00",
        "vix_level": 16.76,
        "allocation_pct": 2.0,
    }
    assert DashboardGenerator._is_unavailable_signal_block(block) is False

    vol = {
        "status": "degraded",
        "generated_at": "2026-07-21T12:00:00+00:00",
        "spy_pct": 36.8,
        "gld_pct": 30.4,
        "tlt_pct": 12.8,
        "weight_unit": "percent_of_portfolio_0_100",
        "live_authoritative": False,
        "role": "advisory_research_sleeve",
    }
    assert DashboardGenerator._is_unavailable_signal_block(vol) is False

    still_bad = {
        "status": "unavailable",
        "exit_reason": "unavailable: no VIX futures cache",
    }
    assert DashboardGenerator._is_unavailable_signal_block(still_bad) is True
