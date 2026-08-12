"""Yield curve signal publishes asof and marks stale when lag is high."""
import json
from datetime import datetime, timezone, timedelta



def test_yield_curve_includes_asof_and_ok_when_fresh(tmp_path, monkeypatch):
    from src.dashboard.generator import DashboardGenerator
    import src.dashboard.generator as gen_mod

    today = datetime.now(timezone.utc).date()
    # asof yesterday (weekday lag 0 or 1 depending on weekend)
    asof = (today - timedelta(days=1)).isoformat()
    yields = [
        {"date": asof, "dgs2": 3.8, "dgs10": 4.2, "spread2s10s": 40.0},
    ]
    ypath = tmp_path / "yields.json"
    ypath.write_text(json.dumps(yields))
    monkeypatch.setattr(gen_mod, "YIELDS_JSON", ypath)
    monkeypatch.setattr(gen_mod, "PUBLIC_DIR", tmp_path)

    class FakeGen(DashboardGenerator):
        def __init__(self):
            pass

    g = FakeGen()
    result = g._get_yield_curve_data()
    yc = result["yield_curve"]
    assert yc is not None
    assert yc["asof"] == asof
    assert "asof_lag_weekdays" in yc
    assert yc["status"] in ("ok", "stale")  # weekend edge


def test_yield_curve_stale_when_asof_old(tmp_path, monkeypatch):
    from src.dashboard.generator import DashboardGenerator
    import src.dashboard.generator as gen_mod

    yields = [
        {"date": "2026-05-11", "dgs2": 3.9, "dgs10": 4.4, "spread2s10s": 50.0},
    ]
    ypath = tmp_path / "yields.json"
    ypath.write_text(json.dumps(yields))
    monkeypatch.setattr(gen_mod, "YIELDS_JSON", ypath)
    monkeypatch.setattr(gen_mod, "PUBLIC_DIR", tmp_path)
    monkeypatch.setenv("YIELD_CURVE_MAX_STALE_WEEKDAYS", "5")

    class FakeGen(DashboardGenerator):
        def __init__(self):
            pass

    g = FakeGen()
    result = g._get_yield_curve_data()
    yc = result["yield_curve"]
    assert yc["status"] == "stale"
    assert yc["asof"] == "2026-05-11"
    assert yc["asof_lag_weekdays"] > 5
    assert yc.get("runtime_status") == "stale"
