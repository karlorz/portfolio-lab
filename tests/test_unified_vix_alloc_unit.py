"""VIX overlay allocation is percent points, not fraction*100."""
import json



def test_vix_alloc_3_percent_points_not_300(tmp_path, monkeypatch):
    import src.monitor.unified_dashboard as ud
    monkeypatch.setattr(ud, "DATA_DIR", tmp_path)
    (tmp_path / "vixy_hedge_state.json").write_text(json.dumps({
        "current_allocation": 3.0,
        "regime": "elevated",
        "last_signal_date": "2026-05-24",
    }))
    section = ud._get_overlays_section()
    vix = section["vix_term_structure"]
    assert vix["active"] is True
    assert abs(float(vix["allocation"]) - 3.0) < 1e-9
    assert vix.get("allocation_unit") == "percent"


def test_vix_alloc_fraction_normalized_to_percent(tmp_path, monkeypatch):
    import src.monitor.unified_dashboard as ud
    monkeypatch.setattr(ud, "DATA_DIR", tmp_path)
    (tmp_path / "vixy_hedge_state.json").write_text(json.dumps({
        "current_allocation": 0.03,
        "regime": "elevated",
    }))
    section = ud._get_overlays_section()
    assert abs(float(section["vix_term_structure"]["allocation"]) - 3.0) < 1e-9


def test_vix_prefers_public_pct_key(tmp_path, monkeypatch):
    import src.monitor.unified_dashboard as ud
    monkeypatch.setattr(ud, "DATA_DIR", tmp_path)
    (tmp_path / "vixy_hedge.json").write_text(json.dumps({
        "current_allocation_pct": 2.5,
        "regime": "elevated",
    }))
    (tmp_path / "vixy_hedge_state.json").write_text(json.dumps({
        "current_allocation": 99.0,
        "regime": "stale",
    }))
    section = ud._get_overlays_section()
    assert abs(float(section["vix_term_structure"]["allocation"]) - 2.5) < 1e-9
