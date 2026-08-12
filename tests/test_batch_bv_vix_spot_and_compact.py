"""Batch BV: true ^VIX term structure provenance + prices_compact last-N honesty."""

from __future__ import annotations

import json
import sqlite3



def test_vix_loader_skips_meta_key(tmp_path, monkeypatch):
    from src.data import vix_futures as vf

    payload = {
        "_meta": {"spot_source": "^VIX", "spot_is_proxy": False, "n_dates": 1},
        "2026-07-18": {
            "date": "2026-07-18",
            "vix_spot": 15.0,
            "front_month": 16.0,
            "second_month": 16.5,
            "third_month": 17.0,
            "contango_1m_2m": 3.0,
            "contango_spot_1m": 6.67,
            "is_contango": True,
            "days_to_expiry_front": 0,
        },
    }
    path = tmp_path / "vix_term_structure.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(vf.VIXDataManager, "VIX_FILE", path)
    monkeypatch.setattr(vf.VIXDataManager, "DATA_DIR", tmp_path)
    mgr = vf.VIXDataManager()
    assert "_meta" not in mgr.data
    assert "2026-07-18" in mgr.data
    assert mgr.data["2026-07-18"].vix_spot == 15.0


def test_update_vix_prefers_true_spot(tmp_path):
    from scripts.update_vix_term_structure import update_vix_term_structure

    db = tmp_path / "market.db"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE prices (symbol TEXT, date TEXT, close REAL)"
    )
    rows = []
    for i, (d, vix, v3) in enumerate(
        [
            ("2026-07-15", 14.0, 15.0),
            ("2026-07-16", 14.5, 15.2),
            ("2026-07-17", 15.0, 15.5),
        ]
    ):
        rows.append(("^VIX", d, vix))
        rows.append(("^VIX3M", d, v3))
    conn.executemany("INSERT INTO prices VALUES (?,?,?)", rows)
    conn.commit()
    conn.close()

    data_dir = tmp_path / "data"
    pub = tmp_path / "public"
    data_dir.mkdir()
    ok = update_vix_term_structure(data_dir=data_dir, public_dir=pub, market_db=db)
    assert ok is True
    body = json.loads((data_dir / "vix_term_structure.json").read_text())
    assert body["_meta"]["spot_source"] == "^VIX"
    assert body["_meta"]["spot_is_proxy"] is False
    assert body["_meta"]["front_source"] == "^VIX3M"
    assert abs(body["2026-07-17"]["vix_spot"] - 15.0) < 1e-9
    assert abs(body["2026-07-17"]["front_month"] - 15.5) < 1e-9
    # public dual-write
    assert (pub / "vix_term_structure.json").exists()
    pub_meta = json.loads((pub / "vix_term_structure.json").read_text())["_meta"]
    assert pub_meta["spot_is_proxy"] is False


def test_update_vix_proxy_when_no_spot(tmp_path):
    from scripts.update_vix_term_structure import update_vix_term_structure

    db = tmp_path / "market.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE prices (symbol TEXT, date TEXT, close REAL)")
    conn.execute("INSERT INTO prices VALUES ('^VIX3M','2026-07-17',16.0)")
    conn.commit()
    conn.close()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    ok = update_vix_term_structure(
        data_dir=data_dir, public_dir=tmp_path / "pub", market_db=db
    )
    assert ok
    meta = json.loads((data_dir / "vix_term_structure.json").read_text())["_meta"]
    assert meta["spot_is_proxy"] is True
    assert meta["spot_source"] == "^VIX3M"


def test_compact_from_prices_last_n():
    from scripts.rebuild_prices_compact import compact_from_prices

    prices = {
        "SPY": [{"d": i, "c": float(i)} for i in range(1000)],
        "GLD": [{"d": i, "c": float(i)} for i in range(200)],
    }
    out = compact_from_prices(prices, n_bars=504)
    assert out["meta"]["schema"] == "prices/compact-v1"
    assert out["meta"]["n_bars"] == 504
    assert len(out["symbols"]["SPY"]) == 504
    assert len(out["symbols"]["GLD"]) == 200  # shorter series kept whole
    assert out["symbols"]["SPY"][0]["d"] == 1000 - 504


def test_signal_load_vix_strips_meta(tmp_path, monkeypatch):
    from src.signals.vix_term_structure import VIXTermStructureSignalGenerator

    path = tmp_path / "vix_term_structure.json"
    path.write_text(
        json.dumps(
            {
                "_meta": {"spot_source": "^VIX"},
                "2026-07-10": {
                    "date": "2026-07-10",
                    "vix_spot": 12.0,
                    "front_month": 13.0,
                    "regime": "contango",
                },
            }
        ),
        encoding="utf-8",
    )
    sig = VIXTermStructureSignalGenerator.__new__(VIXTermStructureSignalGenerator)
    sig.VIX_DATA_PATH = path
    data = sig.load_vix_data()
    assert "_meta" not in data
    assert sig._file_latest_as_of(data) == "2026-07-10"
