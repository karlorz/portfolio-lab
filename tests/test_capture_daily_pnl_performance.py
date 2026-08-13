"""capture_daily_pnl appends deduped performance.jsonl rows."""
import json



def test_append_performance_jsonl_writes_row(tmp_path, monkeypatch):
    import scripts.capture_daily_pnl as cap
    monkeypatch.setattr(cap, "DATA_DIR", tmp_path)
    snap = {
        "date": "2026-07-20",
        "timestamp": "2026-07-20T15:40:28",
        "total_value": 94162.54,
        "cash": 0.0,
        "daily_return": 0.001,
        "positions_count": 3,
        "mode": "paper",
    }
    assert cap.append_performance_jsonl(snap, performance_path=tmp_path / "performance.jsonl")
    path = tmp_path / "performance.jsonl"
    rows = [json.loads(item) for item in path.read_text().splitlines() if item.strip()]
    assert len(rows) == 1
    assert rows[0]["daily_return"] == 0.001
    assert rows[0]["source"] == "capture_daily_pnl"
    assert rows[0]["total_value"] == 94162.54


def test_append_performance_jsonl_dedupes_same_day(tmp_path):
    import scripts.capture_daily_pnl as cap
    path = tmp_path / "performance.jsonl"
    # prior same-day eval-style row
    path.write_text(json.dumps({
        "timestamp": "2026-07-20T14:20:09",
        "total_value": 94000.0,
        "daily_return": -1e-8,
        "positions_count": 3,
        "mode": "paper",
    }) + "\n")
    snap = {
        "date": "2026-07-20",
        "timestamp": "2026-07-20T15:40:28",
        "total_value": 94162.54,
        "cash": 0.0,
        "daily_return": 0.0,
        "positions_count": 3,
        "mode": "paper",
    }
    cap.append_performance_jsonl(snap, performance_path=path)
    rows = [json.loads(item) for item in path.read_text().splitlines() if item.strip()]
    assert len(rows) == 1
    assert rows[0]["timestamp"] == "2026-07-20T15:40:28"
    assert rows[0]["total_value"] == 94162.54
