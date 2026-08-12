"""Batch AP residual honesty: health_ops / index / labs / TS market-data git sha."""

from __future__ import annotations

import json
from pathlib import Path



def test_health_check_report_stamps_generator_git_sha(tmp_path, monkeypatch):
    from src.monitor import health_check as hc

    monkeypatch.setattr(hc, "DATA_DIR", tmp_path)
    monkeypatch.setattr(hc, "HEALTH_PATH", tmp_path / "health.json")
    monkeypatch.setattr(hc, "PUBLIC_DATA_DIR", tmp_path / "public")
    (tmp_path / "public").mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(hc, "_check_data_freshness", lambda: {"prices": {"status": "ok"}})
    monkeypatch.setattr(
        hc,
        "_check_circuit_breaker",
        lambda: {"status": "ok", "state": "closed"},
    )
    monkeypatch.setattr(
        hc,
        "_check_kill_switch",
        lambda: {"enabled": False, "level": "none"},
    )
    monkeypatch.setattr(
        hc,
        "_check_open_incidents",
        lambda: {"open_count": 0, "status": "ok"},
    )
    monkeypatch.setattr(
        hc,
        "_check_fred_md_cache",
        lambda: {"status": "ok", "available": False},
    )
    monkeypatch.setattr(hc, "_compute_system_status", lambda *a, **k: "ok")
    monkeypatch.setattr(
        hc,
        "update_graduation_circuit_breaker_state",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(hc, "publish_ops_health_surfaces", lambda report: None)
    monkeypatch.setattr(hc, "_stamp_health_self_job_running_success", lambda freshness: None)

    monkeypatch.setattr(
        "src.dashboard.generator._generator_git_sha_short",
        lambda: "healthops1234",
    )

    report = hc.run_health_check()
    assert report.get("generator_git_sha") == "healthops1234"
    on_disk = json.loads((tmp_path / "health.json").read_text(encoding="utf-8"))
    assert on_disk.get("generator_git_sha") == "healthops1234"


def test_public_data_index_stamps_and_utc(tmp_path, monkeypatch):
    from src.dashboard import public_data_index as pdi

    monkeypatch.setattr(
        "src.dashboard.generator._generator_git_sha_short",
        lambda: "indexsha12345",
    )
    # Empty public dir → still builds index
    index = pdi.build_public_data_index([], public_dir=tmp_path, use_hash_cache=False)
    assert index["generator_git_sha"] == "indexsha12345"
    assert index["generator_git_sha_status"] == "full_generate"
    # UTC: must include offset or Z
    ga = index["generated_at"]
    assert ga.endswith("+00:00") or ga.endswith("Z") or "+" in ga[10:]


def test_labs_registry_stamps_generator_git_sha(tmp_path, monkeypatch):
    from src.research import experiment_registry as er

    monkeypatch.setattr(
        "src.dashboard.generator._generator_git_sha_short",
        lambda: "labssha123456",
    )
    # No candidates → empty registry still validates
    reg = er.build_labs_registry(data_dirs=[tmp_path], project_root=tmp_path)
    assert reg["generator_git_sha"] == "labssha123456"
    assert reg["generator_git_sha_status"] == "full_generate"
    assert reg["experiments"] == []


def test_price_quality_ts_source_stamps_contract():
    src = Path("src/data/price_quality.ts").read_text(encoding="utf-8")
    assert "export function generatorGitShaShort" in src
    assert "report.generator_git_sha = sha" in src
    assert "generator_git_sha_status" in src


def test_source_manifest_ts_source_stamps_contract():
    src = Path("src/data/source_manifest.ts").read_text(encoding="utf-8")
    assert "generatorGitShaShort" in src
    assert "manifest.generator_git_sha = sha" in src


def test_batch_ap_python_source_contracts():
    health = Path("src/monitor/health_check.py").read_text(encoding="utf-8")
    assert "report = _stamp_generator_git_sha(report)" in health

    index = Path("src/dashboard/public_data_index.py").read_text(encoding="utf-8")
    assert "index = _stamp_generator_git_sha(index)" in index
    assert "datetime.now(timezone.utc)" in index

    labs = Path("src/research/experiment_registry.py").read_text(encoding="utf-8")
    assert "registry = _stamp_generator_git_sha(registry)" in labs
