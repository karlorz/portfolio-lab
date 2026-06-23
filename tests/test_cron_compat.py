"""
Tests for src/cron_compat.py — dual-mode cron backend compatibility.
No ML dependencies, no network calls.
"""
import pytest
import os
import sys
from pathlib import Path
from unittest.mock import patch


class TestBackendDetection:
    """CRON_BACKEND env var detection."""

    def test_default_is_hermes(self):
        with patch.dict(os.environ, {}, clear=True):
            import importlib
            import src.cron_compat as cc
            importlib.reload(cc)
            assert cc.BACKEND == "hermes"
            assert cc.IS_HERMES is True
            assert cc.IS_CRONTAB is False
            assert cc.IS_MANUAL is False

    def test_crontab_backend(self):
        with patch.dict(os.environ, {"CRON_BACKEND": "crontab"}, clear=True):
            import importlib
            import src.cron_compat as cc
            importlib.reload(cc)
            assert cc.BACKEND == "crontab"
            assert cc.IS_HERMES is False
            assert cc.IS_CRONTAB is True

    def test_manual_backend(self):
        with patch.dict(os.environ, {"CRON_BACKEND": "manual"}, clear=True):
            import importlib
            import src.cron_compat as cc
            importlib.reload(cc)
            assert cc.BACKEND == "manual"
            assert cc.IS_MANUAL is True

    def test_claude_code_is_manual(self):
        with patch.dict(os.environ, {"CRON_BACKEND": "claude-code"}, clear=True):
            import importlib
            import src.cron_compat as cc
            importlib.reload(cc)
            assert cc.IS_MANUAL is True

    def test_unknown_backend(self):
        with patch.dict(os.environ, {"CRON_BACKEND": "unknown"}, clear=True):
            import importlib
            import src.cron_compat as cc
            importlib.reload(cc)
            assert cc.BACKEND == "unknown"
            assert cc.IS_HERMES is False
            assert cc.IS_CRONTAB is False
            assert cc.IS_MANUAL is False


class TestCRONTargets:
    """CRON_TARGETS list integrity."""

    def test_targets_is_list(self):
        from src.cron_compat import CRON_TARGETS
        assert isinstance(CRON_TARGETS, list)

    def test_fifteen_default_targets(self):
        from src.cron_compat import CRON_TARGETS
        assert len(CRON_TARGETS) == 15

    def test_all_targets_have_prefix(self):
        from src.cron_compat import CRON_TARGETS
        for target in CRON_TARGETS:
            assert target.startswith("portfolio-lab-"), f"{target} missing prefix"

    def test_no_duplicate_targets(self):
        from src.cron_compat import CRON_TARGETS
        assert len(CRON_TARGETS) == len(set(CRON_TARGETS))

    def test_key_targets_present(self):
        from src.cron_compat import CRON_TARGETS
        required = [
            "portfolio-lab-data",
            "portfolio-lab-dashboard",
            "portfolio-lab-eval",
        ]
        for name in required:
            assert name in CRON_TARGETS, f"{name} missing from CRON_TARGETS"


class TestActiveBackend:
    """active_backend() function."""

    def test_returns_backend_value(self):
        with patch.dict(os.environ, {"CRON_BACKEND": "crontab"}, clear=True):
            import importlib
            import src.cron_compat as cc
            importlib.reload(cc)
            assert cc.active_backend() == "crontab"


class TestCronStatusPath:
    """cron_status_path() returns project-root-relative path."""

    def test_ends_with_correct_filename(self):
        from src.cron_compat import cron_status_path
        path = cron_status_path()
        assert path.endswith("data/cron_status.json")

    def test_returns_string(self):
        from src.cron_compat import cron_status_path
        path = cron_status_path()
        assert isinstance(path, str)

    def test_path_is_absolute(self):
        from src.cron_compat import cron_status_path
        path = cron_status_path()
        assert path.startswith("/")


class TestModuleIntegrity:
    """Basic import and attribute checks."""

    def test_import_does_not_raise(self):
        import src.cron_compat  # noqa

    def test_all_constants_defined(self):
        from src.cron_compat import BACKEND, IS_HERMES, IS_CRONTAB, IS_MANUAL
        assert isinstance(BACKEND, str)
        assert isinstance(IS_HERMES, bool)
        assert isinstance(IS_CRONTAB, bool)
        assert isinstance(IS_MANUAL, bool)

    def test_bools_are_mutually_exclusive_except_manual(self):
        from src.cron_compat import IS_HERMES, IS_CRONTAB, IS_MANUAL
        # hermes and crontab should not both be true
        assert not (IS_HERMES and IS_CRONTAB)


class TestCronExpectedDurations:
    """CRON_EXPECTED_DURATIONS mapping integrity."""

    def test_durations_is_dict(self):
        from src.cron_compat import CRON_EXPECTED_DURATIONS
        assert isinstance(CRON_EXPECTED_DURATIONS, dict)

    def test_all_targets_have_durations(self):
        from src.cron_compat import CRON_TARGETS, CRON_EXPECTED_DURATIONS
        for target in CRON_TARGETS:
            assert target in CRON_EXPECTED_DURATIONS, f"{target} missing from DURATIONS"

    def test_durations_are_positive(self):
        from src.cron_compat import CRON_EXPECTED_DURATIONS
        for target, duration in CRON_EXPECTED_DURATIONS.items():
            assert duration > 0, f"{target} has non-positive duration"

    def test_durations_are_reasonable(self):
        """All durations should be under 30 minutes (1800s)."""
        from src.cron_compat import CRON_EXPECTED_DURATIONS
        for target, duration in CRON_EXPECTED_DURATIONS.items():
            assert duration <= 1800, f"{target} duration {duration}s exceeds 30 min"


class TestCronGuardConfig:
    """CRON_GUARD_CONFIG structure."""

    def test_guard_config_is_dict(self):
        from src.cron_compat import CRON_GUARD_CONFIG
        assert isinstance(CRON_GUARD_CONFIG, dict)

    def test_has_required_keys(self):
        from src.cron_compat import CRON_GUARD_CONFIG
        required_keys = {"max_load", "default_timeout", "memory_mb", "lock_dir"}
        assert required_keys.issubset(set(CRON_GUARD_CONFIG.keys()))

    def test_memory_limit_at_least_1gb(self):
        from src.cron_compat import CRON_GUARD_CONFIG
        assert CRON_GUARD_CONFIG["memory_mb"] >= 1024

    def test_timeout_is_positive(self):
        from src.cron_compat import CRON_GUARD_CONFIG
        assert CRON_GUARD_CONFIG["default_timeout"] > 0

    def test_max_load_positive(self):
        from src.cron_compat import CRON_GUARD_CONFIG
        assert CRON_GUARD_CONFIG["max_load"] > 0
