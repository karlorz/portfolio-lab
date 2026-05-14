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

    def test_eight_default_targets(self):
        from src.cron_compat import CRON_TARGETS
        assert len(CRON_TARGETS) == 8

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
            "portfolio-lab-health",
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
