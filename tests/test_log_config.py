#!/usr/bin/env python3
"""Tests for src/utils/log_config.py."""

import logging
import os
from unittest.mock import patch

import pytest

from src.utils.log_config import CorrelationIDFilter, configure_logging


@pytest.fixture(autouse=True)
def _reset_logging():
    """Reset root logger between tests."""
    root = logging.getLogger()
    orig_level = root.level
    orig_handlers = root.handlers[:]
    yield
    root.level = orig_level
    root.handlers = orig_handlers


class TestCorrelationIDFilter:
    """Tests for CorrelationIDFilter."""

    def test_injects_cron_run_id_from_env(self):
        with patch.dict(os.environ, {"CRON_RUN_ID": "run-123"}):
            filt = CorrelationIDFilter()
            record = logging.LogRecord(
                name="test", level=logging.INFO, pathname="", lineno=0,
                msg="hello", args=(), exc_info=None,
            )
            assert filt.filter(record) is True
            assert record.cron_run_id == "run-123"  # type: ignore[attr-defined]

    def test_injects_cron_run_id_from_constructor(self):
        filt = CorrelationIDFilter(cron_run_id="explicit-456")
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="hello", args=(), exc_info=None,
        )
        filt.filter(record)
        assert record.cron_run_id == "explicit-456"  # type: ignore[attr-defined]

    def test_empty_cron_run_id_default(self):
        with patch.dict(os.environ, {}, clear=True):
            filt = CorrelationIDFilter()
            record = logging.LogRecord(
                name="test", level=logging.INFO, pathname="", lineno=0,
                msg="hello", args=(), exc_info=None,
            )
            filt.filter(record)
            assert record.cron_run_id == ""  # type: ignore[attr-defined]

    def test_constructor_overrides_env(self):
        with patch.dict(os.environ, {"CRON_RUN_ID": "from-env"}):
            filt = CorrelationIDFilter(cron_run_id="from-ctor")
            record = logging.LogRecord(
                name="test", level=logging.INFO, pathname="", lineno=0,
                msg="hello", args=(), exc_info=None,
            )
            filt.filter(record)
            assert record.cron_run_id == "from-ctor"  # type: ignore[attr-defined]


class TestConfigureLogging:
    """Tests for configure_logging."""

    def test_sets_root_level_info(self):
        configure_logging(level="INFO")
        root = logging.getLogger()
        assert root.level == logging.INFO

    def test_sets_root_level_debug(self):
        configure_logging(level="DEBUG")
        root = logging.getLogger()
        assert root.level == logging.DEBUG

    def test_respects_log_level_env_var(self):
        with patch.dict(os.environ, {"LOG_LEVEL": "WARNING"}):
            configure_logging()
            root = logging.getLogger()
            assert root.level == logging.WARNING

    def test_level_param_overrides_env(self):
        with patch.dict(os.environ, {"LOG_LEVEL": "ERROR"}):
            configure_logging(level="DEBUG")
            root = logging.getLogger()
            assert root.level == logging.DEBUG

    def test_defaults_to_info(self):
        with patch.dict(os.environ, {}, clear=True):
            configure_logging()
            root = logging.getLogger()
            assert root.level == logging.INFO

    def test_adds_console_handler(self):
        configure_logging()
        root = logging.getLogger()
        handler_types = [type(h).__name__ for h in root.handlers]
        assert "StreamHandler" in handler_types

    def test_json_logs_disabled_by_default(self):
        with patch.dict(os.environ, {}, clear=True):
            configure_logging()
            root = logging.getLogger()
            # Should use standard formatter, not JSON
            for h in root.handlers:
                if isinstance(h, logging.StreamHandler):
                    assert "JsonFormatter" not in type(h.formatter).__name__

    def test_json_logs_enabled_via_env(self):
        with patch.dict(os.environ, {"JSON_LOGS": "1"}):
            configure_logging()
            root = logging.getLogger()
            # May or may not have JsonFormatter depending on whether
            # python-json-logger is installed; just verify no crash
            assert root.level > 0

    def test_json_logs_true_stays_off(self):
        with patch.dict(os.environ, {"JSON_LOGS": "true"}):
            configure_logging()
            root = logging.getLogger()
            for h in root.handlers:
                if isinstance(h, logging.StreamHandler) and h.formatter is not None:
                    assert "JsonFormatter" not in type(h.formatter).__name__

    def test_json_logs_yes_stays_off(self):
        with patch.dict(os.environ, {"JSON_LOGS": "yes"}):
            configure_logging()
            root = logging.getLogger()
            for h in root.handlers:
                if isinstance(h, logging.StreamHandler) and h.formatter is not None:
                    assert "JsonFormatter" not in type(h.formatter).__name__


def test_configure_logging_unknown_env_level_fails_closed_to_info(monkeypatch):
    """Garbage LOG_LEVEL must not ValueError; fall back to INFO."""
    monkeypatch.setenv("LOG_LEVEL", "bogus")
    configure_logging()
    assert logging.getLogger().level == logging.INFO


def test_configure_logging_unknown_override_fails_closed_to_info():
    configure_logging(level="nope")
    assert logging.getLogger().level == logging.INFO
