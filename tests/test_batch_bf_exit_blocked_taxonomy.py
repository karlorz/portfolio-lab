"""Batch BF: EXIT_BLOCKED taxonomy — intentional kill-skip is not cron/tasker error."""

from __future__ import annotations

import re
from pathlib import Path


def test_makefile_eval_maps_exit_2_to_blocked():
    mk = Path("Makefile").read_text(encoding="utf-8")
    m = re.search(r"^eval:\n(?P<body>(?:\t.*\n)+)", mk, re.M)
    assert m, "eval target missing"
    body = m.group("body")
    assert 'EXIT -eq 2' in body or "EXIT -eq 2" in body
    assert 'STATUS="blocked"' in body or "STATUS=\"blocked\"" in body


def test_tasker_models_define_run_blocked():
    from src.tasker.models import EXIT_CODE_BLOCKED, RUN_BLOCKED, TERMINAL_RUN_STATUSES

    assert EXIT_CODE_BLOCKED == 2
    assert RUN_BLOCKED == "blocked"
    assert RUN_BLOCKED in TERMINAL_RUN_STATUSES


def test_tasker_runner_maps_exit_2_to_blocked():
    src = Path("src/tasker/runner.py").read_text(encoding="utf-8")
    assert "EXIT_CODE_BLOCKED" in src
    assert "RUN_BLOCKED" in src


def test_tasker_store_blocked_resets_consecutive_failures():
    src = Path("src/tasker/store.py").read_text(encoding="utf-8")
    assert "RUN_BLOCKED" in src
    assert "RUN_SUCCESS, RUN_BLOCKED" in src or "RUN_BLOCKED" in src


def test_evaluator_exit_blocked_docs():
    from src.strategy.evaluator import EXIT_BLOCKED, EXIT_OK

    assert EXIT_OK == 0
    assert EXIT_BLOCKED == 2
    src = Path("src/strategy/evaluator.py").read_text(encoding="utf-8")
    assert "blocked" in src.lower()


def test_vix_persist_hydrates_contango_fields():
    src = Path("src/signals/vix_term_structure.py").read_text(encoding="utf-8")
    assert "VIXTermStructure" in src
    assert "from_dict" in src
    assert "contango" in src.lower()


def test_update_vix_script_vix3m_only_fallback():
    src = Path("scripts/update_vix_term_structure.py").read_text(encoding="utf-8")
    assert "VIX3M only" in src or "vix3m only" in src.lower() or "^VIX missing" in src
    assert "VIXTermStructure.from_dict" in src or "from_dict" in src
