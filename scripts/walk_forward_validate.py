#!/usr/bin/env python3
"""Deprecated compatibility wrapper for walk-forward validation.

The canonical walk-forward CLI is ``scripts/walk_forward_validation.py``.
This wrapper exists so old cron jobs or operator commands fail less
surprisingly while the duplicate implementation is retired.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from walk_forward_validation import main as canonical_main

CANONICAL_SCRIPT = "scripts/walk_forward_validation.py"

logger = logging.getLogger(__name__)


def main():
    """Delegate to the canonical walk-forward validation CLI."""
    logger.warning(
        "scripts/walk_forward_validate.py is deprecated; delegating to %s",
        CANONICAL_SCRIPT,
    )
    return canonical_main()


if __name__ == "__main__":
    main()
