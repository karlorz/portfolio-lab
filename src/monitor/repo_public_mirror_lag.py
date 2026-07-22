"""Repo public/data mirror lag probe (Batch EJ).

Operator ``PUBLIC_DATA_DIR`` (often ``/var/www/portfolio-lab/data``) is the
live SoT. Repo ``public/data`` is a derived static mirror used by offline/
deploy canaries. Historical friction: mirror lag 28–32/32 while cron stayed
green. This module wraps ``scripts/mirror_repo_public_data.lag_report`` for
compact health projection without shelling out to ``make``.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path
from typing import Any

from src.paths import DEFAULT_PUBLIC_DATA_DIR, PROJECT_ROOT, PUBLIC_DATA_DIR

logger = logging.getLogger(__name__)

__all__ = [
    "summarize_repo_public_mirror_lag",
    "load_mirror_script_module",
]

_MIRROR_MOD_NAME = "portfolio_lab_mirror_repo_public_data"


def load_mirror_script_module():
    """Load scripts/mirror_repo_public_data.py (register in sys.modules first)."""
    if _MIRROR_MOD_NAME in sys.modules:
        return sys.modules[_MIRROR_MOD_NAME]
    path = PROJECT_ROOT / "scripts" / "mirror_repo_public_data.py"
    if not path.is_file():
        raise FileNotFoundError(f"mirror script missing: {path}")
    spec = importlib.util.spec_from_file_location(_MIRROR_MOD_NAME, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load mirror script: {path}")
    mod = importlib.util.module_from_spec(spec)
    # Dataclass needs module in sys.modules before exec_module
    sys.modules[_MIRROR_MOD_NAME] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        sys.modules.pop(_MIRROR_MOD_NAME, None)
        raise
    return mod


def summarize_repo_public_mirror_lag(
    *,
    source_root: Path | str | None = None,
    dest_root: Path | str | None = None,
) -> dict[str, Any]:
    """Compare operator public SoT vs repo public/data; return lag summary.

    Returns::

        {
          "lagging_count": int,
          "total": int,
          "lagging_paths": list[str],
          "source": str,
          "dest": str,
          "ok": bool,
        }
    """
    src = Path(source_root) if source_root is not None else Path(PUBLIC_DATA_DIR)
    dest = (
        Path(dest_root)
        if dest_root is not None
        else Path(DEFAULT_PUBLIC_DATA_DIR)
    )
    # When PUBLIC_DATA_DIR already points at the repo mirror (dev shells),
    # comparing src==dest is always zero lag — still report honest zeros.
    try:
        mod = load_mirror_script_module()
        rows = mod.lag_report(src, dest)
    except Exception as exc:  # noqa: BLE001
        logger.warning("repo public mirror lag probe failed: %s", exc)
        return {
            "lagging_count": 0,
            "total": 0,
            "lagging_paths": [],
            "source": str(src),
            "dest": str(dest),
            "ok": False,
            "error": str(exc),
        }

    lagging_rows = [r for r in rows if isinstance(r, dict) and r.get("lagging")]
    paths = [
        str(r.get("path"))
        for r in lagging_rows
        if r.get("path") is not None
    ]
    return {
        "lagging_count": len(lagging_rows),
        "total": len(rows),
        "lagging_paths": paths[:12],
        "source": str(src),
        "dest": str(dest),
        "ok": True,
    }
