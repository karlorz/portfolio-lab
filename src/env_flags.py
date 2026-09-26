"""Fail-closed ENABLE/ALLOW-style environment flags.

Only the exact string ``1`` turns a flag on. Truthy strings such as
``true`` / ``yes`` / ``on`` stay off.

``ALPACA_PAPER`` is intentionally fail-*open* to paper (see
``env_paper_unless_false``): live trading requires an explicit false/0/no.
"""

from __future__ import annotations

import os
from collections.abc import Mapping


def env_literal_one(
    name: str,
    *,
    default: str = "0",
    env: Mapping[str, str] | None = None,
) -> bool:
    """Return True only when ``env[name]`` (or os.environ) is exactly ``"1"``."""
    source: Mapping[str, str] = os.environ if env is None else env
    return str(source.get(name, default)) == "1"


def env_paper_unless_false(
    name: str = "ALPACA_PAPER",
    *,
    default: str = "true",
    env: Mapping[str, str] | None = None,
) -> bool:
    """Return True (paper) unless the value is explicitly false/0/no.

    Case-insensitive. Used for ``ALPACA_PAPER`` so ``1`` / ``yes`` / unset
    stay paper; only ``false`` / ``0`` / ``no`` select live.
    """
    source: Mapping[str, str] = os.environ if env is None else env
    return str(source.get(name, default)).lower() not in ("false", "0", "no")
