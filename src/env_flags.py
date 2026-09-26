"""Fail-closed ENABLE/ALLOW-style environment flags.

Only the exact string ``1`` turns a flag on. Truthy strings such as
``true`` / ``yes`` / ``on`` stay off.
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
