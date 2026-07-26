"""Shared helpers for read-only Makefile contract tests."""

from __future__ import annotations

import re


def makefile_recipe(makefile: str, target: str) -> str:
    """Return a target's tab-indented recipe without executing Make."""
    match = re.search(
        rf"^{re.escape(target)}:[^\n]*\n",
        makefile,
        flags=re.MULTILINE,
    )
    if match is None:
        raise AssertionError(f"Makefile target not found: {target}")

    recipe: list[str] = []
    for line in makefile[match.end():].splitlines(keepends=True):
        if line.startswith("\t") or not line.strip():
            recipe.append(line)
        else:
            break
    return "".join(recipe)
