"""Public-data projection and runtime provenance policy.

Private monitor artifacts are intentionally richer than the JSON served by the
dashboard.  This module is the one boundary where private diagnostic paths are
converted to stable logical references before a public write.  Keeping the
policy here avoids a producer-by-producer collection of path redaction rules.

The projection is deliberately conservative:

* business values are copied unchanged;
* only path-bearing strings (or strings containing an internal path) are
  rewritten; and
* runtime provenance is additive and is emitted for known production trees.

The consistency auditor imports :func:`find_public_internal_paths` so a public
tree cannot silently regress after a new producer is added.
"""

from __future__ import annotations

import copy
import math
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from src.paths import (
    DATA_DIR,
    DEFAULT_LIVE_PUBLIC_DATA_DIR,
    DEFAULT_PUBLIC_DATA_DIR,
    PROJECT_ROOT,
)

PUBLIC_PROJECTION_SCHEMA_VERSION = "public-logical-reference/v1"
RUNTIME_PROVENANCE_SCHEMA_VERSION = "runtime-provenance/v1"

# Metadata that may legitimately differ between private and public planes.
# Business fields must remain identical after path projection.  Keep this
# allowlist explicit so a producer cannot hide a semantic drift by adding a
# vaguely named field to the public payload.
PUBLIC_BUSINESS_METADATA_KEYS = frozenset(
    {
        "artifact_id",
        "plane",
        "generated_at",
        "timestamp",
        "updated_at",
        "created_at",
        "checked_at",
        "last_updated",
        "reconciled_at",
        "ssot_reconciled_at",
        "ssot_reconcile_source",
        "generator_git_sha",
        "generator_git_sha_status",
        "last_full_generator_git_sha",
        "generator_git_sha_reason",
        "patch_source",
        "content_patch_source",
        "runtime_provenance",
        "provenance_completeness",
        "schema_version",
        "private_path",
        "public_path",
        "private_mtime",
        "public_mtime",
        "private_content_hash",
        "public_content_hash",
        "dual_write_lag_seconds",
        "dual_write_lag_stale",
        "dual_write_lag_threshold_seconds",
        "dual_write_lag_unit",
        "repo_public_mirror_source",
        "repo_public_mirror_dest",
        "repo_public_mirror_lag",
        "repo_public_mirror_lagging_count",
        "repo_public_mirror_total",
        "mirror_lag_restamped_at",
        "private_health_report",
    }
)

# These are raw/market blobs, not operator artifacts.  Adding an object-level
# provenance block to them would change their established consumer shape.
RUNTIME_PROVENANCE_EXCLUDED_FILES = frozenset(
    {
        "prices.json",
        "prices_compact.json",
        "historical.json",
        "yields.json",
        "vix_term_structure.json",
        "vix_term_structure_history.json",
    }
)

_URL_PREFIXES = ("http://", "https://", "ws://", "wss://", "mailto:")
_WINDOWS_ABSOLUTE_RE = re.compile(r"(?<![A-Za-z0-9])(?:[A-Za-z]:[\\/])[^\s\"'`,;)}\]]+")
_POSIX_ABSOLUTE_RE = re.compile(
    r"(?<![A-Za-z0-9])/(?:root|home|Users|private|tmp|var|opt|srv|mnt|workspace|etc)/[^\s\"'`,;)}\]]*"
)
_KNOWN_ROOTS_CACHE: tuple[tuple[str, Path], ...] | None = None


def _resolved(path: Path | str) -> Path:
    try:
        return Path(path).expanduser().resolve()
    except OSError:
        return Path(path).expanduser()


def _root_candidates() -> tuple[tuple[str, Path], ...]:
    """Return logical-root mappings, including runtime env overrides."""
    global _KNOWN_ROOTS_CACHE
    # Runtime path env vars can be changed by tasker or tests after imports, so
    # rebuild when the relevant values differ instead of relying only on the
    # module-level paths from src.paths.
    env_public = os.environ.get("PUBLIC_DATA_DIR")
    env_live = os.environ.get("PORTFOLIO_LAB_LIVE_PUBLIC_DATA_DIR")
    cache_key = (env_public or "", env_live or "")
    if _KNOWN_ROOTS_CACHE is not None and getattr(_root_candidates, "_cache_key", None) == cache_key:
        return _KNOWN_ROOTS_CACHE

    roots: list[tuple[str, Path]] = []

    def add(label: str, value: Path | str | None) -> None:
        if value is None:
            return
        resolved = _resolved(value)
        if any(existing == resolved for _, existing in roots):
            return
        roots.append((label, resolved))

    # Most-specific roots must precede their parent project/data roots.
    add("public", env_public)
    add("public", DEFAULT_LIVE_PUBLIC_DATA_DIR)
    add("public", env_live)
    add("public", DEFAULT_PUBLIC_DATA_DIR)
    add("private", DATA_DIR)
    add("repo", PROJECT_ROOT)
    _KNOWN_ROOTS_CACHE = tuple(roots)
    setattr(_root_candidates, "_cache_key", cache_key)
    return _KNOWN_ROOTS_CACHE


def _as_path_text(value: str) -> str:
    text = str(value).strip()
    if text.startswith("file://"):
        text = text[7:]
    return text


def _is_url(value: str) -> bool:
    return value.strip().lower().startswith(_URL_PREFIXES)


def _absolute_path_match(value: str) -> str | None:
    """Return an absolute-path token when *value* contains one."""
    if _is_url(value):
        return None
    text = _as_path_text(value)
    if text.startswith("/"):
        return text
    windows = _WINDOWS_ABSOLUTE_RE.search(text)
    if windows:
        return windows.group(0)
    posix = _POSIX_ABSOLUTE_RE.search(text)
    return posix.group(0) if posix else None


def _relative_to(path: Path, root: Path) -> str | None:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return None


def _logical_reference_for_path(path_text: str) -> str:
    raw = _as_path_text(path_text).rstrip(".,;:)")
    path = _resolved(raw)
    normalized = str(path).replace("\\", "/")

    # Hermes is a scheduler implementation detail.  Keep its identity, but
    # expose only a stable logical scheduler reference.
    hermes_marker = "/.hermes/"
    if hermes_marker in normalized:
        suffix = normalized.split(hermes_marker, 1)[1].lstrip("/")
        return f"scheduler/hermes/{suffix or 'jobs.json'}"

    for label, root in _root_candidates():
        relative = _relative_to(path, root)
        if relative is None:
            continue
        if label in {"public", "private"}:
            if relative == "tasker_logs" or relative.startswith("tasker_logs/"):
                suffix = relative.removeprefix("tasker_logs/")
                return f"tasker/logs/{suffix or 'latest.log'}"
            return f"data/{relative}"
        return f"repo/{relative}"

    # Unknown absolute paths must not survive merely because a new producer
    # chose a different host directory.  The basename is useful to an
    # operator, while the host/user/parent directory is not.
    basename = Path(raw).name or "path"
    return f"internal/{basename}"


def logical_reference(value: str | Path, *, fallback: str | None = None) -> str:
    """Convert an absolute diagnostic path to a stable public reference.

    Relative values and ordinary business strings are returned unchanged.
    ``fallback`` is used only when an empty path-like value is supplied.
    """
    text = str(value)
    token = _absolute_path_match(text)
    if token is None:
        return fallback if not text.strip() and fallback is not None else text
    if text.strip() == token or _as_path_text(text).rstrip(".,;:)") == token:
        return _logical_reference_for_path(token)
    return _replace_embedded_paths(text)


def _replace_embedded_paths(value: str) -> str:
    if _is_url(value):
        return value

    def replace(match: re.Match[str]) -> str:
        token = match.group(0)
        trailing = ""
        while token and token[-1] in ".,;:)":
            trailing = token[-1] + trailing
            token = token[:-1]
        return _logical_reference_for_path(token) + trailing

    # Known roots may occur in a key prefix (for example the hash cache uses
    # ``files.<absolute-path>``).  The generic pattern handles those as well.
    return _WINDOWS_ABSOLUTE_RE.sub(replace, _POSIX_ABSOLUTE_RE.sub(replace, value))


_PATH_KEY_TOKENS = (
    "path",
    "source",
    "dest",
    "artifact",
    "output",
    "log",
    "directory",
    "file",
    "history_source",
    "evidence_source",
    "private_health_report",
    "repo_public_mirror",
    "patch_source",
)


def _is_path_key(key: str) -> bool:
    lowered = key.lower()
    return any(token in lowered for token in _PATH_KEY_TOKENS)


def _project_value(value: Any, *, key: str = "") -> Any:
    if isinstance(value, Mapping):
        projected: dict[Any, Any] = {}
        for raw_key, raw_value in value.items():
            projected_key: Any = raw_key
            if isinstance(raw_key, str):
                projected_key = _replace_embedded_paths(raw_key)
            projected[projected_key] = _project_value(
                raw_value,
                key=raw_key if isinstance(raw_key, str) else "",
            )
        return projected
    if isinstance(value, list):
        return [_project_value(item, key=key) for item in value]
    if isinstance(value, tuple):
        return [_project_value(item, key=key) for item in value]
    if isinstance(value, Path):
        return logical_reference(value)
    if isinstance(value, str):
        # Path-bearing fields receive the same treatment even for a string
        # that does not begin with '/', while free-form messages are only
        # changed when they actually contain an absolute internal path.
        if _is_path_key(key):
            return logical_reference(value)
        return _replace_embedded_paths(value)
    return value


def project_public_paths(payload: Any) -> Any:
    """Return a deep public copy with internal absolute paths projected."""
    return _project_value(copy.deepcopy(payload))


def project_public_business_values(payload: Any) -> Any:
    """Return the business portion of a payload for plane-equivalence checks.

    Public/private payloads may differ in disclosure metadata, timestamps,
    schema labels, runtime provenance, and logicalized path references.  This
    helper removes only the explicit metadata contract after applying the
    normal public path projection; it intentionally leaves allocation,
    metrics, status, history, and other business values untouched.
    """
    projected = project_public_paths(payload)

    def strip(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {
                key: strip(child)
                for key, child in value.items()
                if key not in PUBLIC_BUSINESS_METADATA_KEYS
                and not (isinstance(key, str) and key.endswith("_at"))
            }
        if isinstance(value, list):
            return [strip(child) for child in value]
        if isinstance(value, tuple):
            return [strip(child) for child in value]
        return value

    return strip(projected)


def public_business_values_equal(private_payload: Any, public_payload: Any) -> bool:
    """Compare private/public payloads after the approved metadata projection.

    JSON generated by older producers may contain NaN where the public
    serializer emits null or another JSON loader preserves NaN.  Treat two
    non-finite numeric sentinels as equivalent; all finite values and types
    must match exactly.
    """
    private_values = project_public_business_values(private_payload)
    public_values = project_public_business_values(public_payload)

    def equal(left: Any, right: Any) -> bool:
        if isinstance(left, float) and isinstance(right, float):
            if math.isnan(left) and math.isnan(right):
                return True
        if type(left) is not type(right):
            return False
        if isinstance(left, Mapping):
            return left.keys() == right.keys() and all(
                equal(left[key], right[key]) for key in left
            )
        if isinstance(left, list):
            return len(left) == len(right) and all(
                equal(left_item, right_item)
                for left_item, right_item in zip(left, right)
            )
        return left == right

    return equal(private_values, public_values)


def _is_known_runtime_path(path: Path | str | None) -> bool:
    if path is None:
        return False
    candidate = _resolved(path)
    for label, root in _root_candidates():
        if label in {"public", "private"} and _relative_to(candidate, root) is not None:
            return True
    return False


def is_public_output_path(path: Path | str | None) -> bool:
    """Return whether *path* belongs to a configured public-data tree."""
    if path is None:
        return False
    candidate = _resolved(path)
    return any(
        label == "public" and _relative_to(candidate, root) is not None
        for label, root in _root_candidates()
    )


def _artifact_id(path: Path | str | None) -> str:
    if path is None:
        return "artifact"
    candidate = _resolved(path)
    for label, root in _root_candidates():
        relative = _relative_to(candidate, root)
        if relative is not None and label in {"public", "private"}:
            return relative
    return candidate.name or "artifact"


def _current_git_sha() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(PROJECT_ROOT),
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    sha = completed.stdout.strip()
    return sha or None


def _first_nonempty(payload: Mapping[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        value = payload.get(key)
        if value not in (None, ""):
            return value
    return None


def _runtime_metadata(
    payload: Mapping[str, Any],
    output_path: Path | str | None,
    *,
    plane: str | None = None,
    patch_source: str | None = None,
) -> dict[str, Any]:
    existing_runtime = payload.get("runtime_provenance")
    existing = existing_runtime if isinstance(existing_runtime, Mapping) else {}
    generated_at = _first_nonempty(
        payload,
        ("generated_at", "timestamp", "updated_at", "content_patched_at"),
    )
    if generated_at is None:
        generated_at = existing.get("generated_at")
    if generated_at is None:
        generated_at = datetime.now(timezone.utc).isoformat()

    status = _first_nonempty(
        payload,
        ("generator_git_sha_status",),
    ) or existing.get("generator_git_sha_status")
    sha = _first_nonempty(payload, ("generator_git_sha",))
    if sha is None:
        sha = existing.get("generator_git_sha")
    if status is None:
        status = "full_generate" if sha else "unavailable"
    status = str(status)
    if status in {"partial_patch", "partial"}:
        # A partial patch is deliberately not attributed to the current code
        # tip.  Preserve a prior full SHA only as historical context.
        sha = None
    elif not sha and status in {"full", "full_generate"}:
        sha = _current_git_sha()
        if not sha:
            status = "unavailable"

    last_full = _first_nonempty(payload, ("last_full_generator_git_sha",))
    if last_full is None:
        last_full = existing.get("last_full_generator_git_sha")
    if last_full is None and status in {"full", "full_generate"}:
        last_full = sha

    source = patch_source or _first_nonempty(
        payload,
        ("patch_source", "content_patch_source", "generator_git_sha_reason"),
    )
    if source is None:
        source = existing.get("patch_source")

    return {
        "schema_version": RUNTIME_PROVENANCE_SCHEMA_VERSION,
        "artifact_id": _artifact_id(output_path),
        "plane": plane or ("public" if output_path is not None else "unknown"),
        "generated_at": str(generated_at),
        "generator_git_sha": sha,
        "generator_git_sha_status": status,
        "last_full_generator_git_sha": last_full,
        "patch_source": source,
    }


def prepare_payload_for_write(
    payload: Any,
    output_path: Path | str | None,
    *,
    public: bool = False,
    add_runtime_provenance: bool | None = None,
    plane: str | None = None,
    patch_source: str | None = None,
) -> Any:
    """Prepare one payload for a private or public JSON write.

    ``public=True`` applies path projection.  Runtime metadata is enabled by
    default only for known production roots; callers may explicitly enable it
    for a hermetic fixture or disable it for raw market blobs.
    """
    projected = project_public_paths(payload) if public else copy.deepcopy(payload)
    if not isinstance(projected, Mapping):
        return projected

    if add_runtime_provenance is None:
        add_runtime_provenance = _is_known_runtime_path(output_path)
    filename = Path(output_path).name if output_path is not None else ""
    if filename in RUNTIME_PROVENANCE_EXCLUDED_FILES:
        add_runtime_provenance = False
    if not add_runtime_provenance:
        return projected

    out = dict(projected)
    runtime = _runtime_metadata(
        out,
        output_path,
        plane=plane or ("public" if public else "private"),
        patch_source=patch_source,
    )
    # Keep the fields top-level for simple consumers and nest the complete
    # block for future schema evolution.
    out["artifact_id"] = runtime["artifact_id"]
    out["plane"] = runtime["plane"]
    out["generated_at"] = runtime["generated_at"]
    out["generator_git_sha"] = runtime["generator_git_sha"]
    out["generator_git_sha_status"] = runtime["generator_git_sha_status"]
    out["last_full_generator_git_sha"] = runtime["last_full_generator_git_sha"]
    if runtime["patch_source"] is not None:
        out["patch_source"] = runtime["patch_source"]
    out["runtime_provenance"] = runtime
    return out


def _contains_internal_path(value: str) -> bool:
    if _is_url(value):
        return False
    if _absolute_path_match(value) is not None:
        token = _absolute_path_match(value) or ""
        # Public URL routes such as /data/foo are not host paths.  Known roots
        # and conventional private roots are the fail-closed set.
        if token.startswith("/data/") and not any(
            marker in token for marker in ("/root/", "/var/", "/home/", "/tmp/")
        ):
            return False
        return True
    lowered = value.replace("\\", "/")
    return any(
        marker in lowered
        for marker in (
            str(_resolved(PROJECT_ROOT)).replace("\\", "/"),
            str(_resolved(DATA_DIR)).replace("\\", "/"),
            "/var/www/portfolio-lab/data/",
            "/root/.hermes/",
        )
    )


def find_public_internal_paths(payload: Any) -> list[tuple[str, str]]:
    """Return JSON-pointer/value pairs that still contain internal paths."""
    offenders: list[tuple[str, str]] = []

    def walk(value: Any, pointer: str) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                key_text = str(key)
                key_pointer = f"{pointer}/{key_text.replace('~', '~0').replace('/', '~1')}"
                if _contains_internal_path(key_text):
                    offenders.append((key_pointer, key_text))
                walk(child, key_pointer)
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                walk(child, f"{pointer}/{index}")
        elif isinstance(value, str) and _contains_internal_path(value):
            offenders.append((pointer or "/", value))

    walk(payload, "")
    return offenders


__all__ = [
    "PUBLIC_BUSINESS_METADATA_KEYS",
    "PUBLIC_PROJECTION_SCHEMA_VERSION",
    "RUNTIME_PROVENANCE_SCHEMA_VERSION",
    "find_public_internal_paths",
    "logical_reference",
    "prepare_payload_for_write",
    "project_public_business_values",
    "project_public_paths",
    "public_business_values_equal",
    "is_public_output_path",
]
