"""Repo public/data mirror lag probe (Batch EJ).

Operator ``PUBLIC_DATA_DIR`` (often ``/var/www/portfolio-lab/data``) is the
live SoT. Repo ``public/data`` is a derived static mirror used by offline/
deploy canaries. Historical friction: mirror lag 28–32/32 while cron stayed
green. This module wraps ``scripts/mirror_repo_public_data.lag_report`` for
compact health projection without shelling out to ``make``.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import sys
from pathlib import Path
from typing import Any

from src.paths import DEFAULT_PUBLIC_DATA_DIR, PROJECT_ROOT, PUBLIC_DATA_DIR

logger = logging.getLogger(__name__)

__all__ = [
    "summarize_repo_public_mirror_lag",
    "load_mirror_script_module",
    "restamp_mirror_lag_on_health_documents",
    "resolve_mirror_lag_for_consumer",
    "apply_lag_summary_to_health_doc",
    "is_ephemeral_restamp_path",
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


def apply_lag_summary_to_health_doc(
    doc: dict[str, Any] | None,
    lag_summary: dict[str, Any] | None,
    *,
    elevate_status: bool = True,
) -> dict[str, Any]:
    """Project live lag summary onto a health / health_ops document (Batch FX).

    Soft-mirror copies nested ``repo_public_mirror_lag*`` stamps byte-for-byte.
    After the catalog equalizes, those nested fields can freeze a false
    critical until the next health :30. Restamping rewrites nested SLIs from
    a fresh probe without inventing a full health job.
    """
    from src.dashboard.generator import project_repo_public_mirror_lag_onto_health

    if not isinstance(doc, dict):
        doc = {}
    # Snapshot prior top-level status before soft-elevate so we can restore
    # when lag clears (sticky warning must not outlive the SLI).
    prior_status = doc.get("status")
    projected = project_repo_public_mirror_lag_onto_health(
        dict(doc), lag_summary
    )
    # Nested block for health_ops consumers (Batch EK shape)
    projected["repo_public_mirror_lag"] = {
        "lagging_count": projected.get("repo_public_mirror_lagging_count"),
        "total": projected.get("repo_public_mirror_total"),
        "status": projected.get("repo_public_mirror_lag_status"),
        "badge": projected.get("repo_public_mirror_lag_badge"),
        "paths": projected.get("repo_public_mirror_lagging_paths"),
        "source": projected.get("repo_public_mirror_source"),
        "dest": projected.get("repo_public_mirror_dest"),
    }
    if not elevate_status:
        # Nested SLI only — leave top-level status untouched
        projected["status"] = prior_status
    # Soft-elevate is elevate-only (project_repo_public_mirror_lag_onto_health).
    # Never demote top-level status on restamp; next full health job owns heal.
    return projected


def is_ephemeral_restamp_path(path: Path | str) -> bool:
    """True when *path* is a pytest/tmp fixture tree (must not restamp prod SSOT).

    Soft-mirror tests and unit fixtures live under ``/tmp/pytest-of-*`` or
    ``/tmp/pytest-*``. Writing production ``data/health.json`` from those
    runs poisons private lag SLIs with fixture stamp values (Batch HM DA).
    """
    text = str(path or "")
    if not text:
        return False
    # Normalize for substring checks (also catch resolved /private/tmp links)
    lowered = text.replace("\\", "/")
    if "/pytest-of-" in lowered:
        return True
    if "/tmp/pytest-" in lowered or lowered.startswith("/tmp/pytest-"):
        return True
    if "/var/folders/" in lowered and "pytest" in lowered:
        return True
    return False


def restamp_mirror_lag_on_health_documents(
    *,
    paths: list[Path | str] | None = None,
    lag_summary: dict[str, Any] | None = None,
    source_root: Path | str | None = None,
    dest_root: Path | str | None = None,
) -> dict[str, Any]:
    """Rewrite nested mirror-lag SLIs on health docs from a live probe (Batch FX EY).

    Returns::

        {
          "restamped": list[str],  # basenames or paths written
          "skipped": list[str],
          "errors": list[str],
          "lag_summary": dict,
        }
    """
    import os
    from datetime import datetime, timezone

    result: dict[str, Any] = {
        "restamped": [],
        "skipped": [],
        "errors": [],
        "lag_summary": {},
    }

    if lag_summary is None:
        lag_summary = summarize_repo_public_mirror_lag(
            source_root=source_root, dest_root=dest_root
        )
    result["lag_summary"] = lag_summary

    if not paths:
        return result

    # Batch HM: when any restamp path is ephemeral (pytest tmp), never touch
    # non-ephemeral production health docs in the same batch (private DATA_DIR
    # SSOT pollution). Allow pure-fixture batches (all ephemeral) for unit tests.
    resolved_paths = [Path(raw) for raw in paths]
    any_ephemeral = any(is_ephemeral_restamp_path(p) for p in resolved_paths)
    under_pytest = bool(os.environ.get("PYTEST_CURRENT_TEST"))

    # One stamp for the whole batch so twin restamps stay byte-equal when
    # source/dest docs were equal before restamp (avoid re-introducing lag).
    restamped_at = datetime.now(timezone.utc).isoformat()
    restamp_policy = (
        "post soft-mirror nested SLI refresh from live probe "
        "(Batch FX; sticky false-critical fix)"
    )

    for path in resolved_paths:
        if not path.is_file():
            result["skipped"].append(str(path.name))
            continue

        ephemeral = is_ephemeral_restamp_path(path)
        # Guard: pytest fixture trees must not restamp production SSOT, and a
        # mixed batch that includes pytest paths must skip non-ephemeral paths.
        if under_pytest and not ephemeral:
            result["skipped"].append(f"{path}:pytest-path-guard")
            logger.warning(
                "restamp skipped production path under pytest: %s", path
            )
            continue
        if any_ephemeral and not ephemeral:
            result["skipped"].append(f"{path}:mixed-batch-path-guard")
            logger.warning(
                "restamp skipped non-ephemeral path in mixed fixture batch: %s",
                path,
            )
            continue

        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            result["errors"].append(f"{path.name}: read {exc}")
            continue
        if not isinstance(doc, dict):
            result["skipped"].append(str(path.name))
            continue

        # Batch HO: signals.json carries lag under nested ``health`` (compact
        # surface for operators) while health.json / health_ops.json keep the
        # SLI at top-level. Detect both; never invent keys on unrelated JSON.
        nested_health = doc.get("health") if isinstance(doc.get("health"), dict) else None
        sli_keys = (
            "repo_public_mirror_lagging_count",
            "repo_public_mirror_lag_status",
            "repo_public_mirror_lag",
        )
        top_has_sli = any(k in doc for k in sli_keys)
        nested_has_sli = bool(
            nested_health is not None and any(k in nested_health for k in sli_keys)
        )
        if not top_has_sli and not nested_has_sli:
            result["skipped"].append(str(path.name))
            continue

        # Nested-only path (signals.json): restamp health block in place so
        # target_allocations and other authority keys stay byte-stable.
        if nested_has_sli and not top_has_sli:
            updated_nested = apply_lag_summary_to_health_doc(
                nested_health, lag_summary
            )
            updated_nested["mirror_lag_restamped_at"] = restamped_at
            updated_nested["mirror_lag_restamp_policy"] = restamp_policy
            doc["health"] = updated_nested
            try:
                _atomic_write_json_doc(path, doc)
                result["restamped"].append(str(path.name))
            except OSError as exc:
                result["errors"].append(f"{path.name}: write {exc}")
            continue

        updated = apply_lag_summary_to_health_doc(doc, lag_summary)
        updated["mirror_lag_restamped_at"] = restamped_at
        updated["mirror_lag_restamp_policy"] = restamp_policy
        # When both top-level and nested health carry the SLI, keep nested
        # honest too (signals dual-shape edge case).
        if nested_has_sli and isinstance(updated.get("health"), dict):
            nested_upd = apply_lag_summary_to_health_doc(
                updated["health"], lag_summary
            )
            nested_upd["mirror_lag_restamped_at"] = restamped_at
            nested_upd["mirror_lag_restamp_policy"] = restamp_policy
            updated["health"] = nested_upd
        try:
            _atomic_write_json_doc(path, updated)
            result["restamped"].append(str(path.name))
        except OSError as exc:
            result["errors"].append(f"{path.name}: write {exc}")

    return result


def _atomic_write_json_doc(path: Path, doc: dict[str, Any]) -> None:
    """Atomic JSON write with 0o644 so restamp never re-darkens Caddy.

    Reuses signal_authority._atomic_write_text (mkstemp + chmod) rather than
    bare write_text+replace, which inherits 0600 and broke HTTPS on signals.
    """
    text = json.dumps(doc, indent=2) + "\n"
    try:
        from src.monitor.signal_authority import _atomic_write_text

        _atomic_write_text(path, text, mode=0o644)
    except Exception:
        # Fallback if import fails in minimal envs — still chmod after replace.
        import os
        import tempfile

        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=str(path.parent),
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(text)
                fh.flush()
                os.fsync(fh.fileno())
            try:
                os.chmod(tmp_name, 0o644)
            except OSError:
                pass
            os.replace(tmp_name, path)
            try:
                os.chmod(path, 0o644)
            except OSError:
                pass
        except Exception:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise


def resolve_mirror_lag_for_consumer(
    *,
    stamp: dict[str, Any] | None,
    live: dict[str, Any] | None,
    warn_threshold: int = 1,
    critical_threshold: int = 10,
) -> dict[str, Any]:
    """Consumer honesty: max(live, stamp) lagging_count (Batch FX EW).

    Under-report polarity (stamp 0 / live 11) and sticky over-report (stamp 11 /
    live 0) both mislead operators. Prefer the worse lagging_count so green
    cannot hide behind a stale heal stamp, and restamp path clears sticky
    critical when live is clean (caller restamps docs; this is read-path).
    """
    stamp = stamp if isinstance(stamp, dict) else {}
    live = live if isinstance(live, dict) else {}

    def _count(d: dict[str, Any]) -> int:
        try:
            return int(d.get("lagging_count") or 0)
        except (TypeError, ValueError):
            return 0

    def _total(d: dict[str, Any]) -> int:
        try:
            return int(d.get("total") or 0)
        except (TypeError, ValueError):
            return 0

    live_n = _count(live)
    stamp_n = _count(stamp)
    # max for under-report defense; if live is authoritative ok after heal
    # and stamp is higher, still surface max unless live ok and caller restamped.
    # Policy: max(live, stamp) always for consumer read honesty.
    chosen_n = max(live_n, stamp_n)
    source = "live" if live_n >= stamp_n else "stamp"
    total = max(_total(live), _total(stamp))

    if chosen_n >= int(critical_threshold):
        status = "critical"
    elif chosen_n >= int(warn_threshold):
        status = "lagging"
    elif total > 0:
        status = "ok"
    else:
        status = "unknown"

    paths: list[str] = []
    if source == "live":
        raw_paths = live.get("lagging_paths") or live.get("paths") or []
    else:
        raw_paths = stamp.get("lagging_paths") or stamp.get("paths") or []
    if isinstance(raw_paths, list):
        paths = [str(p) for p in raw_paths[:12]]

    return {
        "lagging_count": chosen_n,
        "total": total,
        "lagging_paths": paths,
        "source_of_truth": source,
        "repo_public_mirror_lag_status": status,
        "live_lagging_count": live_n,
        "stamp_lagging_count": stamp_n,
        "policy": "max(live,stamp) lagging_count for consumer honesty (Batch FX EW)",
    }
