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
    "restamp_mirror_lag_on_health_documents",
    "resolve_mirror_lag_for_consumer",
    "apply_lag_summary_to_health_doc",
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
    import json
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

    # One stamp for the whole batch so twin restamps stay byte-equal when
    # source/dest docs were equal before restamp (avoid re-introducing lag).
    restamped_at = datetime.now(timezone.utc).isoformat()
    restamp_policy = (
        "post soft-mirror nested SLI refresh from live probe "
        "(Batch FX; sticky false-critical fix)"
    )

    for raw in paths:
        path = Path(raw)
        if not path.is_file():
            result["skipped"].append(str(path.name))
            continue
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            result["errors"].append(f"{path.name}: read {exc}")
            continue
        if not isinstance(doc, dict):
            result["skipped"].append(str(path.name))
            continue

        # Skip docs that never carried the SLI (avoid inventing keys on unrelated JSON)
        has_sli = any(
            k in doc
            for k in (
                "repo_public_mirror_lagging_count",
                "repo_public_mirror_lag_status",
                "repo_public_mirror_lag",
            )
        )
        if not has_sli:
            result["skipped"].append(str(path.name))
            continue

        updated = apply_lag_summary_to_health_doc(doc, lag_summary)
        updated["mirror_lag_restamped_at"] = restamped_at
        updated["mirror_lag_restamp_policy"] = restamp_policy
        try:
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(
                json.dumps(updated, indent=2) + "\n", encoding="utf-8"
            )
            tmp.replace(path)
            result["restamped"].append(str(path.name))
        except OSError as exc:
            result["errors"].append(f"{path.name}: write {exc}")

    return result


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
