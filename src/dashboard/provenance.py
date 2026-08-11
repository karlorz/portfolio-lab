"""Provenance / dual-write helpers extracted from ``src.dashboard.generator``.

Git-sha honesty stamps, source-manifest rows, canonical content hashing,
dual-write provenance finalization and the dist-data mirror helpers moved
here by Item 9 (2026-08-12). ``generator.py`` re-exports every name below to
preserve its public attribute surface (lazy importers in
public_data_index / overlay_dashboard / experiment_registry / rebalance_health
/ daily_brief / unified_dashboard resolve these via the generator module).
"""

import json
import re
import shutil
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from src.paths import DATA_DIR

# Dist mirror contract files (moved with _mirror_public_data_contract_files_to_dist)
PUBLIC_DATA_DIST_MIRROR_FILES = ("source_manifest.json", "index.json", "health.json")

def _apply_partial_patch_git_sha_honesty(
    payload: Dict[str, Any],
    *,
    patch_source: str,
) -> None:
    """Clear sticky full-generation git sha on partial section rewrites.

    Partial writers advance ``generated_at`` / ``content_patched_at`` but leave
    ``generator_git_sha`` from the last full dashboard run. Operators then
    attribute a partial patch to a wrong code tip. Keep the prior full-run sha
    under ``last_full_generator_git_sha`` for lag forensics; null the live stamp
    and disclose ``generator_git_sha_status=partial_patch``.
    """
    prior = payload.get("generator_git_sha")
    if prior is not None and prior != "":
        payload.setdefault("last_full_generator_git_sha", prior)
    payload["generator_git_sha"] = None
    payload["generator_git_sha_status"] = "partial_patch"
    payload["generator_git_sha_reason"] = (
        f"cleared by partial rewrite ({patch_source}); "
        "not a full dashboard generation"
    )


def _enrich_duration_allocation_provenance(
    payload: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Ensure duration_allocation never publishes bare weights without role/unit.

    Live partial patches and legacy consumers have left only ``{tlt,ief,shy,bil}``
    which looks like a live sleeve without advisory disclosure.
    """
    if not isinstance(payload, dict) or not payload:
        return payload
    out = dict(payload)
    # Collect weight symbols if nested under weights or flat
    weights = out.get("weights")
    if not isinstance(weights, dict):
        weights = {
            k: out[k]
            for k in ("tlt", "ief", "shy", "bil")
            if isinstance(out.get(k), (int, float))
        }
        if weights:
            out["weights"] = weights
    if weights:
        try:
            out["sum"] = round(sum(float(v) for v in weights.values()), 4)
        except (TypeError, ValueError):
            pass
    out.setdefault("unit", "portfolio_weight_fraction")
    out.setdefault("live_authoritative", False)
    out.setdefault("role", "advisory_sleeve")
    out.setdefault(
        "description",
        "Bond duration sleeve from 2s10s regime table; "
        "not target_allocations / order-routing authority",
    )
    out.setdefault("source", "yield_curve_regime_table")
    return out


def _source_manifest_row_for(public_dir: Path, artifact_name: str) -> dict[str, Any] | None:
    """Return the compact source-manifest row for a public data artifact."""
    manifest_path = public_dir / "source_manifest.json"
    try:
        with open(manifest_path) as f:
            manifest = json.load(f)
    except (OSError, json.JSONDecodeError, TypeError):
        return None

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        return None

    for row in artifacts:
        if not isinstance(row, dict):
            continue
        candidates = {
            row.get("artifact"),
            row.get("filename"),
            row.get("path"),
        }
        if artifact_name in candidates:
            return row
    return None


def _yield_source_provenance(public_dir: Path) -> dict[str, Any]:
    """Map yields source-manifest metadata into the yield curve payload."""
    row = _source_manifest_row_for(public_dir, "yields.json")
    if row is None:
        return {}
    return {
        "source_mode": row.get("source_mode"),
        "source_status": row.get("status"),
        "source_reason": row.get("failure_reason") or row.get("reason"),
        "source_provider": row.get("provider"),
        "source_generated_at": row.get("generated_at"),
        "source_latest_observation": row.get("latest_observation"),
    }


def _first_known_value(*values: Any, default: str = "unknown") -> Any:
    """Return the first non-empty metadata value, treating 'unknown' as absent."""
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and value.lower() in {"", "unknown"}:
            continue
        return value
    return default

# Common exception types caught when signal generators fail.
# ValueError/TypeError indicate likely bugs — callers should log these at
# error level. ImportError/AttributeError indicate missing deps — warning.
SIGNAL_EXCEPTIONS = (
    ImportError, AttributeError, KeyError,
    ValueError, TypeError, RuntimeError, OSError,
)

# Lighter exception tuple for monitoring/utility modules that don't
# touch external data structures (no AttributeError/KeyError risk).
MONITOR_EXCEPTIONS = (ImportError, ValueError, OSError, RuntimeError)

# Exceptions that indicate likely bugs rather than missing dependencies.
_BUG_EXCEPTIONS = (ValueError, TypeError)


def _attach_signal_metadata(output: Dict, *, generated_at: str | None = None) -> Dict:
    """Attach dashboard-level generation timestamps to a signals payload."""
    timestamp = generated_at or datetime.now(timezone.utc).isoformat()
    enriched = dict(output)
    enriched["generated_at"] = timestamp
    enriched.setdefault("timestamp", timestamp)
    return enriched


def _generator_git_sha_short() -> str | None:
    """Short HEAD for operator lag detection (code vs projected artifact)."""
    try:
        from src.monitor.decision_registry import _git_sha_short

        return _git_sha_short()
    except Exception:
        return None


def _stamp_generator_git_sha(
    payload: Dict[str, Any],
    *,
    status: str = "full_generate",
) -> Dict[str, Any]:
    """Attach generator_git_sha when available (stats/analytics/graduation/overlay).

    Batch BJ residual honesty (SLSA-style prior identity retention):
    when the new tip differs from the previous full stamp, archive the prior
    under ``last_full_generator_git_sha`` for lag forensics. Never clear an
    existing last_full trail on full_generate.
    """
    out = dict(payload)
    # Defer to the generator module's attribute at call time: tests patch
    # ``src.dashboard.generator._generator_git_sha_short`` (39 refs); module
    # -level aliasing here would silently break those patch targets.
    from src.dashboard import generator as _generator

    sha = _generator._generator_git_sha_short()
    if sha:
        prior = out.get("generator_git_sha")
        prior_s = str(prior).strip() if prior not in (None, "") else ""
        if prior_s and prior_s != sha:
            out["last_full_generator_git_sha"] = prior_s
        # Never drop an existing last_full trail when re-stamping same tip
        # or when prior was already cleared by a partial_patch path.
        out["generator_git_sha"] = sha
        out["generator_git_sha_status"] = status
        # Batch CB: full_generate with empty last_full gets self-trail so lag
        # forensics always has a non-null full stamp after a complete generate.
        if status == "full_generate":
            existing_last = out.get("last_full_generator_git_sha")
            if existing_last in (None, ""):
                out["last_full_generator_git_sha"] = sha
    return out


def _canonical_file_content_hash(path: Path) -> str | None:
    """SHA-256 of file bytes after stripping a single trailing newline.

    Dual-write trees often differ only by final-newline policy; content hash
    treats those as identical so sticky lag is not reported when payloads match.
    """
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    # Normalize trailing newlines only (do not alter interior whitespace)
    while raw.endswith(b"\n"):
        raw = raw[:-1]
    import hashlib

    return hashlib.sha256(raw).hexdigest()


def _attach_dual_write_provenance(
    payload: Dict[str, Any],
    *,
    private_path: str | Path | None = None,
    public_path: str | Path | None = None,
    dual_write_attempted: bool = False,
    dual_write_ok: bool | None = None,
    paths_identical: bool | None = None,
    note: str | None = None,
    lag_threshold_seconds: float = 120.0,
) -> Dict[str, Any]:
    """Attach dual-write completeness block for operator lag / split-brain forensics.

    Does not alter live authority. Complements generator_git_sha stamps (Batch AR).

    When both private and public paths exist and differ, sets
    ``dual_write_lag_seconds`` = public_mtime - private_mtime (negative means
    public is older than private — typical split-brain lag). Advisory only.

    Batch dual-write: if path resolves differ but **canonical content hashes**
    match (trailing-newline-normalized), clear sticky lag_stale and set lag
    seconds to 0. ``paths_identical`` remains path-resolve identity (caller
    flag or resolve equality); content match is ``content_hash_identical``.
    """
    out = dict(payload)
    sha = out.get("generator_git_sha")
    priv = Path(private_path) if private_path is not None else None
    pub = Path(public_path) if public_path is not None else None

    lag_seconds: float | None = None
    private_mtime: float | None = None
    public_mtime: float | None = None
    lag_stale = False
    content_hash_identical: bool | None = None
    private_content_hash: str | None = None
    public_content_hash: str | None = None
    # Path identity: caller flag, else compare resolves when both exist
    if paths_identical is None and priv is not None and pub is not None:
        try:
            paths_identical = priv.resolve() == pub.resolve()
        except OSError:
            paths_identical = False
    path_identical = bool(paths_identical)
    try:
        if priv is not None and priv.is_file():
            private_mtime = float(priv.stat().st_mtime)
            private_content_hash = _canonical_file_content_hash(priv)
        if pub is not None and pub.is_file():
            public_mtime = float(pub.stat().st_mtime)
            public_content_hash = _canonical_file_content_hash(pub)
        if private_content_hash is not None and public_content_hash is not None:
            content_hash_identical = private_content_hash == public_content_hash
        # Lag is irrelevant when paths are the same OR content hashes match
        lag_cleared = path_identical or bool(content_hash_identical)
        if (
            private_mtime is not None
            and public_mtime is not None
            and not lag_cleared
        ):
            # public - private: negative => public behind private (lag)
            lag_seconds = round(public_mtime - private_mtime, 3)
            # Stale if public is older than private by more than threshold
            if lag_seconds < -abs(float(lag_threshold_seconds)):
                lag_stale = True
        elif lag_cleared and private_mtime is not None and public_mtime is not None:
            lag_seconds = 0.0
            lag_stale = False
    except OSError:
        pass

    block: Dict[str, Any] = {
        "generator_git_sha_present": bool(sha),
        "dual_write_attempted": bool(dual_write_attempted),
        "dual_write_ok": dual_write_ok,
        "private_path": str(priv) if priv is not None else None,
        "public_path": str(pub) if pub is not None else None,
        "paths_identical": path_identical,
        "content_hash_identical": content_hash_identical,
        "private_content_hash": private_content_hash,
        "public_content_hash": public_content_hash,
        "dual_write_lag_seconds": lag_seconds,
        "dual_write_lag_unit": "seconds_public_mtime_minus_private",
        "dual_write_lag_stale": lag_stale,
        "dual_write_lag_threshold_seconds": float(lag_threshold_seconds),
        "private_mtime": private_mtime,
        "public_mtime": public_mtime,
        "disclosure": (
            "Dual-write provenance is advisory for split-brain detection; "
            "private DATA_DIR remains the producer SSOT when paths differ. "
            "Lag uses filesystem mtimes (public - private); negative means "
            "public is older than private. Content-hash equality "
            "(trailing-newline-normalized) clears sticky lag when payloads match."
        ),
    }
    if note:
        block["note"] = note
    out["provenance_completeness"] = block
    return out


def finalize_dual_write_provenance_after_sync(
    payload: Dict[str, Any],
    *,
    private_path: str | Path,
    public_path: str | Path,
    dual_write_ok: bool = True,
    note: str | None = None,
    lag_threshold_seconds: float = 120.0,
    write_json: bool = True,
) -> Dict[str, Any]:
    """Recompute dual-write lag/hash **after** both trees exist on disk (Batch CJ).

    Producers often stamp provenance *before* the public write, freezing the
    previous public mtime into ``dual_write_lag_stale=true`` forever even when
    the subsequent dual-write succeeds and content hashes match. Call this
    after both files are written (or after public replace) so lag/hash reflect
    post-sync reality. Optionally rewrites private + public with the honest
    block so operator canaries clear.

    Deep-research: content-hash / sync_verified events beat sticky pre-write
    lag gauges.
    """
    priv = Path(private_path)
    pub = Path(public_path)
    paths_identical = False
    try:
        paths_identical = priv.resolve() == pub.resolve()
    except OSError:
        paths_identical = False

    stamped = _attach_dual_write_provenance(
        payload,
        private_path=priv,
        public_path=pub,
        dual_write_attempted=not paths_identical,
        dual_write_ok=dual_write_ok if not paths_identical else True,
        paths_identical=paths_identical,
        note=note
        or (
            "post_sync dual-write provenance (Batch CJ): lag/hash after both "
            "trees exist"
        ),
        lag_threshold_seconds=lag_threshold_seconds,
    )
    if not write_json:
        return stamped

    try:
        # Keep private diagnostics and public logical references separate.  A
        # single serialized body here used to reintroduce absolute private
        # paths after the normal fan-out had projected them.
        from src.monitor.signal_authority import write_json_multi_dest

        write_json_multi_dest(
            stamped,
            private_path=priv,
            public_path=pub,
            soft_mirror_repo=False,
        )
        # Second pass: mtimes now both post-sync; refresh lag/hash once more
        stamped = _attach_dual_write_provenance(
            stamped,
            private_path=priv,
            public_path=pub,
            dual_write_attempted=not paths_identical,
            dual_write_ok=True if dual_write_ok or paths_identical else dual_write_ok,
            paths_identical=paths_identical,
            note=note
            or (
                "post_sync dual-write provenance (Batch CJ): lag/hash after both "
                "trees exist"
            ),
            lag_threshold_seconds=lag_threshold_seconds,
        )
        write_json_multi_dest(
            stamped,
            private_path=priv,
            public_path=pub,
            soft_mirror_repo=False,
        )
    except OSError:
        # Best-effort; return last stamped payload even if rewrite fails
        pass
    return stamped


def _finalize_signal_metadata(output: Dict, *, finalized_at: str | None = None) -> Dict:
    """Stamp final artifact metadata after all signal sections are assembled."""
    timestamp = finalized_at or datetime.now(timezone.utc).isoformat()
    finalized = dict(output)
    finalized["generated_at"] = timestamp
    finalized["timestamp"] = timestamp
    # Batch BJ: full stamp with last_full retention + status (same contract as
    # health/overlay paths via _stamp_generator_git_sha).
    return _stamp_generator_git_sha(finalized, status="full_generate")


def _dist_data_dir_for_public_dir(public_dir: Path) -> Path:
    """Return the app dist/data directory that mirrors public/data."""
    if public_dir.name == "data" and public_dir.parent.name == "public":
        app_root = public_dir.parent.parent
    else:
        app_root = public_dir.parent
    return app_root / "dist" / "data"


def _mirror_public_data_contract_files_to_dist(public_dir: Path) -> None:
    """Mirror deploy-checked public data files after final generation."""
    dist_data = _dist_data_dir_for_public_dir(public_dir)
    dist_data.mkdir(parents=True, exist_ok=True)
    for filename in PUBLIC_DATA_DIST_MIRROR_FILES:
        source = public_dir / filename
        if source.exists():
            shutil.copyfile(source, dist_data / filename)

