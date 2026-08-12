#!/usr/bin/env python3
"""Mirror live operator public/data → repo public/data (Batch BW / H22b).

After deploy, tasker/dashboard write ``PUBLIC_DATA_DIR`` (often
``/var/www/portfolio-lab/data``). Repo ``public/data`` is a static checkout
mirror used by offline/dev and canary when ``--allow-repo-public-data`` is set.
Without an explicit post-deploy mirror, repo files keep stale
``generator_git_sha`` (SLSA dual-write / static-mirror refresh pattern).

This script **copies** verified live artifacts into the repo mirror. It never
invents git SHAs — provenance rides with the source payload (including
``partial_patch`` honesty).

Usage::

    uv run python scripts/mirror_repo_public_data.py
    uv run python scripts/mirror_repo_public_data.py --dry-run
    uv run python scripts/mirror_repo_public_data.py --source /var/www/portfolio-lab/data
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Align with consistency canary + common dual-write operators
DEFAULT_FILE_GLOBS: tuple[str, ...] = (
    "index.json",
    "source_manifest.json",
    "data_quality.json",
    "health.json",
    "health_ops.json",
    "alerts.json",
    "incidents.json",
    "decision_registry.json",
    "stats.json",
    "analytics.json",
    "graduation.json",
    "signals.json",
    "adaptive_sizing.json",
    "risk_decomposition.json",
    "overlay_dashboard.json",
    "rebalance_health.json",
    "labs_registry.json",
    "labs_scorecards.json",
    "labs_validation.json",
    "dashboard.json",
    "garch_cvar.json",
    "risk_metrics.json",  # Batch EH: private twin dual-write for WWW/index
    "unified_dashboard.json",
    "regime_gate.json",
    "tsmom.json",
    "cross_asset_rv.json",
    "black_litterman.json",
    "turnover_validator.json",
    "vixy_hedge.json",
    "vix_term_structure.json",
    "prices_compact.json",
    # Batch HV: market-data + tasker ops artifacts written by fetch-data /
    # tasker store under PUBLIC_DATA_DIR. Omitting them left residual live
    # byte-diff (prices/yields/tasker) while catalog lag stamped 0/33.
    "prices.json",
    "yields.json",
    "tasker_status.json",
    # The public index discovers every dated attribution shard, not only the
    # latest pointer.  Mirror the governed nested trees as a bounded glob so
    # index entries always resolve in the checkout mirror.
    "attribution/*.json",
    "explainability/*.json",
)

# The health job writes this artifact after evaluating mirror lag. Comparing
# that new output with the prior repo mirror would make the SLI report its own
# write as lag. Keep it in DEFAULT_FILE_GLOBS for copying and raw diagnostics;
# exclude it only from the qualifying operational SLI.
MIRROR_LAG_SLI_EXEMPT_PATHS: frozenset[str] = frozenset({"health_ops.json"})


@dataclass
class MirrorReport:
    source: str
    dest: str
    copied: list[str] = field(default_factory=list)
    skipped_missing: list[str] = field(default_factory=list)
    skipped_unchanged: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    dry_run: bool = False
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "dest": self.dest,
            "copied": list(self.copied),
            "skipped_missing": list(self.skipped_missing),
            "skipped_unchanged": list(self.skipped_unchanged),
            "errors": list(self.errors),
            "dry_run": self.dry_run,
            "generated_at": self.generated_at,
            "copied_count": len(self.copied),
            "live_authoritative": False,
            "note": (
                "Repo public/data is a static mirror of operator PUBLIC_DATA_DIR; "
                "provenance SHAs are copied, not re-invented (Batch BW / SLSA dual-write)."
            ),
        }


def _read_bytes(path: Path) -> Optional[bytes]:
    try:
        return path.read_bytes()
    except OSError:
        return None


def _atomic_write_bytes(path: Path, data: bytes, *, mode: int = 0o644) -> None:
    """Atomic replace with explicit mode (Batch HQ).

    Prefer signal_authority's text atomic writer for JSON when available so
    multi-dest and soft-mirror share the same 0o644 contract (mkstemp defaults
    to 0o600 → Caddy 403). Fall back to same-dir tempfile for binary-safe copy.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # JSON soft-mirror: reuse production multi-dest atomic path when importable.
    if path.suffix == ".json":
        try:
            from src.monitor.signal_authority import _atomic_write_text

            _atomic_write_text(path, data.decode("utf-8"), mode=mode)
            return
        except Exception:  # noqa: BLE001 — fall through to byte path
            pass
    import os
    import tempfile

    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        try:
            os.chmod(tmp_name, mode)
        except OSError:
            pass
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _sha_from_bytes(raw: bytes) -> Optional[str]:
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if isinstance(payload, dict):
        sha = payload.get("generator_git_sha")
        if sha is None and isinstance(payload.get("_meta"), dict):
            sha = payload["_meta"].get("generator_git_sha")
        if sha is None and isinstance(payload.get("meta"), dict):
            sha = payload["meta"].get("generator_git_sha")
        return str(sha) if sha is not None else None
    return None


def resolve_mirror_paths(
    relative: str,
    source_root: Path,
    dest_root: Path,
) -> tuple[Path, Path]:
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"refusing unsafe relative path: {relative!r}")
    return source_root / rel, dest_root / rel


def expand_mirror_file_specs(
    source_root: Path,
    files: Sequence[str],
) -> tuple[str, ...]:
    """Expand the small set of supported nested-tree mirror globs.

    Flat artifact names retain their existing missing-file reporting.  A
    nested glob with no matches is simply absent because those trees are
    optional public surfaces; every matching JSON file is copied explicitly.
    """
    expanded: list[str] = []
    seen: set[str] = set()
    for spec in files:
        if not any(token in spec for token in ("*", "?", "[")):
            candidates = (spec,)
        else:
            candidates = tuple(
                path.relative_to(source_root).as_posix()
                for path in sorted(source_root.glob(spec))
                if path.is_file()
            )
        for relative in candidates:
            if relative in seen:
                continue
            seen.add(relative)
            expanded.append(relative)
    return tuple(expanded)


# Health documents that carry nested repo_public_mirror_lag* SLIs and must be
# restamped after a soft-mirror so sticky false-critical does not freeze.
# Batch HO: signals.json nests the SLI under health (compact operator surface).
HEALTH_LAG_RESTAMP_FILES: tuple[str, ...] = (
    "health.json",
    "health_ops.json",
    "signals.json",
)


def mirror_repo_public_data(
    *,
    source_root: Path,
    dest_root: Path,
    files: Sequence[str] = DEFAULT_FILE_GLOBS,
    dry_run: bool = False,
    force: bool = False,
    restamp_health_lag: bool = True,
) -> MirrorReport:
    """Copy ``files`` from source_root → dest_root when content differs.

    Batch FX (EN/EY): after a successful copy pass, re-probe live lag and
    restamp nested ``repo_public_mirror_lag*`` fields on health docs at both
    source (operator SoT) and dest (repo mirror). Soft-mirror alone copies
    sticky critical stamps byte-for-byte and freezes false-critical until the
    next health :30.
    """
    report = MirrorReport(
        source=str(source_root),
        dest=str(dest_root),
        dry_run=dry_run,
    )
    source_root = Path(source_root)
    dest_root = Path(dest_root)
    files = expand_mirror_file_specs(source_root, files)
    if not source_root.is_dir():
        report.errors.append(f"source root missing: {source_root}")
        return report

    for rel in files:
        try:
            src, dst = resolve_mirror_paths(rel, source_root, dest_root)
        except ValueError as exc:
            report.errors.append(str(exc))
            continue
        if not src.is_file():
            report.skipped_missing.append(rel)
            continue
        src_bytes = _read_bytes(src)
        if src_bytes is None:
            report.errors.append(f"cannot read {src}")
            continue
        dst_bytes = _read_bytes(dst) if dst.is_file() else None
        if not force and dst_bytes is not None and dst_bytes == src_bytes:
            report.skipped_unchanged.append(rel)
            continue
        if dry_run:
            report.copied.append(rel)
            continue
        try:
            # Batch HQ: atomic 0o644 (not sticky 0600 / umask-dependent write_bytes)
            _atomic_write_bytes(dst, src_bytes, mode=0o644)
            report.copied.append(rel)
        except OSError as exc:
            report.errors.append(f"write failed {dst}: {exc}")

    # End-pipeline nested lag finalize (Batch FX EN/EY) — skip dry-run.
    if restamp_health_lag and not dry_run:
        try:
            from src.monitor.repo_public_mirror_lag import (
                restamp_mirror_lag_on_health_documents,
                summarize_repo_public_mirror_lag,
            )

            lag_summary = summarize_repo_public_mirror_lag(
                source_root=source_root, dest_root=dest_root
            )
            restamp_paths: list[Path] = []
            for rel in HEALTH_LAG_RESTAMP_FILES:
                for root in (source_root, dest_root):
                    candidate = root / rel
                    if candidate.is_file():
                        restamp_paths.append(candidate)
            # Also restamp private monitor SSOT when distinct from PUBLIC tree
            # (data/health.json freezes lag stamps between health :30 jobs).
            # Batch HM: never append production DATA_DIR when source/dest are
            # pytest ephemeral trees (soft-mirror unit tests would poison SSOT).
            # Batch HO: also restamp private data/signals.json nested health SLI.
            # Batch ID: also restamp private data/health_ops.json (IC multi-dest
            # twin) so soft-mirror restamp cannot re-split public/repo vs private
            # by adding only mirror_lag_restamped_at on PUBLIC roots.
            try:
                from src.paths import DATA_DIR as _DATA_DIR
                from src.monitor.repo_public_mirror_lag import (
                    is_ephemeral_restamp_path,
                )

                roots_ephemeral = is_ephemeral_restamp_path(
                    source_root
                ) or is_ephemeral_restamp_path(dest_root)
                if not roots_ephemeral:
                    for basename in (
                        "health.json",
                        "health_ops.json",
                        "signals.json",
                    ):
                        private_doc = Path(_DATA_DIR) / basename
                        if private_doc.is_file() and not is_ephemeral_restamp_path(
                            private_doc
                        ):
                            restamp_paths.append(private_doc)
            except Exception:  # noqa: BLE001
                pass
            # Deduplicate by resolve when src==dest (dev shells)
            seen: set[str] = set()
            unique_paths: list[Path] = []
            for p in restamp_paths:
                key = str(p.resolve())
                if key in seen:
                    continue
                seen.add(key)
                unique_paths.append(p)
            if unique_paths:
                restamp_mirror_lag_on_health_documents(
                    paths=unique_paths,
                    lag_summary=lag_summary,
                )
            # Health-only cron writes alerts mid-job while live lag can still be
            # non-zero (self-induced health/index writes). After soft-mirror +
            # restamp heal lag to 0, re-publish alerts so dual-plane labeling
            # sees final ops (signal_quality for SH-only, not Health Warning: ops).
            # Skip under pytest / ephemeral roots — never touch prod alerts from
            # fixture mirrors.
            try:
                import os

                try:
                    lag_n = int((lag_summary or {}).get("lagging_count") or 0)
                except (TypeError, ValueError):
                    lag_n = -1
                under_pytest = bool(os.environ.get("PYTEST_CURRENT_TEST"))
                roots_ephemeral = False
                try:
                    from src.monitor.repo_public_mirror_lag import (
                        is_ephemeral_restamp_path,
                    )

                    roots_ephemeral = is_ephemeral_restamp_path(
                        source_root
                    ) or is_ephemeral_restamp_path(dest_root)
                except Exception:  # noqa: BLE001
                    roots_ephemeral = False
                if lag_n == 0 and not under_pytest and not roots_ephemeral:
                    from src.monitor.health_check import publish_health_alerts_json

                    publish_health_alerts_json()
            except Exception as alerts_exc:  # noqa: BLE001 — never fail mirror
                report.errors.append(
                    f"alerts republish after lag heal skipped: {alerts_exc}"
                )
        except Exception as exc:  # noqa: BLE001 — never fail soft-mirror on restamp
            report.errors.append(f"health lag restamp skipped: {exc}")

    return report


def lag_report(
    source_root: Path,
    dest_root: Path,
    files: Sequence[str] = DEFAULT_FILE_GLOBS,
) -> list[dict[str, Any]]:
    """Compare generator_git_sha (or presence) between source and dest."""
    rows: list[dict[str, Any]] = []
    for rel in expand_mirror_file_specs(Path(source_root), files):
        try:
            src, dst = resolve_mirror_paths(rel, Path(source_root), Path(dest_root))
        except ValueError:
            continue
        src_b = _read_bytes(src) if src.is_file() else None
        dst_b = _read_bytes(dst) if dst.is_file() else None
        if src_b is None and dst_b is None:
            continue
        rows.append(
            {
                "path": rel,
                "source_present": src_b is not None,
                "dest_present": dst_b is not None,
                "source_sha": _sha_from_bytes(src_b) if src_b else None,
                "dest_sha": _sha_from_bytes(dst_b) if dst_b else None,
                "bytes_equal": (
                    src_b is not None
                    and dst_b is not None
                    and src_b == dst_b
                ),
                "lagging": (
                    src_b is not None
                    and (dst_b is None or src_b != dst_b)
                ),
            }
        )
    return rows


def lag_sli_report(
    source_root: Path,
    dest_root: Path,
    files: Sequence[str] = DEFAULT_FILE_GLOBS,
) -> list[dict[str, Any]]:
    """Return only comparisons that qualify for the mirror-lag SLI."""
    return [
        row
        for row in lag_report(source_root, dest_root, files=files)
        if row.get("path") not in MIRROR_LAG_SLI_EXEMPT_PATHS
    ]


def _default_source() -> Path:
    from src.paths import PUBLIC_DATA_DIR

    return Path(PUBLIC_DATA_DIR)


def _default_dest() -> Path:
    return _PROJECT_ROOT / "public" / "data"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Operator public data dir (default: PUBLIC_DATA_DIR)",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=None,
        help="Repo public/data (default: <repo>/public/data)",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rewrite even when bytes already match",
    )
    parser.add_argument(
        "--lag-only",
        action="store_true",
        help="Print lag report JSON and exit 1 if any lagging files",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    source = Path(args.source) if args.source else _default_source()
    dest = Path(args.dest) if args.dest else _default_dest()

    if args.lag_only:
        rows = lag_sli_report(source, dest)
        lagging = [r for r in rows if r.get("lagging")]
        print(json.dumps({"lagging": lagging, "total": len(rows)}, indent=2))
        return 1 if lagging else 0

    report = mirror_repo_public_data(
        source_root=source,
        dest_root=dest,
        dry_run=bool(args.dry_run),
        force=bool(args.force),
    )
    print(json.dumps(report.to_dict(), indent=2))
    return 1 if report.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
