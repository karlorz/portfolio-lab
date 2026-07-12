"""Dry-run retention reporting for Labs and dashboard artifacts."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from src.paths import DATA_DIR, PROJECT_ROOT, PUBLIC_DATA_DIR, WIKI_DIR

ARTIFACT_RETENTION_SCHEMA_VERSION = "artifact-retention-report/v1"
LABS_ARTIFACT_ARCHIVE_PLAN_SCHEMA_VERSION = "labs-artifact-archive-plan/v1"

ARCHIVE_AFTER_DAYS = 180
PRUNE_LOG_AFTER_DAYS = 90

Recommendation = str
ReferenceMap = dict[str, set[str]]

__all__ = [
    "ARTIFACT_RETENTION_SCHEMA_VERSION",
    "LABS_ARTIFACT_ARCHIVE_PLAN_SCHEMA_VERSION",
    "build_archive_dry_run_plan",
    "build_retention_report",
]


def _display_path(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _normalize_reference_path(path: str | Path, project_root: Path) -> str:
    path_obj = Path(path)
    if path_obj.is_absolute():
        return _display_path(path_obj, project_root)
    return path_obj.as_posix().lstrip("./")


def _read_json(path: Path) -> Mapping[str, Any] | None:
    try:
        with open(path) as f:
            payload = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    return payload if isinstance(payload, Mapping) else None


def _candidate_paths(data_dir: Path, public_data_dir: Path) -> dict[Path, str]:
    candidates: dict[Path, str] = {}

    def add_many(paths: Iterable[Path], category: str) -> None:
        for path in paths:
            if path.is_file() and not path.name.endswith(".manifest.json"):
                candidates.setdefault(path, category)

    add_many((data_dir / "backtest_results").glob("*.json"), "experiment_result")
    add_many((data_dir / "historical_orders").glob("*.json"), "operational_history")
    add_many((data_dir / "attribution").glob("*.json"), "attribution_snapshot")
    add_many((data_dir / "llm_costs").glob("*.json"), "cost_history")
    add_many(data_dir.glob("daily_pnl*.json"), "daily_summary")
    add_many(data_dir.glob("daily_pnl*.jsonl"), "daily_summary")
    add_many(public_data_dir.glob("*.json"), "dashboard_state")
    add_many((data_dir / "logs").glob("*.log"), "raw_log")
    add_many(data_dir.glob("*.log"), "raw_log")

    return candidates


def _add_reference(reference_map: ReferenceMap, artifact_path: Any, source: str, project_root: Path) -> None:
    if not isinstance(artifact_path, str) or not artifact_path.strip():
        return
    key = _normalize_reference_path(artifact_path, project_root)
    reference_map[key].add(source)


def _collect_registry_references(data_dir: Path, public_data_dir: Path, project_root: Path) -> ReferenceMap:
    references: ReferenceMap = defaultdict(set)
    registry_paths = {
        data_dir / "labs_registry.json",
        public_data_dir / "labs_registry.json",
    }
    for registry_path in sorted(registry_paths):
        payload = _read_json(registry_path)
        if payload is None:
            continue
        experiments = payload.get("experiments")
        if not isinstance(experiments, list):
            continue
        source_path = _display_path(registry_path, project_root)
        for index, entry in enumerate(experiments):
            if not isinstance(entry, Mapping):
                continue
            source = f"{source_path}:experiments[{index}].artifact_path"
            _add_reference(references, entry.get("artifact_path"), source, project_root)
    return references


def _collect_manifest_references(data_dir: Path, public_data_dir: Path, project_root: Path) -> ReferenceMap:
    references: ReferenceMap = defaultdict(set)
    manifest_paths: set[Path] = set()
    for root in (data_dir, public_data_dir):
        if root.exists():
            manifest_paths.update(root.rglob("*.manifest.json"))

    for manifest_path in sorted(manifest_paths):
        payload = _read_json(manifest_path)
        if payload is None:
            continue
        source_path = _display_path(manifest_path, project_root)
        _add_reference(references, payload.get("source_artifact_path"), f"{source_path}:source_artifact_path", project_root)
    return references


def _collect_public_index_references(public_data_dir: Path, project_root: Path) -> ReferenceMap:
    references: ReferenceMap = defaultdict(set)
    index_path = public_data_dir / "index.json"
    payload = _read_json(index_path)
    if payload is None:
        return references

    source_path = _display_path(index_path, project_root)
    files = payload.get("files")
    if isinstance(files, list):
        for index, filename in enumerate(files):
            if isinstance(filename, str) and filename.strip():
                target_path = public_data_dir / filename
                _add_reference(references, str(target_path), f"{source_path}:files[{index}]", project_root)

    entries = payload.get("entries")
    if isinstance(entries, list):
        for index, entry in enumerate(entries):
            if not isinstance(entry, Mapping) or entry.get("status") == "missing":
                continue
            path_value = entry.get("path") or entry.get("filename")
            if not isinstance(path_value, str) or not path_value.strip():
                continue
            path = Path(path_value)
            target_path = path if path.is_absolute() else public_data_dir / path
            _add_reference(references, str(target_path), f"{source_path}:entries[{index}].path", project_root)

    return references


def _reference_roots(reference_roots: Sequence[str | Path] | None) -> list[Path]:
    if reference_roots is not None:
        return [Path(root) for root in reference_roots]

    default_root = WIKI_DIR / "projects" / "portfolio-lab"
    return [default_root] if default_root.exists() else []


def _collect_text_references(
    *,
    artifact_keys: Iterable[str],
    project_root: Path,
    reference_roots: Sequence[str | Path] | None,
) -> ReferenceMap:
    references: ReferenceMap = defaultdict(set)
    keys = tuple(sorted(set(artifact_keys), key=lambda value: (-len(value), value)))
    if not keys:
        return references
    key_pattern = re.compile("|".join(re.escape(key) for key in keys))

    for root in _reference_roots(reference_roots):
        if not root.exists():
            continue
        files = [root] if root.is_file() else sorted(root.rglob("*"))
        for path in files:
            if not path.is_file() or path.suffix.lower() not in {".md", ".txt", ".rst"}:
                continue
            try:
                text = path.read_text(errors="ignore")
            except OSError:
                continue
            source = _display_path(path, project_root)
            for match in key_pattern.finditer(text):
                references[match.group(0)].add(source)
    return references


def _merge_references(*maps: ReferenceMap) -> ReferenceMap:
    merged: ReferenceMap = defaultdict(set)
    for reference_map in maps:
        for path, sources in reference_map.items():
            merged[path].update(sources)
    return merged


def _age_days(path: Path, now: datetime) -> int | None:
    """Return age in whole days, or None if the path vanished (TOCTOU)."""
    try:
        modified_at = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    except OSError:
        return None
    return max(0, int((now - modified_at).total_seconds() // 86_400))


def _policy_for(category: str, age_days: int) -> tuple[Recommendation, str, str]:
    if category == "dashboard_state":
        return "keep", "public_dashboard", "public dashboard files remain visible until replaced"
    if category == "raw_log":
        if age_days >= PRUNE_LOG_AFTER_DAYS:
            return "prune", "transient_log", f"raw log older than {PRUNE_LOG_AFTER_DAYS} days"
        return "keep", "recent_log", "recent raw log remains useful for operations"
    if category in {
        "attribution_snapshot",
        "cost_history",
        "daily_summary",
        "experiment_result",
        "operational_history",
    }:
        if age_days >= ARCHIVE_AFTER_DAYS:
            return "archive", "archive_candidate", f"artifact older than {ARCHIVE_AFTER_DAYS} days"
        return "keep", "active_history", "artifact is within the active retention window"
    return "keep", "unclassified", "no retention rule matched"


def _entry_for_path(
    *,
    path: Path,
    category: str,
    project_root: Path,
    now: datetime,
    references: ReferenceMap,
) -> dict[str, Any] | None:
    """Build one retention entry, or None if the path is no longer readable."""
    display_path = _display_path(path, project_root)
    referenced_by = sorted(references.get(display_path, set()))
    age_days = _age_days(path, now)
    if age_days is None:
        return None

    if referenced_by:
        recommendation = "keep"
        retention_tier = "protected_reference"
        reason = "artifact is referenced by registry, provenance, or project documentation"
        protected = True
    else:
        recommendation, retention_tier, reason = _policy_for(category, age_days)
        protected = False

    try:
        size_bytes = path.stat().st_size
    except OSError:
        return None

    return {
        "path": display_path,
        "category": category,
        "retention_tier": retention_tier,
        "recommendation": recommendation,
        "protected": protected,
        "referenced_by": referenced_by,
        "size_bytes": size_bytes,
        "age_days": age_days,
        "reason": reason,
    }


def _counts(entries: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = {"keep": 0, "archive": 0, "prune": 0}
    for entry in entries:
        recommendation = entry.get("recommendation")
        if isinstance(recommendation, str):
            counts[recommendation] = counts.get(recommendation, 0) + 1
    return counts


def _reason_codes_for_entry(entry: Mapping[str, Any]) -> list[str]:
    recommendation = entry.get("recommendation")
    retention_tier = entry.get("retention_tier")
    category = entry.get("category")
    referenced_by = entry.get("referenced_by")
    references = referenced_by if isinstance(referenced_by, list) else []
    codes: list[str] = []

    if recommendation == "archive":
        codes.append("archive_candidate")
        age_days = entry.get("age_days")
        if isinstance(age_days, int) and age_days >= ARCHIVE_AFTER_DAYS:
            codes.append(f"age_gte_{ARCHIVE_AFTER_DAYS}_days")
    elif retention_tier == "protected_reference":
        codes.append("protected_reference")
    elif recommendation == "keep" and retention_tier == "active_history":
        codes.append("active_retention_window")
    elif recommendation == "keep" and category == "dashboard_state":
        codes.append("recent_dashboard_output")
    elif recommendation == "keep" and retention_tier == "recent_log":
        codes.append("recent_log")
    elif recommendation == "prune":
        codes.append("prune_candidate")

    if any("index.json" in source for source in references):
        codes.append("public_index_reference")

    if not codes and isinstance(retention_tier, str):
        codes.append(retention_tier)
    return codes


def _planned_archive_path(source_path: str, archive_root: str) -> str:
    return f"{archive_root.rstrip('/')}/{source_path.lstrip('/')}"


def _archive_move_candidate(entry: Mapping[str, Any], archive_root: str) -> dict[str, Any]:
    source_path = str(entry["path"])
    return {
        "source_path": source_path,
        "planned_archive_path": _planned_archive_path(source_path, archive_root),
        "category": entry["category"],
        "age_days": entry["age_days"],
        "size_bytes": entry["size_bytes"],
        "reason_codes": _reason_codes_for_entry(entry),
    }


def _protected_archive_entry(entry: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source_path": entry["path"],
        "category": entry["category"],
        "recommendation": entry["recommendation"],
        "retention_tier": entry["retention_tier"],
        "age_days": entry["age_days"],
        "size_bytes": entry["size_bytes"],
        "reason_codes": _reason_codes_for_entry(entry),
        "referenced_by": entry["referenced_by"],
    }


def build_retention_report(
    *,
    data_dir: str | Path = DATA_DIR,
    public_data_dir: str | Path = PUBLIC_DATA_DIR,
    project_root: str | Path = PROJECT_ROOT,
    reference_roots: Sequence[str | Path] | None = None,
    archive_root: str | Path | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Build a non-destructive keep/archive/prune report for project artifacts."""
    data_dir = Path(data_dir)
    public_data_dir = Path(public_data_dir)
    project_root = Path(project_root)
    archive_root_path = Path(archive_root) if archive_root is not None else data_dir / "archive"
    now = now or datetime.now(timezone.utc)

    candidates = _candidate_paths(data_dir, public_data_dir)
    artifact_keys = [_display_path(path, project_root) for path in candidates]
    references = _merge_references(
        _collect_registry_references(data_dir, public_data_dir, project_root),
        _collect_manifest_references(data_dir, public_data_dir, project_root),
        _collect_public_index_references(public_data_dir, project_root),
        _collect_text_references(
            artifact_keys=artifact_keys,
            project_root=project_root,
            reference_roots=reference_roots,
        ),
    )

    entries = [
        entry
        for entry in (
            _entry_for_path(
                path=path,
                category=category,
                project_root=project_root,
                now=now,
                references=references,
            )
            for path, category in sorted(
                candidates.items(),
                key=lambda item: _display_path(item[0], project_root),
            )
        )
        if entry is not None
    ]

    return {
        "schema_version": ARTIFACT_RETENTION_SCHEMA_VERSION,
        "generated_at": now.isoformat(),
        "dry_run": True,
        "archive_root": _display_path(archive_root_path, project_root),
        "counts": _counts(entries),
        "entries": entries,
    }


def build_archive_dry_run_plan(
    *,
    data_dir: str | Path = DATA_DIR,
    public_data_dir: str | Path = PUBLIC_DATA_DIR,
    project_root: str | Path = PROJECT_ROOT,
    reference_roots: Sequence[str | Path] | None = None,
    archive_root: str | Path | None = None,
    now: datetime | None = None,
    retention_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a report-only archive plan from the artifact retention report."""
    report = (
        dict(retention_report)
        if retention_report is not None
        else build_retention_report(
            data_dir=data_dir,
            public_data_dir=public_data_dir,
            project_root=project_root,
            reference_roots=reference_roots,
            archive_root=archive_root,
            now=now,
        )
    )
    archive_root_display = str(report.get("archive_root", "data/archive"))
    report_entries = report.get("entries")
    entries = [entry for entry in report_entries if isinstance(entry, Mapping)] if isinstance(report_entries, list) else []

    move_candidates = [
        _archive_move_candidate(entry, archive_root_display)
        for entry in entries
        if entry.get("recommendation") == "archive" and not entry.get("protected", False)
    ]
    protected = [
        _protected_archive_entry(entry)
        for entry in entries
        if entry.get("protected", False) or entry.get("recommendation") == "keep"
    ]
    move_candidates.sort(key=lambda entry: entry["source_path"])
    protected.sort(key=lambda entry: entry["source_path"])

    return {
        "schema_version": LABS_ARTIFACT_ARCHIVE_PLAN_SCHEMA_VERSION,
        "source_report_schema_version": report.get("schema_version", ARTIFACT_RETENTION_SCHEMA_VERSION),
        "generated_at": report.get("generated_at"),
        "dry_run": True,
        "move_enabled": False,
        "archive_root": archive_root_display,
        "guardrails": {
            "destructive_actions_allowed": False,
            "requires_explicit_move_opt_in": True,
            "move_opt_in_flag": "--execute-move",
        },
        "counts": {
            "move_candidates": len(move_candidates),
            "protected": len(protected),
            "source_entries": len(entries),
        },
        "move_candidates": move_candidates,
        "protected": protected,
    }
