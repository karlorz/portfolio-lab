"""Immutable generation store (Option C, operator-approved 2026-08-11).

Each completed generator run records its public artifact surface as an
immutable generation under ``GENERATIONS_DIR/<run_id>/``: a manifest plus
per-file sha256 for every catalogued JSON artifact. A ``current`` pointer
(JSON + symlink) is updated atomically (temp file + ``os.replace``) only
after the generation is fully written, so operators can always resolve the
last complete generation and roll the flat public dir back to any previous
generation (``rollback_to``). Generations are never rewritten (immutable);
``prune`` keeps the newest N.

The flat public dir remains the compatibility surface (no route changes);
generations are the durable, verifiable record of what each run produced —
the 07:15Z partial-dashboard.json failure mode is recoverable by rolling
back to the previous generation.

CLI::

    python -m src.dashboard.generation_store --list
    python -m src.dashboard.generation_store --rollback <run_id>
    python -m src.dashboard.generation_store --prune
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

from src.monitor.decision_registry import _git_sha_short
from src.paths import DATA_DIR

logger = logging.getLogger(__name__)

GENERATION_SCHEMA = "generation-store/v1"
GENERATIONS_DIR = DATA_DIR / "generations"
CURRENT_POINTER = GENERATIONS_DIR / "current.json"
CURRENT_LINK = GENERATIONS_DIR / "current"
MANIFEST_NAME = "manifest.json"
KEEP_GENERATIONS = int(os.environ.get("GENERATION_KEEP", "7"))
# Hash cache is a private artifact, not part of the public surface.
_EXCLUDED_FILENAMES = {".public_data_index_hash_cache.json"}

_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,120}$")


@dataclass
class GenerationManifest:
    """Schema for one immutable generation."""

    schema: str
    run_id: str
    generated_at: str
    git_sha: str
    files: dict[str, str] = field(default_factory=dict)  # rel path -> sha256
    file_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "run_id": self.run_id,
            "generated_at": self.generated_at,
            "git_sha": self.git_sha,
            "files": dict(self.files),
            "file_count": self.file_count,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "GenerationManifest | None":
        if not isinstance(payload, dict):
            return None
        if payload.get("schema") != GENERATION_SCHEMA:
            return None
        run_id = str(payload.get("run_id") or "")
        if not run_id or not _RUN_ID_RE.match(run_id):
            return None
        files = payload.get("files")
        if not isinstance(files, dict):
            return None
        return cls(
            schema=GENERATION_SCHEMA,
            run_id=run_id,
            generated_at=str(payload.get("generated_at") or ""),
            git_sha=str(payload.get("git_sha") or ""),
            files={str(k): str(v) for k, v in files.items()},
            file_count=int(payload.get("file_count") or len(files)),
        )


def _sha256_bytes(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _run_id_default(now: Optional[datetime] = None) -> str:
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%S")
    return f"gen-{stamp}"


def _iter_public_json(public_dir: Path):
    """Yield (relative_posix_path, Path) for every catalogued JSON artifact.

    Recurses subdirectories (explainability/, attribution/ shards) but skips
    the hash-cache file and anything outside the public dir.
    """
    public_dir = Path(public_dir)
    for path in sorted(public_dir.rglob("*.json")):
        if path.name in _EXCLUDED_FILENAMES:
            continue
        rel = path.relative_to(public_dir)
        yield rel.as_posix(), path


class GenerationStore:
    """Immutable generation dirs + atomic current pointer over a public dir."""

    def __init__(
        self,
        generations_dir: Optional[Path] = None,
        public_dir: Optional[Path] = None,
    ) -> None:
        self.generations_dir = Path(generations_dir or GENERATIONS_DIR)
        self.public_dir = Path(public_dir) if public_dir is not None else None
        self.current_pointer = self.generations_dir / "current.json"
        self.current_link = self.generations_dir / "current"

    # ── generation dirs ──

    def _run_dir(self, run_id: str) -> Path:
        if not _RUN_ID_RE.match(run_id):
            raise ValueError(f"invalid run_id: {run_id!r}")
        return self.generations_dir / run_id

    def _manifest_path(self, run_id: str) -> Path:
        return self._run_dir(run_id) / MANIFEST_NAME

    # ── record ──

    def record(
        self,
        run_id: Optional[str] = None,
        public_dir: Optional[Path] = None,
        now: Optional[datetime] = None,
    ) -> GenerationManifest:
        """Snapshot the public JSON surface into an immutable generation and
        atomically flip the current pointer. Raises on failure (the caller
        decides whether recording is fatal)."""
        public_dir = Path(public_dir or self.public_dir or DATA_DIR)
        run_id = run_id or _run_id_default(now)
        if not _RUN_ID_RE.match(run_id):
            raise ValueError(f"invalid run_id: {run_id!r}")
        if not public_dir.is_dir():
            raise FileNotFoundError(f"public_dir not found: {public_dir}")

        run_dir = self._run_dir(run_id)
        if run_dir.exists():
            raise FileExistsError(f"generation already exists: {run_id}")
        run_dir.mkdir(parents=True, exist_ok=False)

        files: dict[str, str] = {}
        try:
            for rel, path in _iter_public_json(public_dir):
                sha = _sha256_file(path)
                files[rel] = sha
                dest = run_dir / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(path.read_bytes())
        except Exception:
            # Never leave a half-written generation behind.
            import shutil

            shutil.rmtree(run_dir, ignore_errors=True)
            raise

        manifest = GenerationManifest(
            schema=GENERATION_SCHEMA,
            run_id=run_id,
            generated_at=(now or datetime.now(timezone.utc)).isoformat(timespec="seconds"),
            git_sha=str(_git_sha_short() or ""),
            files=files,
            file_count=len(files),
        )
        (run_dir / MANIFEST_NAME).write_text(
            json.dumps(manifest.to_dict(), indent=2), encoding="utf-8"
        )
        self._activate(manifest)
        self.prune(keep=KEEP_GENERATIONS)
        logger.info(
            "generation recorded: %s (%d files, sha %s)",
            run_id,
            manifest.file_count,
            manifest.git_sha,
        )
        return manifest

    # ── pointer ──

    def _activate(self, manifest: GenerationManifest) -> None:
        """Atomically update current.json + the current symlink."""
        self.generations_dir.mkdir(parents=True, exist_ok=True)
        tmp_json = self.current_pointer.with_name("current.json.tmp")
        tmp_json.write_text(
            json.dumps(manifest.to_dict(), indent=2), encoding="utf-8"
        )
        os.replace(tmp_json, self.current_pointer)
        tmp_link = self.current_link.with_name("current.link.tmp")
        try:
            if tmp_link.exists() or tmp_link.is_symlink():
                tmp_link.unlink()
        except OSError:
            pass
        os.symlink(manifest.run_id, tmp_link)
        os.replace(tmp_link, self.current_link)

    def current(self) -> Optional[GenerationManifest]:
        """Resolve the current pointer, or None when no generation exists."""
        if not self.current_pointer.is_file():
            return None
        try:
            payload = json.loads(self.current_pointer.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        manifest = GenerationManifest.from_dict(payload)
        if manifest is None:
            return None
        if not self._manifest_path(manifest.run_id).is_file():
            return None
        return manifest

    def generations(self) -> list[GenerationManifest]:
        """All recorded generations, newest first."""
        out: list[GenerationManifest] = []
        if not self.generations_dir.is_dir():
            return out
        for run_dir in sorted(self.generations_dir.iterdir(), reverse=True):
            # Generation dirs are real directories; the current symlink (and
            # any other non-generation entries) must not be counted.
            if not run_dir.is_dir() or run_dir.is_symlink():
                continue
            manifest_path = run_dir / MANIFEST_NAME
            if not manifest_path.is_file():
                continue
            try:
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            manifest = GenerationManifest.from_dict(payload)
            if manifest is not None:
                out.append(manifest)
        return out

    # ── rollback / prune ──

    def rollback_to(self, run_id: str, public_dir: Optional[Path] = None) -> int:
        """Mirror a previous generation's files over the flat public dir.

        Returns the number of files restored. Each file's sha256 is verified
        after the copy; a mismatch is logged as an error but does not abort
        the remaining files. The pointer is NOT flipped (rollback is an
        operator recovery action on the compatibility surface; the current
        generation stays authoritative until the next record).
        """
        public_dir = Path(public_dir or self.public_dir or DATA_DIR)
        manifest_path = self._manifest_path(run_id)
        if not manifest_path.is_file():
            raise FileNotFoundError(f"no such generation: {run_id}")
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = GenerationManifest.from_dict(payload)
        if manifest is None:
            raise ValueError(f"invalid manifest for generation: {run_id}")

        restored = 0
        for rel, expected_sha in sorted(manifest.files.items()):
            src = self._run_dir(run_id) / rel
            if not src.is_file():
                logger.error("rollback: missing file in generation %s: %s", run_id, rel)
                continue
            dest = public_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(src.read_bytes())
            if _sha256_file(dest) != expected_sha:
                logger.error(
                    "rollback: sha mismatch after restore for %s (%s)", rel, run_id
                )
                continue
            restored += 1
        logger.info("rollback to %s restored %d/%d files", run_id, restored, len(manifest.files))
        return restored

    def prune(self, keep: int = KEEP_GENERATIONS) -> list[str]:
        """Delete the oldest generations beyond ``keep`` (never the current)."""
        if keep < 1:
            keep = 1
        current_id = self.current().run_id if self.current() else None
        removed: list[str] = []
        for manifest in reversed(self.generations()):
            if len(self.generations()) <= keep:
                break
            if manifest.run_id == current_id:
                continue
            import shutil

            shutil.rmtree(self._run_dir(manifest.run_id), ignore_errors=True)
            removed.append(manifest.run_id)
            logger.info("pruned generation %s", manifest.run_id)
        return removed


def main() -> None:
    parser = argparse.ArgumentParser(description="Generation store operations")
    parser.add_argument("--list", action="store_true", help="list generations")
    parser.add_argument("--rollback", metavar="RUN_ID", help="roll flat public dir back to a generation")
    parser.add_argument("--prune", action="store_true", help="prune old generations")
    args = parser.parse_args()

    store = GenerationStore()
    if args.list:
        for m in store.generations():
            print(f"{m.run_id}  {m.generated_at}  {m.file_count} files  sha={m.git_sha}")
        cur = store.current()
        print(f"current: {cur.run_id if cur else '(none)'}")
    elif args.rollback:
        count = store.rollback_to(args.rollback)
        print(f"restored {count} files from {args.rollback}")
    elif args.prune:
        removed = store.prune()
        print(f"pruned: {removed}")
    else:
        parser.print_help()


if __name__ == "__main__":
    from src.utils.log_config import configure_logging

    configure_logging()
    main()
