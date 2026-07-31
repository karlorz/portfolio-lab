#!/usr/bin/env python3
"""Build a verified Portfolio Lab static release from a clean git SHA."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from verify_lab_release import (
    MANIFEST_NAME,
    SCHEMA_VERSION,
    iter_deployable_assets,
    sha256_file,
    verify_release,
)


DEFAULT_BUILD_COMMAND = "bun run build"
DEFAULT_INSTALL_COMMAND = "bun install --frozen-lockfile"

# These files are deliberately tracked because scheduled writers use them as
# runtime inputs, but the deploy refresh regenerates their provenance stamp
# from the reviewed source SHA.  They are excluded from the static release
# identity (see copy_static_tree), so a normal data refresh must not make an
# otherwise reviewed source checkout undeployable.  Keep this allowlist exact:
# implementation edits and all other source changes must still fail closed.
GENERATED_RUNTIME_SOURCE_PATHS = frozenset(
    {
        "data/ensemble_weights.json",
        "data/vix_term_structure.json",
    }
)


def run(command: list[str], *, cwd: Path) -> str:
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout.strip()


def run_shell(command: str, *, cwd: Path) -> None:
    subprocess.run(command, cwd=cwd, check=True, shell=True)


def full_git_sha(repo_dir: Path, ref: str) -> str:
    return run(["git", "rev-parse", "--verify", f"{ref}^{{commit}}"], cwd=repo_dir)


def ensure_clean_source(repo_dir: Path) -> None:
    status = subprocess.run(
        ["git", "status", "--porcelain=v1"],
        cwd=repo_dir,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.rstrip("\n")
    if not status:
        return

    unexpected: list[str] = []
    for line in status.splitlines():
        # Porcelain v1 starts with the two-character index/worktree status and
        # a space.  Only ordinary modifications of the two known generated
        # files are permitted; additions, deletions, renames, and untracked
        # files remain deployment blockers.
        status_code = line[:2]
        path = line[3:] if len(line) >= 4 else ""
        if status_code not in {" M", "M ", "MM"} or path not in GENERATED_RUNTIME_SOURCE_PATHS:
            unexpected.append(line)

    if unexpected:
        raise SystemExit(
            "Refusing to build release from dirty source tree. Commit, stash, or remove changes first. "
            "Only modified scheduled runtime artifacts are exempt: "
            + ", ".join(sorted(GENERATED_RUNTIME_SOURCE_PATHS))
            + ". Unexpected changes:\n"
            + "\n".join(unexpected)
        )


def bun_version(worktree_dir: Path) -> str:
    try:
        return run(["bun", "--version"], cwd=worktree_dir)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unavailable"


def select_lockfile(repo_dir: Path) -> Path:
    for name in ("bun.lock", "bun.lockb", "package-lock.json"):
        candidate = repo_dir / name
        if candidate.is_file():
            return candidate
    raise SystemExit("No supported lockfile found; expected bun.lock or package-lock.json")


def copy_static_tree(dist_dir: Path, release_dir: Path) -> None:
    if not dist_dir.is_dir():
        raise SystemExit(f"Missing build output directory: {dist_dir}")

    tmp = release_dir.with_name(f".{release_dir.name}.tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    for item in dist_dir.iterdir():
        if item.name == "data":
            continue
        dest = tmp / item.name
        if item.is_dir():
            shutil.copytree(item, dest)
        elif item.is_file():
            shutil.copy2(item, dest)

    if release_dir.exists():
        shutil.rmtree(release_dir)
    tmp.replace(release_dir)


def build_manifest(
    release_dir: Path,
    *,
    source_git_sha: str,
    build_command: str,
    bun_version_value: str,
    lockfile_path: str,
    lockfile_sha256: str,
    build_time_utc: str | None = None,
) -> dict[str, object]:
    assets = []
    for path in iter_deployable_assets(release_dir):
        rel = path.relative_to(release_dir).as_posix()
        assets.append(
            {
                "path": rel,
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "source_git_sha": source_git_sha,
        "build_time_utc": build_time_utc
        or datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "build_command": build_command,
        "bun_version": bun_version_value,
        "lockfile": {
            "path": lockfile_path,
            "sha256": lockfile_sha256,
        },
        "policy": {
            "mutable_data_excluded": True,
            "excluded_paths": ["data/**"],
            "public_integrity_metadata_only": True,
        },
        "assets": assets,
    }


def write_manifest(release_dir: Path, manifest: dict[str, object]) -> Path:
    path = release_dir / MANIFEST_NAME
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def add_detached_worktree(repo_dir: Path, source_sha: str, worktree_dir: Path) -> None:
    if worktree_dir.exists():
        shutil.rmtree(worktree_dir)
    run(["git", "worktree", "add", "--detach", str(worktree_dir), source_sha], cwd=repo_dir)


def remove_worktree(repo_dir: Path, worktree_dir: Path) -> None:
    subprocess.run(["git", "worktree", "remove", "--force", str(worktree_dir)], cwd=repo_dir)


def build_release(
    *,
    repo_dir: Path,
    release_dir: Path,
    source_ref: str = "HEAD",
    build_command: str = DEFAULT_BUILD_COMMAND,
    install_command: str = DEFAULT_INSTALL_COMMAND,
    worktree_dir: Path | None = None,
    keep_worktree: bool = False,
) -> Path:
    repo_dir = repo_dir.resolve()
    release_dir = release_dir.resolve()
    ensure_clean_source(repo_dir)
    source_sha = full_git_sha(repo_dir, source_ref)
    lockfile = select_lockfile(repo_dir)
    lockfile_sha = sha256_file(lockfile)

    owned_temp_path: Path | None = None
    if worktree_dir is None:
        owned_temp_path = Path(tempfile.mkdtemp(prefix="portfolio-lab-release-worktree-"))
        worktree_dir = owned_temp_path
    else:
        worktree_dir = worktree_dir.resolve()

    add_detached_worktree(repo_dir, source_sha, worktree_dir)
    try:
        if install_command:
            run_shell(install_command, cwd=worktree_dir)
        run_shell(build_command, cwd=worktree_dir)
        copy_static_tree(worktree_dir / "dist", release_dir)
        manifest = build_manifest(
            release_dir,
            source_git_sha=source_sha,
            build_command=build_command,
            bun_version_value=bun_version(worktree_dir),
            lockfile_path=lockfile.relative_to(repo_dir).as_posix(),
            lockfile_sha256=lockfile_sha,
        )
        write_manifest(release_dir, manifest)
        result = verify_release(
            release_dir,
            expected_source_sha=source_sha,
            repo_dir=repo_dir,
        )
        if not result.ok:
            raise SystemExit("\n".join(result.errors))
        return release_dir
    finally:
        if not keep_worktree:
            remove_worktree(repo_dir, worktree_dir)
            if owned_temp_path is not None and owned_temp_path.exists():
                shutil.rmtree(owned_temp_path, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, default=Path.cwd())
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--source-ref", default=os.environ.get("PORTFOLIO_LAB_RELEASE_REF", "HEAD"))
    parser.add_argument("--build-command", default=DEFAULT_BUILD_COMMAND)
    parser.add_argument("--install-command", default=DEFAULT_INSTALL_COMMAND)
    parser.add_argument("--worktree-dir", type=Path)
    parser.add_argument("--keep-worktree", action="store_true")
    args = parser.parse_args(argv)

    release_dir = build_release(
        repo_dir=args.repo_dir,
        release_dir=args.release_dir,
        source_ref=args.source_ref,
        build_command=args.build_command,
        install_command=args.install_command,
        worktree_dir=args.worktree_dir,
        keep_worktree=args.keep_worktree,
    )
    print(f"built verified lab release: {release_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
