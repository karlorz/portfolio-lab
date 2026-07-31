import importlib.util
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
SOURCE_SHA = "a" * 40
LOCK_SHA = "b" * 64


def _load_script(name: str):
    script_path = SCRIPTS_DIR / f"{name}.py"
    sys.path.insert(0, str(SCRIPTS_DIR))
    spec = importlib.util.spec_from_file_location(name, script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_release_tree(root: Path) -> None:
    (root / "assets").mkdir(parents=True)
    (root / "data").mkdir()
    (root / "index.html").write_text("<main>Portfolio Lab</main>", encoding="utf-8")
    (root / "assets/app.js").write_text("console.log('ok')\n", encoding="utf-8")
    (root / "data/signals.json").write_text('{"mutable":true}\n', encoding="utf-8")


def _write_manifest(root: Path, *, source_sha: str = SOURCE_SHA) -> dict:
    builder = _load_script("build_lab_release")
    manifest = builder.build_manifest(
        root,
        source_git_sha=source_sha,
        build_command="bun run build",
        bun_version_value="1.2.3",
        lockfile_path="bun.lock",
        lockfile_sha256=LOCK_SHA,
        build_time_utc="2026-07-31T00:00:00Z",
    )
    builder.write_manifest(root, manifest)
    return manifest


def test_release_manifest_contains_static_identity_and_excludes_data(tmp_path: Path) -> None:
    _write_release_tree(tmp_path)

    manifest = _write_manifest(tmp_path)

    assert manifest["schema_version"] == "portfolio-lab-static-release/v1"
    assert manifest["source_git_sha"] == SOURCE_SHA
    assert manifest["build_time_utc"] == "2026-07-31T00:00:00Z"
    assert manifest["build_command"] == "bun run build"
    assert manifest["bun_version"] == "1.2.3"
    assert manifest["lockfile"] == {"path": "bun.lock", "sha256": LOCK_SHA}
    assert manifest["policy"]["mutable_data_excluded"] is True
    assert manifest["policy"]["excluded_paths"] == ["data/**"]
    assert {row["path"] for row in manifest["assets"]} == {"assets/app.js", "index.html"}


def test_release_verifier_accepts_matching_manifest_and_ignores_data(tmp_path: Path) -> None:
    verifier = _load_script("verify_lab_release")
    _write_release_tree(tmp_path)
    _write_manifest(tmp_path)

    result = verifier.verify_release(tmp_path, expected_source_sha=SOURCE_SHA)

    assert result.ok is True, result.errors


def test_release_verifier_rejects_digest_mismatch(tmp_path: Path) -> None:
    verifier = _load_script("verify_lab_release")
    _write_release_tree(tmp_path)
    _write_manifest(tmp_path)
    (tmp_path / "assets/app.js").write_text("console.log('changed')\n", encoding="utf-8")

    result = verifier.verify_release(tmp_path, expected_source_sha=SOURCE_SHA)

    assert result.ok is False
    assert "asset digest mismatch: assets/app.js" in result.errors


def test_release_verifier_rejects_missing_and_extra_deployable_assets(tmp_path: Path) -> None:
    verifier = _load_script("verify_lab_release")
    _write_release_tree(tmp_path)
    _write_manifest(tmp_path)
    (tmp_path / "index.html").unlink()
    (tmp_path / "extra.txt").write_text("extra\n", encoding="utf-8")

    result = verifier.verify_release(tmp_path, expected_source_sha=SOURCE_SHA)

    assert result.ok is False
    assert "manifest asset is missing from release tree: index.html" in result.errors
    assert "release tree has unmanifested deployable asset: extra.txt" in result.errors


def test_release_verifier_rejects_source_sha_mismatch(tmp_path: Path) -> None:
    verifier = _load_script("verify_lab_release")
    _write_release_tree(tmp_path)
    _write_manifest(tmp_path)

    result = verifier.verify_release(tmp_path, expected_source_sha="c" * 40)

    assert result.ok is False
    assert any("does not match expected" in error for error in result.errors)


def test_release_verifier_rejects_data_assets_in_manifest(tmp_path: Path) -> None:
    verifier = _load_script("verify_lab_release")
    _write_release_tree(tmp_path)
    manifest = _write_manifest(tmp_path)
    manifest["assets"].append({"path": "data/signals.json", "sha256": "d" * 64, "bytes": 17})
    (tmp_path / "_release.json").write_text(json.dumps(manifest), encoding="utf-8")

    result = verifier.verify_release(tmp_path, expected_source_sha=SOURCE_SHA)

    assert result.ok is False
    assert "manifest must not include mutable data asset: data/signals.json" in result.errors


def test_copy_static_tree_removes_dist_data_from_release_identity(tmp_path: Path) -> None:
    builder = _load_script("build_lab_release")
    dist = tmp_path / "dist"
    release = tmp_path / "release"
    _write_release_tree(dist)

    builder.copy_static_tree(dist, release)

    assert (release / "index.html").is_file()
    assert (release / "assets/app.js").is_file()
    assert not (release / "data").exists()


def test_release_builder_refuses_dirty_git_source(tmp_path: Path) -> None:
    builder = _load_script("build_lab_release")
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, stdout=subprocess.PIPE)
    (tmp_path / "bun.lock").write_text("lock\n", encoding="utf-8")
    (tmp_path / "package.json").write_text("{}\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Portfolio Lab Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-m",
            "init",
        ],
        cwd=tmp_path,
        check=True,
        stdout=subprocess.PIPE,
    )
    (tmp_path / "package.json").write_text('{"dirty":true}\n', encoding="utf-8")

    try:
        builder.ensure_clean_source(tmp_path)
    except SystemExit as exc:
        assert "Refusing to build release from dirty source tree" in str(exc)
    else:
        raise AssertionError("dirty source was accepted")


def test_release_builder_allows_modified_scheduled_runtime_artifacts(tmp_path: Path) -> None:
    builder = _load_script("build_lab_release")
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, stdout=subprocess.PIPE)
    (tmp_path / "bun.lock").write_text("lock\n", encoding="utf-8")
    (tmp_path / "package.json").write_text("{}\n", encoding="utf-8")
    for relative in builder.GENERATED_RUNTIME_SOURCE_PATHS:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"generated": 1}\n', encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, stdout=subprocess.PIPE)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Portfolio Lab Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-m",
            "init",
        ],
        cwd=tmp_path,
        check=True,
        stdout=subprocess.PIPE,
    )

    for relative in builder.GENERATED_RUNTIME_SOURCE_PATHS:
        path = tmp_path / relative
        path.write_text('{"generated": 2}\n', encoding="utf-8")

    builder.ensure_clean_source(tmp_path)
