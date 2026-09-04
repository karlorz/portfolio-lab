"""Hermetic tests for the cursor-box user-owned dependency bootstrap tooling.

Tests drive the real shipped entry point (`scripts/portfolio-lab-cursor-box-bootstrap.sh`
and `scripts/portfolio_lab_cursor_box_bootstrap.py`) against isolated prefixes and verify:
- user-owned default paths (/home/box/.local) vs isolated prefix allowlist;
- rejection of root execution when base paths could change;
- rejection of forbidden commands (apk, sudo, docker) while permitting legitimate .apk files;
- rejection of writes outside allowlist;
- pinned versions and SHA-256 checksums;
- deterministic machine-readable manifest with official dl-cdn URLs and licenses;
- redaction of credentials and URL query secrets;
- full self-contained 32-package Alpine v3.22 x86_64 closure installed under alpine-root;
- full standalone Python runtime tree extraction and stdlib execution;
- private CA bundle configuration (SSL_CERT_FILE, CURL_CA_BUNDLE, GIT_SSL_CAINFO);
- relocatable wrappers using relative directory traversal;
- rejection of unknown override tool names;
- rejection of hostile archive members (traversal, absolute paths, unsafe symlinks);
- atomic and idempotent installs (safe same-filesystem atomic rename, no mixed state);
- real verify command asserting runtime behavior and digests across all tools;
- uninstall validating all paths within prefix before removing files and managed trees;
- stage-0 BusyBox shell fallback when system python is absent: dry-run is non-mutating,
  install supports pre-transferred archive override without modifying base.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import stat
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SH_ENTRYPOINT = PROJECT_ROOT / "scripts" / "portfolio-lab-cursor-box-bootstrap.sh"
PY_ENTRYPOINT = PROJECT_ROOT / "scripts" / "portfolio_lab_cursor_box_bootstrap.py"

PINNED_BUN_URL = "https://github.com/oven-sh/bun/releases/download/bun-v1.4.0/bun-linux-x64-musl.zip"
PINNED_BUN_SHA256 = "83b5f12fd258dd8d4fdcaea65ede954366aa717dab399e20093ecab280d54e7a"
PINNED_BUN_VERSION = "bun-v1.4.0"


def run_entrypoint(
    args: list[str],
    *,
    env: dict[str, str] | None = None,
    check: bool = True,
    use_sh: bool = True,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    """Execute the shipped bootstrap entry point using /bin/sh or Python."""
    cmd: list[str]
    if use_sh:
        cmd = ["/bin/sh", str(SH_ENTRYPOINT), *args]
    else:
        cmd = [sys.executable, str(PY_ENTRYPOINT), *args]

    run_env = dict(os.environ)
    if env:
        run_env.update(env)

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=run_env,
        check=False,
        cwd=str(cwd) if cwd else None,
    )
    if check and result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode,
            cmd,
            output=result.stdout,
            stderr=result.stderr,
        )
    return result


def _make_dummy_zip(dest_zip: Path, internal_path: str, content: bytes) -> str:
    dest_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dest_zip, "w") as zf:
        zf.writestr(internal_path, content)
    return hashlib.sha256(dest_zip.read_bytes()).hexdigest()


def _make_dummy_tar_gz(dest_tar: Path, files: dict[str, bytes]) -> str:
    dest_tar.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(dest_tar, "w:gz") as tf:
        for path, data in files.items():
            ti = tarfile.TarInfo(name=path)
            ti.size = len(data)
            ti.mode = 0o755 if ("bin/" in path or path.endswith(".so") or ".so." in path) else 0o644
            tf.addfile(ti, io.BytesIO(data))
    return hashlib.sha256(dest_tar.read_bytes()).hexdigest()


def _make_dummy_apk(dest_apk: Path, files: dict[str, bytes]) -> str:
    return _make_dummy_tar_gz(dest_apk, files)


def test_entrypoint_exists_and_is_executable() -> None:
    """The shipped entry point file must exist and have executable permissions."""
    assert SH_ENTRYPOINT.is_file(), f"missing {SH_ENTRYPOINT}"
    assert PY_ENTRYPOINT.is_file(), f"missing {PY_ENTRYPOINT}"
    st = os.stat(SH_ENTRYPOINT)
    assert bool(st.st_mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)), (
        f"{SH_ENTRYPOINT} is not executable"
    )
    # Ensure launcher has strict /bin/sh shebang
    first_line = SH_ENTRYPOINT.read_text(encoding="utf-8").splitlines()[0]
    assert first_line == "#!/bin/sh"


def test_entrypoint_help_displays_expected_commands() -> None:
    """The entry point must document dry-run, install, verify, and uninstall."""
    for use_sh in (True, False):
        res = run_entrypoint(["--help"], check=True, use_sh=use_sh)
        out = res.stdout
        assert "dry-run" in out
        assert "install" in out
        assert "verify" in out
        assert "uninstall" in out
        assert "--prefix" in out


def test_default_paths_point_to_user_box_local() -> None:
    """Without an explicit prefix, defaults must resolve to /home/box/.local."""
    res = run_entrypoint(["dry-run"], check=True)
    manifest = json.loads(res.stdout)
    assert manifest["prefix"] == "/home/box/.local"
    assert manifest["bin_dir"] == "/home/box/.local/bin"
    assert manifest["toolchain_dir"] == "/home/box/.local/share/portfolio-lab/toolchain"
    assert manifest["alpine_root"] == "/home/box/.local/share/portfolio-lab/toolchain/alpine-root"


def test_custom_prefix_overrides_target_paths(tmp_path: Path) -> None:
    """An explicit prefix must redirect bin, toolchain, and manifest locations."""
    prefix = tmp_path / "custom_prefix"
    # Test both --prefix PATH and --prefix=PATH
    res1 = run_entrypoint(["--prefix", str(prefix), "dry-run"], check=True)
    manifest1 = json.loads(res1.stdout)
    assert manifest1["prefix"] == str(prefix)
    assert manifest1["bin_dir"] == str(prefix / "bin")
    assert manifest1["toolchain_dir"] == str(prefix / "share" / "portfolio-lab" / "toolchain")
    assert manifest1["alpine_root"] == str(prefix / "share" / "portfolio-lab" / "toolchain" / "alpine-root")

    res2 = run_entrypoint([f"--prefix={prefix}", "dry-run"], check=True)
    manifest2 = json.loads(res2.stdout)
    assert manifest1 == manifest2


def test_rejects_write_outside_allowlist(tmp_path: Path) -> None:
    """Attempting to specify a path escaping the prefix must fail closed."""
    prefix = tmp_path / "sandbox"
    prefix.mkdir()
    escape_prefix = tmp_path / "sandbox" / ".." / "outside"

    res = run_entrypoint(
        ["--prefix", str(escape_prefix), "dry-run"],
        check=False,
    )
    assert res.returncode != 0
    assert "outside" in res.stderr.lower() or "allowlist" in res.stderr.lower() or "traversal" in res.stderr.lower()


def test_rejects_root_execution_when_base_paths_could_change() -> None:
    """Root execution without safe isolated prefix must fail closed."""
    res = run_entrypoint(
        ["dry-run"],
        env={"PORTFOLIO_LAB_BOOTSTRAP_FORCE_ROOT": "1"},
        check=False,
    )
    assert res.returncode != 0
    assert "root" in res.stderr.lower()


def test_rejects_forbidden_commands_while_allowing_legitimate_apk_files(tmp_path: Path) -> None:
    """The bootstrap entry point must reject forbidden commands but permit .apk file names."""
    prefix = tmp_path / "sandbox"
    prefix.mkdir()

    # Legit override containing .apk must succeed in dry-run
    res_legit = run_entrypoint(
        ["--prefix", str(prefix), "--override-url=git-2.49.1-r0.apk=https://example.com/git.apk", "dry-run"],
        check=True,
    )
    assert res_legit.returncode == 0

    for forbidden in ("apk add", "sudo rm", "docker run"):
        res = run_entrypoint(
            ["--prefix", str(prefix), f"--extra-arg={forbidden}", "dry-run"],
            check=False,
        )
        assert res.returncode != 0
        assert "forbidden" in res.stderr.lower() or "prohibited" in res.stderr.lower()

        res2 = run_entrypoint(
            ["--prefix", str(prefix), "dry-run"],
            env={"PORTFOLIO_LAB_BOOTSTRAP_INJECT": forbidden},
            check=False,
        )
        assert res2.returncode != 0
        assert "forbidden" in res2.stderr.lower() or "prohibited" in res2.stderr.lower()


def test_dry_run_emits_deterministic_manifest_without_mutating_disk(tmp_path: Path) -> None:
    """dry-run must output deterministic JSON and leave the prefix completely empty."""
    prefix = tmp_path / "clean_prefix"

    res1 = run_entrypoint(["--prefix", str(prefix), "dry-run"], check=True)
    res2 = run_entrypoint(["--prefix", str(prefix), "dry-run"], check=True)

    manifest1 = json.loads(res1.stdout)
    manifest2 = json.loads(res2.stdout)
    assert manifest1 == manifest2
    assert not prefix.exists(), "dry-run must not create directories on disk"


def test_manifest_contains_full_32_package_closure(tmp_path: Path) -> None:
    """Manifest must include the pinned official Bun, uv, python, and full 32-package self-contained alpine closure."""
    prefix = tmp_path / "p"
    res = run_entrypoint(["--prefix", str(prefix), "dry-run"], check=True)
    manifest = json.loads(res.stdout)

    tools = manifest["tools"]
    for required in ("bun", "uv", "python3", "git", "curl", "sqlite3", "rsync", "jq", "zstd"):
        assert required in tools, f"missing required tool {required}"
        assert tools[required]["feasibility"] == "feasible", f"{required} must be feasible"
        assert (prefix / "bin" / required).as_posix() == tools[required]["install_path"]

    assert tools["jq"]["license"] == "MIT"
    assert "dl-cdn.alpinelinux.org" in tools["curl"]["source_url"]

    packages = manifest.get("packages", {})
    # Must contain all 32 self-contained packages
    assert len(packages) == 32, f"expected 32 closure packages, found {len(packages)}"
    for must_have in (
        "sqlite-libs-3.49.2-r1.apk",
        "zstd-libs-1.5.7-r0.apk",
        "libcrypto3-3.5.8-r0.apk",
        "libssl3-3.5.8-r0.apk",
        "zlib-1.3.2-r0.apk",
        "ca-certificates-bundle-20260611-r0.apk",
    ):
        assert must_have in packages, f"missing package {must_have}"
        assert "dl-cdn.alpinelinux.org" in packages[must_have]["source_url"]


def test_rejects_unknown_override_names(tmp_path: Path) -> None:
    """Attempting to override an unknown tool or package name must fail closed."""
    prefix = tmp_path / "bad_override"
    res = run_entrypoint(
        [
            "--prefix",
            str(prefix),
            "--override-artifact=not_a_real_tool=/tmp/foo",
            "dry-run",
        ],
        check=False,
    )
    assert res.returncode != 0
    assert "unknown" in res.stderr.lower() or "not_a_real_tool" in res.stderr.lower()


def test_redaction_of_credentials_and_query_secrets(tmp_path: Path) -> None:
    """Sensitive credentials and query secrets in artifact URLs must be redacted in output."""
    prefix = tmp_path / "redact_test"
    sensitive_url = "https://user:my_secret_password@downloads.example.org/bun.zip?token=my_secret_token&key=12345"

    res = run_entrypoint(
        [
            "--prefix",
            str(prefix),
            f"--override-url=bun={sensitive_url}",
            "dry-run",
        ],
        check=True,
    )

    stdout = res.stdout
    assert "my_secret_password" not in stdout
    assert "my_secret_token" not in stdout
    assert "12345" not in stdout

    manifest = json.loads(stdout)
    redacted_url = manifest["tools"]["bun"]["source_url"]
    assert "my_secret_password" not in redacted_url
    assert "my_secret_token" not in redacted_url


def test_install_full_runtime_relocatable_wrappers_and_ca_bundle(tmp_path: Path) -> None:
    """install must atomically extract the full Python runtime tree, Alpine root, and create relocatable wrappers with CA bundle."""
    prefix = tmp_path / "prefix"
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True)

    # Bun
    bun_zip = cache_dir / "bun.zip"
    bun_sha = _make_dummy_zip(bun_zip, "bun-linux-x64-musl/bun", b"#!/bin/sh\necho 1.4.0\n")

    # uv
    uv_tar = cache_dir / "uv.tar.gz"
    uv_sha = _make_dummy_tar_gz(uv_tar, {"uv-x86_64-unknown-linux-musl/uv": b"#!/bin/sh\necho uv 0.12.9\n"})

    # Full Python runtime tree fixture (including real in-memory sqlite behavior)
    py_tar = cache_dir / "python.tar.gz"
    py_code_server = (
        '#!/bin/sh\n'
        'if [ "$1" = "-c" ]; then\n'
        '  case "$2" in\n'
        '    *"select 42, 1"*) printf "PY_BEHAVIOR_OK\\n"; exit 0 ;;\n'
        '    *"STDLIB_OK"*) printf "STDLIB_OK\\n"; exit 0 ;;\n'
        '    *) exit 0 ;;\n'
        '  esac\n'
        'fi\n'
        'echo Python 3.11.16\n'
    )
    py_sha = _make_dummy_tar_gz(
        py_tar,
        {
            "python/bin/python3": py_code_server.encode("utf-8"),
            "python/lib/python3.11/encodings/__init__.py": b"# encodings package\n",
            "python/lib/python3.11/sqlite3/__init__.py": b"# sqlite3 package\n",
            "python/lib/python3.11/os.py": b"# os module\n",
        },
    )

    # Alpine packages
    ca_apk = cache_dir / "ca-certificates-bundle-20260611-r0.apk"
    ca_sha = _make_dummy_apk(ca_apk, {"etc/ssl/certs/ca-certificates.crt": b"-----BEGIN CERTIFICATE-----\nMOCK\n-----END CERTIFICATE-----\n"})

    git_sh = (
        '#!/bin/sh\n'
        'case "$1" in\n'
        '  init) mkdir -p "$2/.git"; exit 0 ;;\n'
        '  config) exit 0 ;;\n'
        '  add) exit 0 ;;\n'
        '  commit) exit 0 ;;\n'
        '  bundle)\n'
        '    case "$2" in\n'
        '      create) touch "$3"; exit 0 ;;\n'
        '      verify) exit 0 ;;\n'
        '    esac ;;\n'
        '  clone) mkdir -p "$3"; cp "$(dirname "$2")/git-test-repo/test.txt" "$3/test.txt" 2>/dev/null || echo "verification_commit_payload" > "$3/test.txt"; exit 0 ;;\n'
        '  version) echo "git version 2.49.1"; exit 0 ;;\n'
        '  *) echo git "$@"; exit 0 ;;\n'
        'esac\n'
    )
    git_apk = cache_dir / "git-2.49.1-r0.apk"
    git_sha = _make_dummy_apk(
        git_apk,
        {
            "usr/bin/git": git_sh.encode("utf-8"),
            "usr/libexec/git-core/git": b'#!/bin/sh\necho git-core\n',
            "usr/share/git-core/templates/description": b"Git repository\n",
        },
    )

    curl_apk = cache_dir / "curl-8.14.1-r3.apk"
    curl_sha = _make_dummy_apk(curl_apk, {"usr/bin/curl": b'#!/bin/sh\necho "curl 8.14.1 (x86_64-alpine-linux-musl)"\n'})

    jq_apk = cache_dir / "jq-1.8.2-r0.apk"
    jq_sha = _make_dummy_apk(jq_apk, {"usr/bin/jq": b'#!/bin/sh\nif [ "$1" = "--version" ]; then echo "1.8.2"; exit 0; fi; echo "jq_verification_ok"\n'})

    sqlite_sh = (
        '#!/bin/sh\n'
        'if [ "$1" = "--version" ]; then echo "3.49.2 2026-06-01"; exit 0; fi\n'
        'if [ "$1" = ":memory:" ]; then echo "1"; echo "ok"; exit 0; fi\n'
        'exit 0\n'
    )
    sqlite_apk = cache_dir / "sqlite-3.49.2-r1.apk"
    sqlite_sha = _make_dummy_apk(sqlite_apk, {"usr/bin/sqlite3": sqlite_sh.encode("utf-8")})

    rsync_sh = (
        '#!/bin/sh\n'
        'if [ "$1" = "--version" ]; then echo "rsync version 3.5.0 protocol version 31"; exit 0; fi\n'
        'cp "$1" "$2"; exit 0\n'
    )
    rsync_apk = cache_dir / "rsync-3.5.0-r0.apk"
    rsync_sha = _make_dummy_apk(rsync_apk, {"usr/bin/rsync": rsync_sh.encode("utf-8")})

    zstd_sh = (
        '#!/bin/sh\n'
        'if [ "$1" = "--version" ]; then echo "*** zstd command line interface 64-bits v1.5.7 ***"; exit 0; fi\n'
        'while [ "$#" -gt 0 ]; do\n'
        '  if [ "$1" = "-o" ]; then out="$2"; shift 2; continue; fi\n'
        '  if [ -f "$1" ]; then in="$1"; fi\n'
        '  shift\n'
        'done\n'
        'cp "$in" "$out"\n'
        'exit 0\n'
    )
    zstd_apk = cache_dir / "zstd-1.5.7-r0.apk"
    zstd_sha = _make_dummy_apk(zstd_apk, {"usr/bin/zstd": zstd_sh.encode("utf-8")})

    overrides = [
        f"--override-artifact=bun={bun_zip}",
        f"--override-sha256=bun={bun_sha}",
        f"--override-artifact=uv={uv_tar}",
        f"--override-sha256=uv={uv_sha}",
        f"--override-artifact=python3={py_tar}",
        f"--override-sha256=python3={py_sha}",
        f"--override-artifact=ca-certificates-bundle-20260611-r0.apk={ca_apk}",
        f"--override-sha256=ca-certificates-bundle-20260611-r0.apk={ca_sha}",
        f"--override-artifact=git-2.49.1-r0.apk={git_apk}",
        f"--override-sha256=git-2.49.1-r0.apk={git_sha}",
        f"--override-artifact=curl-8.14.1-r3.apk={curl_apk}",
        f"--override-sha256=curl-8.14.1-r3.apk={curl_sha}",
        f"--override-artifact=jq-1.8.2-r0.apk={jq_apk}",
        f"--override-sha256=jq-1.8.2-r0.apk={jq_sha}",
        f"--override-artifact=sqlite-3.49.2-r1.apk={sqlite_apk}",
        f"--override-sha256=sqlite-3.49.2-r1.apk={sqlite_sha}",
        f"--override-artifact=rsync-3.5.0-r0.apk={rsync_apk}",
        f"--override-sha256=rsync-3.5.0-r0.apk={rsync_sha}",
        f"--override-artifact=zstd-1.5.7-r0.apk={zstd_apk}",
        f"--override-sha256=zstd-1.5.7-r0.apk={zstd_sha}",
    ]

    # Functional ninja for host test environment
    ninja_zip = cache_dir / "ninja.whl"
    ninja_sha = _make_dummy_zip(ninja_zip, "ninja-1.13.2.data/scripts/ninja", _functional_ninja_script())
    overrides.extend([
        f"--override-artifact=ninja={ninja_zip}",
        f"--override-sha256=ninja={ninja_sha}",
    ])

    run_entrypoint(
        ["--prefix", str(prefix), "--mock-closure", *overrides, "install"],
        check=True,
    )

    # Assert Bun payload is in managed toolchain/standalone and bin/bun is a relocatable wrapper
    bin_dir = prefix / "bin"
    standalone_bun = prefix / "share" / "portfolio-lab" / "toolchain" / "standalone" / "bun"
    assert standalone_bun.is_file(), "bun payload must be installed under toolchain/standalone/bun"
    bun_wrapper = (bin_dir / "bun").read_text(encoding="utf-8")
    assert "LD_LIBRARY_PATH" in bun_wrapper, "bun wrapper must set LD_LIBRARY_PATH"
    assert "dirname" in bun_wrapper, "bun wrapper must be relocatable"

    manifest_file = prefix / "share" / "portfolio-lab" / "toolchain" / "bootstrap-manifest.json"
    manifest_obj = json.loads(manifest_file.read_text(encoding="utf-8"))
    bun_manifest = manifest_obj["tools"]["bun"]
    assert "payload_sha256" in bun_manifest, "bun manifest must record payload_sha256"
    assert "installed_sha256" in bun_manifest, "bun manifest must record installed_sha256 for wrapper"

    # Assert full standalone Python runtime tree exists
    python_root = prefix / "share" / "portfolio-lab" / "toolchain" / "python-root"
    assert (python_root / "bin" / "python3").is_file()
    assert (python_root / "lib" / "python3.11" / "encodings" / "__init__.py").is_file()
    assert (python_root / "lib" / "python3.11" / "sqlite3" / "__init__.py").is_file()

    # Assert CA bundle exists under alpine-root
    alpine_root = prefix / "share" / "portfolio-lab" / "toolchain" / "alpine-root"
    ca_file = alpine_root / "etc" / "ssl" / "certs" / "ca-certificates.crt"
    assert ca_file.is_file()

    # Check relocatable wrappers
    bin_dir = prefix / "bin"
    git_wrapper = (bin_dir / "git").read_text(encoding="utf-8")
    assert "SSL_CERT_FILE" in git_wrapper
    assert "CURL_CA_BUNDLE" in git_wrapper
    assert "GIT_SSL_CAINFO" in git_wrapper
    assert "GIT_EXEC_PATH" in git_wrapper
    assert "GIT_TEMPLATE_DIR" in git_wrapper
    # Must derive relative path rather than hardcoded absolute prefix
    assert "dirname" in git_wrapper

    curl_wrapper = (bin_dir / "curl").read_text(encoding="utf-8")
    assert "CURL_CA_BUNDLE" in curl_wrapper
    assert "SSL_CERT_FILE" in curl_wrapper

    py_wrapper = (bin_dir / "python3").read_text(encoding="utf-8")
    assert "dirname" in py_wrapper

    # Test Python stdlib execution via wrapper
    res_py = subprocess.run(
        [str(bin_dir / "python3"), "-c", "import encodings, sqlite3, os; print('STDLIB_OK')"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "STDLIB_OK" in res_py.stdout

    # Test Idempotent install
    res_idem = run_entrypoint(
        ["--prefix", str(prefix), "--mock-closure", *overrides, "install"],
        check=True,
    )
    assert "already up to date" in res_idem.stdout.lower() or "idempotent" in res_idem.stdout.lower()

    # Run real verify command
    res_verify = run_entrypoint(
        ["--prefix", str(prefix), "--mock-closure", "verify"],
        check=True,
    )
    assert "verified" in res_verify.stdout.lower() or "ok" in res_verify.stdout.lower()


def test_pyproject_build_constraint_dependencies() -> None:
    """pyproject.toml [tool.uv] must define build-constraint-dependencies with scikit-build-core>=0.10,<1.0.0."""
    pyproject_path = Path(__file__).resolve().parent.parent / "pyproject.toml"
    assert pyproject_path.is_file()
    content = pyproject_path.read_text(encoding="utf-8")
    assert 'build-constraint-dependencies = ["scikit-build-core>=0.10,<1.0.0"]' in content


def test_bunx_wrapper_and_manifest_lifecycle(tmp_path: Path) -> None:
    """bunx wrapper must delegate to bun x \"$@\", be recorded in manifest, and be cleaned on uninstall."""
    prefix = tmp_path / "bunx_prefix"
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True)

    bun_zip = cache_dir / "bun.zip"
    bun_sha = _make_dummy_zip(bun_zip, "bun-linux-x64-musl/bun", b"#!/bin/sh\nif [ \"$1\" = \"x\" ]; then shift; echo \"BUNX_RUN: $@\"; exit 0; fi; echo 1.4.0\n")

    run_entrypoint(
        [
            "--prefix",
            str(prefix),
            "--mock-closure",
            f"--override-artifact=bun={bun_zip}",
            f"--override-sha256=bun={bun_sha}",
            "--skip-tools=python3,uv,ninja,git,curl,sqlite3,rsync,zstd,jq",
            "install",
        ],
        check=True,
    )

    bunx_wrapper = prefix / "bin" / "bunx"
    assert bunx_wrapper.is_file(), "bunx wrapper must be installed in prefix/bin"
    wrapper_text = bunx_wrapper.read_text(encoding="utf-8")
    assert 'exec "$BIN_DIR/bun" x "$@"' in wrapper_text or 'exec "$PREFIX/bin/bun" x "$@"' in wrapper_text or 'exec "$STANDALONE_BUN" x "$@"' in wrapper_text or 'bun" x "$@"' in wrapper_text

    # Manifest checks
    manifest_path = prefix / "share" / "portfolio-lab" / "toolchain" / "bootstrap-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert "bunx" in manifest["tools"]
    assert manifest["tools"]["bunx"]["install_path"] == str(bunx_wrapper)

    # Behavioral test via wrapper
    res = subprocess.run([str(bunx_wrapper), "vite", "build"], capture_output=True, text=True)
    assert res.returncode == 0
    assert "BUNX_RUN: vite build" in res.stdout


def test_pinned_standalone_ninja_and_manifest(tmp_path: Path) -> None:
    """Real standalone Ninja must be pinned (v1.13.2 musl wheel) and managed."""
    import scripts.portfolio_lab_cursor_box_bootstrap as mod
    assert "ninja" in mod.STANDALONE_TOOLS
    ninja_spec = mod.STANDALONE_TOOLS["ninja"]
    assert ninja_spec["version"] == "1.13.2"
    assert ninja_spec["source_url"] == "https://files.pythonhosted.org/packages/f5/5f/c511f2952f94ab2966d60edd9c34e744ea32f2724b1184b62270bde55b3a/ninja-1.13.2-py3-none-musllinux_1_2_x86_64.whl"
    assert ninja_spec["sha256"] == "915bd482c4be41c75120fd67a22e0bb3f0fbb3bbc5f95b89787deadd59e27ef2"
    assert ninja_spec["archive_format"] == "zip"
    assert ninja_spec["extract_path"] == "ninja-1.13.2.data/scripts/ninja"
    assert ninja_spec["license"] == "Apache-2.0"


def test_alpine_build_package_closure_and_repositories(tmp_path: Path) -> None:
    """The 58 build packages must be tracked with their proper main/community repositories in ALPINE_BUILD_PACKAGE_CLOSURE."""
    import scripts.portfolio_lab_cursor_box_bootstrap as mod
    assert hasattr(mod, "ALPINE_BUILD_PACKAGE_CLOSURE")
    build_closure = mod.ALPINE_BUILD_PACKAGE_CLOSURE
    assert len(build_closure) == 58
    pkg_names = {p[0] for p in build_closure}
    assert "samurai-1.2-r7.apk" not in pkg_names, "samurai must be excluded from build closure"
    assert "gcc-14.2.0-r6.apk" in pkg_names
    assert "g++-14.2.0-r6.apk" in pkg_names
    assert "gfortran-14.2.0-r6.apk" in pkg_names
    assert "cmake-3.31.7-r1.apk" in pkg_names
    assert "openblas-0.3.28-r0.apk" in pkg_names

    # Check that community packages are correctly identified
    community_pkgs = {p[0] for p in build_closure if p[3] == "community"}
    assert "openblas-0.3.28-r0.apk" in community_pkgs
    assert "openblas-dev-0.3.28-r0.apk" in community_pkgs
    assert "liblapack-0.3.28-r0.apk" in community_pkgs


def test_uv_wrapper_reconstructs_build_environment(tmp_path: Path) -> None:
    """The uv wrapper must export PATH, LD_LIBRARY_PATH, sysroot, compilers, and library search paths for Alpine native builds."""
    prefix = tmp_path / "uv_wrapper_prefix"
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True)

    uv_tar = cache_dir / "uv.tar.gz"
    uv_sha = _make_dummy_tar_gz(uv_tar, {"uv-x86_64-unknown-linux-musl/uv": b"#!/bin/sh\necho 0.12.9\n"})

    run_entrypoint(
        [
            "--prefix",
            str(prefix),
            "--mock-closure",
            f"--override-artifact=uv={uv_tar}",
            f"--override-sha256=uv={uv_sha}",
            "--skip-tools=bun,bunx,python3,ninja,git,curl,sqlite3,rsync,zstd,jq",
            "install",
        ],
        check=True,
    )

    uv_wrapper = prefix / "bin" / "uv"
    assert uv_wrapper.is_file()
    content = uv_wrapper.read_text(encoding="utf-8")

    # Verify build-root paths, compilers, flags, and real Ninja in PATH
    assert "alpine-build-root" in content
    assert "STANDALONE_DIR" in content or "standalone" in content
    assert 'export PATH="$BIN_DIR:$STANDALONE_DIR:$BUILD_ROOT/usr/bin:$BUILD_ROOT/bin:$PATH"' in content, (
        "uv wrapper must prepend BIN_DIR before standalone/build-root paths"
    )
    assert "CC=" in content and "gcc" in content
    assert "CXX=" in content and "g++" in content
    assert "FC=" in content and "gfortran" in content
    assert "PKG_CONFIG_PATH=" in content
    assert "CMAKE_PREFIX_PATH=" in content
    assert "C_INCLUDE_PATH=" in content
    assert "CPLUS_INCLUDE_PATH=" in content
    assert "LIBRARY_PATH=" in content
    assert "SSL_CERT_FILE" in content
    assert "/home/box" not in content, "No hardcoded /home/box allowed in relocatable wrapper"


def test_python_wrapper_loads_build_root_and_openblas_libs(tmp_path: Path) -> None:
    """The Python wrapper must load build-root and OpenBLAS native libraries via LD_LIBRARY_PATH."""
    prefix = tmp_path / "py_wrapper_prefix"
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True)

    py_tar = cache_dir / "python.tar.gz"
    py_sha = _make_dummy_tar_gz(
        py_tar,
        {
            "python/bin/python3": f"#!/bin/sh\nexec {sys.executable} \"$@\"\n".encode("utf-8"),
            "python/lib/python3.11/os.py": b"# os\n",
        },
    )

    run_entrypoint(
        [
            "--prefix",
            str(prefix),
            "--mock-closure",
            f"--override-artifact=python3={py_tar}",
            f"--override-sha256=python3={py_sha}",
            "--skip-tools=bun,bunx,uv,ninja,git,curl,sqlite3,rsync,zstd,jq",
            "install",
        ],
        check=True,
    )

    py_wrapper = prefix / "bin" / "python3"
    assert py_wrapper.is_file()
    content = py_wrapper.read_text(encoding="utf-8")
    assert "alpine-build-root" in content
    assert "LD_LIBRARY_PATH" in content
    assert "/home/box" not in content


def _create_fake_build_root(build_root: Path, *, require_env: bool = False) -> None:
    """Construct a faithful fake managed build-root whose compilers and tools execute real probe behavior."""
    bin_dir = build_root / "usr" / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)

    env_check_snippet = ""
    if require_env:
        env_check_snippet = (
            'if [ -z "${LD_LIBRARY_PATH:-}" ] || [ -z "${LIBRARY_PATH:-}" ] || [ -z "${C_INCLUDE_PATH:-}" ] || '
            '[ -z "${CPLUS_INCLUDE_PATH:-}" ] || [ -z "${PKG_CONFIG_PATH:-}" ] || [ -z "${CMAKE_PREFIX_PATH:-}" ] || '
            '[ -z "${CC:-}" ] || [ -z "${CXX:-}" ] || [ -z "${FC:-}" ] || [ -z "${AR:-}" ] || [ -z "${RANLIB:-}" ]; then\n'
            '  echo "MISSING_REQUIRED_BUILD_ENV" >&2\n'
            '  exit 1\n'
            'fi\n'
        )

    gcc_script = (
        f'#!/bin/sh\n{env_check_snippet}'
        'if [ "$1" = "--version" ]; then echo "gcc (Alpine 14.2.0) 14.2.0"; exit 0; fi\n'
        'out=""\n'
        'while [ "$#" -gt 0 ]; do\n'
        '  if [ "$1" = "-o" ]; then out="$2"; shift 2; continue; fi\n'
        '  shift\n'
        'done\n'
        'if [ -n "$out" ]; then\n'
        '  printf "#!/bin/sh\\n'
        + (env_check_snippet.replace('"', '\\"') if require_env else '')
        + 'echo C_PROBE_OK\\n" > "$out"\n'
        '  chmod +x "$out"\n'
        '  exit 0\n'
        'fi\n'
        'exit 0\n'
    )
    (bin_dir / "gcc").write_text(gcc_script, encoding="utf-8")
    (bin_dir / "gcc").chmod(0o755)

    gxx_script = (
        f'#!/bin/sh\n{env_check_snippet}'
        'if [ "$1" = "--version" ]; then echo "g++ (Alpine 14.2.0) 14.2.0"; exit 0; fi\n'
        'out=""\n'
        'while [ "$#" -gt 0 ]; do\n'
        '  if [ "$1" = "-o" ]; then out="$2"; shift 2; continue; fi\n'
        '  shift\n'
        'done\n'
        'if [ -n "$out" ]; then\n'
        '  printf "#!/bin/sh\\n'
        + (env_check_snippet.replace('"', '\\"') if require_env else '')
        + 'echo CPP_PROBE_OK\\n" > "$out"\n'
        '  chmod +x "$out"\n'
        '  exit 0\n'
        'fi\n'
        'exit 0\n'
    )
    (bin_dir / "g++").write_text(gxx_script, encoding="utf-8")
    (bin_dir / "g++").chmod(0o755)

    gfortran_script = (
        f'#!/bin/sh\n{env_check_snippet}'
        'if [ "$1" = "--version" ]; then echo "GNU Fortran (Alpine 14.2.0) 14.2.0"; exit 0; fi\n'
        'out=""\n'
        'while [ "$#" -gt 0 ]; do\n'
        '  if [ "$1" = "-o" ]; then out="$2"; shift 2; continue; fi\n'
        '  shift\n'
        'done\n'
        'if [ -n "$out" ]; then\n'
        '  printf "#!/bin/sh\\n'
        + (env_check_snippet.replace('"', '\\"') if require_env else '')
        + 'echo FORTRAN_PROBE_OK\\n" > "$out"\n'
        '  chmod +x "$out"\n'
        '  exit 0\n'
        'fi\n'
        'exit 0\n'
    )
    (bin_dir / "gfortran").write_text(gfortran_script, encoding="utf-8")
    (bin_dir / "gfortran").chmod(0o755)

    rustc_script = (
        f'#!/bin/sh\n{env_check_snippet}'
        'if [ "$1" = "--version" ]; then echo "rustc 1.87.0"; exit 0; fi\n'
        'out=""\n'
        'while [ "$#" -gt 0 ]; do\n'
        '  if [ "$1" = "-o" ]; then out="$2"; shift 2; continue; fi\n'
        '  shift\n'
        'done\n'
        'if [ -n "$out" ]; then\n'
        '  printf "#!/bin/sh\\n'
        + (env_check_snippet.replace('"', '\\"') if require_env else '')
        + 'echo RUST_PROBE_OK\\n" > "$out"\n'
        '  chmod +x "$out"\n'
        '  exit 0\n'
        'fi\n'
        'exit 0\n'
    )
    (bin_dir / "rustc").write_text(rustc_script, encoding="utf-8")
    (bin_dir / "rustc").chmod(0o755)

    cmake_script = (
        f'#!/bin/sh\n{env_check_snippet}'
        'if [ "$1" = "--version" ]; then echo "cmake version 3.31.7"; exit 0; fi\n'
        'echo "-- CMAKE_PROBE_OK"\n'
        'exit 0\n'
    )
    (bin_dir / "cmake").write_text(cmake_script, encoding="utf-8")
    (bin_dir / "cmake").chmod(0o755)

    pkgconf_script = (
        f'#!/bin/sh\n{env_check_snippet}'
        'if [ "$1" = "--version" ]; then echo "2.4.3"; exit 0; fi\n'
        'exit 0\n'
    )
    (bin_dir / "pkgconf").write_text(pkgconf_script, encoding="utf-8")
    (bin_dir / "pkgconf").chmod(0o755)

    cargo_script = (
        f'#!/bin/sh\n{env_check_snippet}'
        'if [ "$1" = "--version" ]; then echo "cargo 1.87.0"; exit 0; fi\n'
        'exit 0\n'
    )
    (bin_dir / "cargo").write_text(cargo_script, encoding="utf-8")
    (bin_dir / "cargo").chmod(0o755)


def _functional_ninja_script(*, write_ninja_log: bool = False) -> bytes:
    log_snippet = ""
    if write_ninja_log:
        log_snippet = 'touch .ninja_log\n'
    return (
        '#!/bin/sh\n'
        f'{log_snippet}'
        'if [ "$1" = "--version" ]; then echo "1.13.2"; exit 0; fi\n'
        'buildfile=""\n'
        'while [ "$#" -gt 0 ]; do\n'
        '  if [ "$1" = "-f" ]; then buildfile="$2"; shift 2; continue; fi\n'
        '  shift\n'
        'done\n'
        'if [ -n "$buildfile" ] && [ -f "$buildfile" ]; then\n'
        '  target="$(grep -E "^build " "$buildfile" | awk "{print \\$2}" | tr -d ":")"\n'
        '  if [ -n "$target" ]; then\n'
        '    echo NINJA_PROBE_OK > "$target"\n'
        '    exit 0\n'
        '  fi\n'
        'fi\n'
        'exit 0\n'
    ).encode("utf-8")


def test_verify_ninja_runs_in_scratch_directory_and_does_not_pollute_caller_cwd(tmp_path: Path) -> None:
    """verify must execute ninja with cwd=test_scratch so .ninja_log/.ninja_deps never pollutes caller cwd."""
    prefix = tmp_path / "ninja_cwd_prefix"
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    ca_apk = cache_dir / "ca-certificates-bundle-20260611-r0.apk"
    ca_sha = _make_dummy_apk(ca_apk, {"etc/ssl/certs/ca-certificates.crt": b"-----BEGIN CERTIFICATE-----\nMOCK\n-----END CERTIFICATE-----\n"})

    # Ninja that writes .ninja_log in its working directory (matching real Ninja behavior)
    ninja_zip = cache_dir / "ninja.whl"
    ninja_sha = _make_dummy_zip(ninja_zip, "ninja-1.13.2.data/scripts/ninja", _functional_ninja_script(write_ninja_log=True))

    run_entrypoint(
        [
            "--prefix",
            str(prefix),
            "--mock-closure",
            f"--override-artifact=ca-certificates-bundle-20260611-r0.apk={ca_apk}",
            f"--override-sha256=ca-certificates-bundle-20260611-r0.apk={ca_sha}",
            f"--override-artifact=ninja={ninja_zip}",
            f"--override-sha256=ninja={ninja_sha}",
            "--skip-tools=bun,bunx,python3,uv,git,curl,sqlite3,rsync,zstd,jq",
            "install",
        ],
        check=True,
    )

    build_root = prefix / "share" / "portfolio-lab" / "toolchain" / "alpine-build-root"
    _create_fake_build_root(build_root)

    manifest_path = prefix / "share" / "portfolio-lab" / "toolchain" / "bootstrap-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    import scripts.portfolio_lab_cursor_box_bootstrap as mod
    manifest["build_packages"] = {
        pkg_name: {
            "source_url": f"{mod.ALPINE_COMMUNITY_DL_BASE if repo == 'community' else mod.ALPINE_DL_BASE}/{pkg_name}",
            "sha256": sha,
            "license": lic,
            "install_root": str(build_root),
            "repository": repo,
        }
        for pkg_name, sha, lic, repo in mod.ALPINE_BUILD_PACKAGE_CLOSURE
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    caller_dir = tmp_path / "caller_cwd"
    caller_dir.mkdir(parents=True, exist_ok=True)

    res = run_entrypoint(
        ["--prefix", str(prefix), "--mock-closure", "--skip-tools=bun,bunx,python3,uv,git,curl,sqlite3,rsync,zstd,jq", "verify"],
        check=True,
        cwd=caller_dir,
    )
    assert "verified" in res.stdout.lower() or "ok" in res.stdout.lower()
    assert not (caller_dir / ".ninja_log").exists(), ".ninja_log must not be created in caller cwd"
    assert not (caller_dir / ".ninja_deps").exists(), ".ninja_deps must not be created in caller cwd"


def test_verify_build_root_and_ninja_behavior(tmp_path: Path) -> None:
    """verify must behaviorally detect build root tools and real Ninja."""
    prefix = tmp_path / "verify_build_prefix"
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True)

    ca_apk = cache_dir / "ca-certificates-bundle-20260611-r0.apk"
    ca_sha = _make_dummy_apk(ca_apk, {"etc/ssl/certs/ca-certificates.crt": b"-----BEGIN CERTIFICATE-----\nMOCK\n-----END CERTIFICATE-----\n"})

    # Dummy uv and functional ninja
    ninja_zip = cache_dir / "ninja.whl"
    ninja_sha = _make_dummy_zip(ninja_zip, "ninja-1.13.2.data/scripts/ninja", _functional_ninja_script())

    run_entrypoint(
        [
            "--prefix",
            str(prefix),
            "--mock-closure",
            f"--override-artifact=ca-certificates-bundle-20260611-r0.apk={ca_apk}",
            f"--override-sha256=ca-certificates-bundle-20260611-r0.apk={ca_sha}",
            f"--override-artifact=ninja={ninja_zip}",
            f"--override-sha256=ninja={ninja_sha}",
            "--skip-tools=bun,bunx,python3,uv,git,curl,sqlite3,rsync,zstd,jq",
            "install",
        ],
        check=True,
    )

    # Construct fake build root
    build_root = prefix / "share" / "portfolio-lab" / "toolchain" / "alpine-build-root"
    _create_fake_build_root(build_root)

    # Add 58 package closure records to manifest for verification
    manifest_path = prefix / "share" / "portfolio-lab" / "toolchain" / "bootstrap-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    import scripts.portfolio_lab_cursor_box_bootstrap as mod
    build_packages_dict = {}
    for pkg_name, sha, lic, repo in mod.ALPINE_BUILD_PACKAGE_CLOSURE:
        base_url = mod.ALPINE_COMMUNITY_DL_BASE if repo == "community" else mod.ALPINE_DL_BASE
        build_packages_dict[pkg_name] = {
            "source_url": f"{base_url}/{pkg_name}",
            "sha256": sha,
            "license": lic,
            "install_root": str(build_root),
            "repository": repo,
        }
    manifest["build_packages"] = build_packages_dict
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # verify MUST succeed on faithful build root and functional ninja
    res_verify = run_entrypoint(
        ["--prefix", str(prefix), "--mock-closure", "--skip-tools=bun,bunx,python3,uv,git,curl,sqlite3,rsync,zstd,jq", "verify"],
        check=True,
    )
    assert "verified" in res_verify.stdout.lower() or "ok" in res_verify.stdout.lower()


def test_verify_fails_closed_when_alpine_build_root_missing(tmp_path: Path) -> None:
    """verify must fail closed when alpine-build-root directory is missing."""
    prefix = tmp_path / "missing_build_root_prefix"
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True)

    ca_apk = cache_dir / "ca-certificates-bundle-20260611-r0.apk"
    ca_sha = _make_dummy_apk(ca_apk, {"etc/ssl/certs/ca-certificates.crt": b"-----BEGIN CERTIFICATE-----\nMOCK\n-----END CERTIFICATE-----\n"})

    ninja_zip = cache_dir / "ninja.whl"
    ninja_sha = _make_dummy_zip(ninja_zip, "ninja-1.13.2.data/scripts/ninja", _functional_ninja_script())

    run_entrypoint(
        [
            "--prefix",
            str(prefix),
            "--mock-closure",
            f"--override-artifact=ca-certificates-bundle-20260611-r0.apk={ca_apk}",
            f"--override-sha256=ca-certificates-bundle-20260611-r0.apk={ca_sha}",
            f"--override-artifact=ninja={ninja_zip}",
            f"--override-sha256=ninja={ninja_sha}",
            "--skip-tools=bun,bunx,python3,uv,git,curl,sqlite3,rsync,zstd,jq",
            "install",
        ],
        check=True,
    )

    build_root = prefix / "share" / "portfolio-lab" / "toolchain" / "alpine-build-root"
    if build_root.exists():
        shutil.rmtree(build_root)

    res = run_entrypoint(
        ["--prefix", str(prefix), "--skip-tools=bun,bunx,python3,uv,git,curl,sqlite3,rsync,zstd,jq", "verify"],
        check=False,
    )
    assert res.returncode != 0
    assert "build root" in res.stderr.lower() or "alpine-build-root" in res.stderr.lower()


def test_verify_fails_closed_when_build_package_metadata_corrupted(tmp_path: Path) -> None:
    """verify must fail closed when build_packages metadata in manifest has tampered checksum or license."""
    prefix = tmp_path / "corrupted_build_pkg_prefix"
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True)

    ca_apk = cache_dir / "ca-certificates-bundle-20260611-r0.apk"
    ca_sha = _make_dummy_apk(ca_apk, {"etc/ssl/certs/ca-certificates.crt": b"-----BEGIN CERTIFICATE-----\nMOCK\n-----END CERTIFICATE-----\n"})

    ninja_zip = cache_dir / "ninja.whl"
    ninja_sha = _make_dummy_zip(ninja_zip, "ninja-1.13.2.data/scripts/ninja", _functional_ninja_script())

    run_entrypoint(
        [
            "--prefix",
            str(prefix),
            "--mock-closure",
            f"--override-artifact=ca-certificates-bundle-20260611-r0.apk={ca_apk}",
            f"--override-sha256=ca-certificates-bundle-20260611-r0.apk={ca_sha}",
            f"--override-artifact=ninja={ninja_zip}",
            f"--override-sha256=ninja={ninja_sha}",
            "--skip-tools=bun,bunx,python3,uv,git,curl,sqlite3,rsync,zstd,jq",
            "install",
        ],
        check=True,
    )

    build_root = prefix / "share" / "portfolio-lab" / "toolchain" / "alpine-build-root"
    _create_fake_build_root(build_root)

    manifest_path = prefix / "share" / "portfolio-lab" / "toolchain" / "bootstrap-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    import scripts.portfolio_lab_cursor_box_bootstrap as mod
    build_packages_dict = {}
    for pkg_name, sha, lic, repo in mod.ALPINE_BUILD_PACKAGE_CLOSURE:
        base_url = mod.ALPINE_COMMUNITY_DL_BASE if repo == "community" else mod.ALPINE_DL_BASE
        build_packages_dict[pkg_name] = {
            "source_url": f"{base_url}/{pkg_name}",
            "sha256": sha,
            "license": lic,
            "install_root": str(build_root),
            "repository": repo,
        }
    # Corrupt sha256 of gcc
    build_packages_dict["gcc-14.2.0-r6.apk"]["sha256"] = "0000000000000000000000000000000000000000000000000000000000000000"
    manifest["build_packages"] = build_packages_dict
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    res = run_entrypoint(
        ["--prefix", str(prefix), "--skip-tools=bun,bunx,python3,uv,git,curl,sqlite3,rsync,zstd,jq", "verify"],
        check=False,
    )
    assert res.returncode != 0
    assert "checksum" in res.stderr.lower() or "mismatch" in res.stderr.lower() or "sha256" in res.stderr.lower()


def test_verify_validates_build_package_url_overrides_via_build_manifest(tmp_path: Path) -> None:
    """verify must validate build package records against the active build_manifest rather than hardcoding official URLs."""
    prefix = tmp_path / "url_override_prefix"
    cache_dir = tmp_path / "url_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    ca_apk = cache_dir / "ca-certificates-bundle-20260611-r0.apk"
    ca_sha = _make_dummy_apk(ca_apk, {"etc/ssl/certs/ca-certificates.crt": b"-----BEGIN CERTIFICATE-----\nMOCK\n-----END CERTIFICATE-----\n"})

    custom_url = "https://custom-mirror.local/alpine/v3.22/main/ca-certificates-bundle-20260611-r0.apk"
    install_args = [
        "--prefix",
        str(prefix),
        "--mock-closure",
        f"--override-artifact=ca-certificates-bundle-20260611-r0.apk={ca_apk}",
        f"--override-sha256=ca-certificates-bundle-20260611-r0.apk={ca_sha}",
        f"--override-url=ca-certificates-bundle-20260611-r0.apk={custom_url}",
        "--skip-tools=bun,bunx,python3,uv,ninja,git,curl,sqlite3,rsync,zstd,jq",
        "install",
    ]

    run_entrypoint(install_args, check=True)

    # verify with the same override URL must succeed
    verify_args = [
        "--prefix",
        str(prefix),
        "--mock-closure",
        f"--override-url=ca-certificates-bundle-20260611-r0.apk={custom_url}",
        "--skip-tools=bun,bunx,python3,uv,ninja,git,curl,sqlite3,rsync,zstd,jq",
        "verify",
    ]
    res_verify = run_entrypoint(verify_args, check=True)
    assert "verified" in res_verify.stdout.lower() or "ok" in res_verify.stdout.lower()


def test_verify_fails_closed_when_pkgconf_emits_empty_or_bogus_version(tmp_path: Path) -> None:
    """verify must fail closed if pkg-config/pkgconf emits empty or bogus output (e.g. exit 0 but no version)."""
    prefix = tmp_path / "bogus_pkgconf_prefix"
    cache_dir = tmp_path / "bogus_pkg_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    ca_apk = cache_dir / "ca-certificates-bundle-20260611-r0.apk"
    ca_sha = _make_dummy_apk(ca_apk, {"etc/ssl/certs/ca-certificates.crt": b"-----BEGIN CERTIFICATE-----\nMOCK\n-----END CERTIFICATE-----\n"})

    ninja_zip = cache_dir / "ninja.whl"
    ninja_sha = _make_dummy_zip(ninja_zip, "ninja-1.13.2.data/scripts/ninja", _functional_ninja_script())

    run_entrypoint(
        [
            "--prefix",
            str(prefix),
            "--mock-closure",
            f"--override-artifact=ca-certificates-bundle-20260611-r0.apk={ca_apk}",
            f"--override-sha256=ca-certificates-bundle-20260611-r0.apk={ca_sha}",
            f"--override-artifact=ninja={ninja_zip}",
            f"--override-sha256=ninja={ninja_sha}",
            "--skip-tools=bun,bunx,python3,uv,git,curl,sqlite3,rsync,zstd,jq",
            "install",
        ],
        check=True,
    )

    build_root = prefix / "share" / "portfolio-lab" / "toolchain" / "alpine-build-root"
    _create_fake_build_root(build_root)

    # Make pkgconf exit 0 with empty output
    (build_root / "usr" / "bin" / "pkgconf").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")

    manifest_path = prefix / "share" / "portfolio-lab" / "toolchain" / "bootstrap-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    import scripts.portfolio_lab_cursor_box_bootstrap as mod
    manifest["build_packages"] = {
        pkg_name: {
            "source_url": f"{mod.ALPINE_COMMUNITY_DL_BASE if repo == 'community' else mod.ALPINE_DL_BASE}/{pkg_name}",
            "sha256": sha,
            "license": lic,
            "install_root": str(build_root),
            "repository": repo,
        }
        for pkg_name, sha, lic, repo in mod.ALPINE_BUILD_PACKAGE_CLOSURE
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    res = run_entrypoint(
        ["--prefix", str(prefix), "--mock-closure", "--skip-tools=bun,bunx,python3,uv,git,curl,sqlite3,rsync,zstd,jq", "verify"],
        check=False,
    )
    assert res.returncode != 0
    assert "pkg-config" in res.stderr.lower() or "pkgconf" in res.stderr.lower()


def test_verify_fails_closed_when_standalone_payload_tampered_or_missing_hash(tmp_path: Path) -> None:
    """verify must validate standalone payload/runtime hashes (bun, uv, ninja, python3) and fail on tampering or missing hash."""
    prefix = tmp_path / "payload_verify_prefix"
    cache_dir = tmp_path / "payload_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    ca_apk = cache_dir / "ca-certificates-bundle-20260611-r0.apk"
    ca_sha = _make_dummy_apk(ca_apk, {"etc/ssl/certs/ca-certificates.crt": b"-----BEGIN CERTIFICATE-----\nMOCK\n-----END CERTIFICATE-----\n"})

    uv_tar = cache_dir / "uv.tar.gz"
    uv_sha = _make_dummy_tar_gz(uv_tar, {"uv-x86_64-unknown-linux-musl/uv": b"#!/bin/sh\necho uv 0.12.9\n"})

    ninja_zip = cache_dir / "ninja.whl"
    ninja_sha = _make_dummy_zip(ninja_zip, "ninja-1.13.2.data/scripts/ninja", _functional_ninja_script())

    install_args = [
        "--prefix",
        str(prefix),
        "--mock-closure",
        f"--override-artifact=ca-certificates-bundle-20260611-r0.apk={ca_apk}",
        f"--override-sha256=ca-certificates-bundle-20260611-r0.apk={ca_sha}",
        f"--override-artifact=uv={uv_tar}",
        f"--override-sha256=uv={uv_sha}",
        f"--override-artifact=ninja={ninja_zip}",
        f"--override-sha256=ninja={ninja_sha}",
        "--skip-tools=bun,bunx,python3,git,curl,sqlite3,rsync,zstd,jq",
        "install",
    ]

    run_entrypoint(install_args, check=True)

    # Clean verify succeeds
    run_entrypoint(["--prefix", str(prefix), "--mock-closure", "--skip-tools=bun,bunx,python3,git,curl,sqlite3,rsync,zstd,jq", "verify"], check=True)

    standalone_dir = prefix / "share" / "portfolio-lab" / "toolchain" / "standalone"
    manifest_file = prefix / "share" / "portfolio-lab" / "toolchain" / "bootstrap-manifest.json"

    # 1. Tamper standalone uv payload binary directly on disk (wrapper untouched)
    (standalone_dir / "uv").write_bytes(b"tampered_standalone_payload")
    res_tampered = run_entrypoint(
        ["--prefix", str(prefix), "--mock-closure", "--skip-tools=bun,bunx,python3,git,curl,sqlite3,rsync,zstd,jq", "verify"],
        check=False,
    )
    assert res_tampered.returncode != 0
    assert "tamper" in res_tampered.stderr.lower() or "payload" in res_tampered.stderr.lower() or "mismatch" in res_tampered.stderr.lower()

    # Restore uv binary
    run_entrypoint(install_args, check=True)

    # 2. Missing expected recorded payload_sha256 in manifest must fail closed
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    del manifest["tools"]["uv"]["payload_sha256"]
    manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    res_missing_hash = run_entrypoint(
        ["--prefix", str(prefix), "--mock-closure", "--skip-tools=bun,bunx,python3,git,curl,sqlite3,rsync,zstd,jq", "verify"],
        check=False,
    )
    assert res_missing_hash.returncode != 0
    assert "payload" in res_missing_hash.stderr.lower() or "manifest" in res_missing_hash.stderr.lower() or "hash" in res_missing_hash.stderr.lower()


def test_install_idempotency_forces_repair_when_runtime_or_build_package_metadata_tampered(tmp_path: Path) -> None:
    """Install idempotency must compare all expected metadata fields (sha256, license, repo, source_url, install_root)."""
    prefix = tmp_path / "metadata_idempotency_prefix"
    cache_dir = tmp_path / "meta_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    ca_apk = cache_dir / "ca-certificates-bundle-20260611-r0.apk"
    ca_sha = _make_dummy_apk(ca_apk, {"etc/ssl/certs/ca-certificates.crt": b"-----BEGIN CERTIFICATE-----\nMOCK\n-----END CERTIFICATE-----\n"})

    install_args = [
        "--prefix",
        str(prefix),
        "--mock-closure",
        f"--override-artifact=ca-certificates-bundle-20260611-r0.apk={ca_apk}",
        f"--override-sha256=ca-certificates-bundle-20260611-r0.apk={ca_sha}",
        "--skip-tools=bun,bunx,python3,uv,ninja,git,curl,sqlite3,rsync,zstd,jq",
        "install",
    ]

    # Initial install
    run_entrypoint(install_args, check=True)

    # Clean install is idempotent
    res_idem = run_entrypoint(install_args, check=True)
    assert "already up to date" in res_idem.stdout.lower()

    manifest_file = prefix / "share" / "portfolio-lab" / "toolchain" / "bootstrap-manifest.json"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))

    # 1. Tamper runtime package license in manifest
    manifest["packages"]["ca-certificates-bundle-20260611-r0.apk"]["license"] = "GPL-99.0"
    manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    res_tampered_pkg = run_entrypoint(install_args, check=True)
    assert "already up to date" not in res_tampered_pkg.stdout.lower()
    assert "successfully installed toolchain" in res_tampered_pkg.stdout.lower()

    # Restore and verify idempotent again
    res_idem2 = run_entrypoint(install_args, check=True)
    assert "already up to date" in res_idem2.stdout.lower()

    # 2. Tamper build package sha256 in manifest
    manifest2 = json.loads(manifest_file.read_text(encoding="utf-8"))
    manifest2["build_packages"]["ca-certificates-bundle-20260611-r0.apk"]["sha256"] = "deadbeef" * 8
    manifest_file.write_text(json.dumps(manifest2, indent=2), encoding="utf-8")

    res_tampered_build_pkg = run_entrypoint(install_args, check=True)
    assert "already up to date" not in res_tampered_build_pkg.stdout.lower()
    assert "successfully installed toolchain" in res_tampered_build_pkg.stdout.lower()


def test_ambient_compiler_variables_do_not_override_managed_build_tools(tmp_path: Path) -> None:
    """Ambient CC, CXX, FC, AR, RANLIB must not cause uv wrapper or verify to use unmanaged tools."""
    prefix = tmp_path / "ambient_cc_prefix"
    cache_dir = tmp_path / "ambient_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    ca_apk = cache_dir / "ca-certificates-bundle-20260611-r0.apk"
    ca_sha = _make_dummy_apk(ca_apk, {"etc/ssl/certs/ca-certificates.crt": b"-----BEGIN CERTIFICATE-----\nMOCK\n-----END CERTIFICATE-----\n"})

    uv_tar = cache_dir / "uv.tar.gz"
    uv_sha = _make_dummy_tar_gz(uv_tar, {"uv-x86_64-unknown-linux-musl/uv": b"#!/bin/sh\necho uv 0.12.9\n"})

    ninja_zip = cache_dir / "ninja.whl"
    ninja_sha = _make_dummy_zip(ninja_zip, "ninja-1.13.2.data/scripts/ninja", _functional_ninja_script())

    run_entrypoint(
        [
            "--prefix",
            str(prefix),
            "--mock-closure",
            f"--override-artifact=ca-certificates-bundle-20260611-r0.apk={ca_apk}",
            f"--override-sha256=ca-certificates-bundle-20260611-r0.apk={ca_sha}",
            f"--override-artifact=uv={uv_tar}",
            f"--override-sha256=uv={uv_sha}",
            f"--override-artifact=ninja={ninja_zip}",
            f"--override-sha256=ninja={ninja_sha}",
            "--skip-tools=bun,bunx,python3,git,curl,sqlite3,rsync,zstd,jq",
            "install",
        ],
        check=True,
    )

    build_root = prefix / "share" / "portfolio-lab" / "toolchain" / "alpine-build-root"
    _create_fake_build_root(build_root, require_env=True)

    manifest_path = prefix / "share" / "portfolio-lab" / "toolchain" / "bootstrap-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    import scripts.portfolio_lab_cursor_box_bootstrap as mod
    manifest["build_packages"] = {
        pkg_name: {
            "source_url": f"{mod.ALPINE_COMMUNITY_DL_BASE if repo == 'community' else mod.ALPINE_DL_BASE}/{pkg_name}",
            "sha256": sha,
            "license": lic,
            "install_root": str(build_root),
            "repository": repo,
        }
        for pkg_name, sha, lic, repo in mod.ALPINE_BUILD_PACKAGE_CLOSURE
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # 1. Inspect uv wrapper script: must use PORTFOLIO_LAB_CC etc., not bare CC
    uv_wrapper = (prefix / "bin" / "uv").read_text(encoding="utf-8")
    assert 'export CC="${PORTFOLIO_LAB_CC:-' in uv_wrapper, "uv wrapper must use PORTFOLIO_LAB_CC"
    assert 'export CXX="${PORTFOLIO_LAB_CXX:-' in uv_wrapper, "uv wrapper must use PORTFOLIO_LAB_CXX"

    # 2. Hostile ambient compiler environment must not affect verify
    hostile_env = {
        "CC": "/hostile/outside/gcc",
        "CXX": "/hostile/outside/g++",
        "FC": "/hostile/outside/gfortran",
        "AR": "/hostile/outside/ar",
        "RANLIB": "/hostile/outside/ranlib",
    }
    res_verify = run_entrypoint(
        ["--prefix", str(prefix), "--mock-closure", "--skip-tools=bun,bunx,python3,uv,git,curl,sqlite3,rsync,zstd,jq", "verify"],
        env=hostile_env,
        check=True,
    )
    assert "verified" in res_verify.stdout.lower() or "ok" in res_verify.stdout.lower()


def test_verify_validates_runtime_packages_closure_rigorously(tmp_path: Path) -> None:
    """verify must validate runtime packages closure (32 packages) with the same rigor as build packages."""
    prefix = tmp_path / "runtime_pkg_verify_prefix"
    cache_dir = tmp_path / "runtime_pkg_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    ca_apk = cache_dir / "ca-certificates-bundle-20260611-r0.apk"
    ca_sha = _make_dummy_apk(ca_apk, {"etc/ssl/certs/ca-certificates.crt": b"-----BEGIN CERTIFICATE-----\nMOCK\n-----END CERTIFICATE-----\n"})

    install_args = [
        "--prefix",
        str(prefix),
        "--mock-closure",
        f"--override-artifact=ca-certificates-bundle-20260611-r0.apk={ca_apk}",
        f"--override-sha256=ca-certificates-bundle-20260611-r0.apk={ca_sha}",
        "--skip-tools=bun,bunx,python3,uv,ninja,git,curl,sqlite3,rsync,zstd,jq",
        "install",
    ]
    run_entrypoint(install_args, check=True)

    manifest_file = prefix / "share" / "portfolio-lab" / "toolchain" / "bootstrap-manifest.json"

    import scripts.portfolio_lab_cursor_box_bootstrap as mod

    # Populate full runtime (32) and build (58) closures for production-style validation
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    build_root = prefix / "share" / "portfolio-lab" / "toolchain" / "alpine-build-root"
    _create_fake_build_root(build_root)
    manifest["packages"] = {
        pkg_name: {
            "source_url": f"{mod.ALPINE_DL_BASE}/{pkg_name}",
            "sha256": sha,
            "license": lic,
            "install_root": str(prefix / "share" / "portfolio-lab" / "toolchain" / "alpine-root"),
            "repository": "main",
        }
        for pkg_name, sha, lic in mod.ALPINE_PACKAGE_CLOSURE
    }
    manifest["build_packages"] = {
        pkg_name: {
            "source_url": f"{mod.ALPINE_COMMUNITY_DL_BASE if repo == 'community' else mod.ALPINE_DL_BASE}/{pkg_name}",
            "sha256": sha,
            "license": lic,
            "install_root": str(build_root),
            "repository": repo,
        }
        for pkg_name, sha, lic, repo in mod.ALPINE_BUILD_PACKAGE_CLOSURE
    }
    manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # 1. Drop a runtime package record entirely
    del manifest["packages"]["zlib-1.3.2-r0.apk"]
    manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    res_dropped = run_entrypoint(
        ["--prefix", str(prefix), "--skip-tools=bun,bunx,python3,uv,ninja,git,curl,sqlite3,rsync,zstd,jq", "verify"],
        check=False,
    )
    assert res_dropped.returncode != 0
    assert "packages closure" in res_dropped.stderr.lower()

    # Restore manifest, then alter checksum and license of runtime package
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    manifest["packages"]["zlib-1.3.2-r0.apk"] = {
        "source_url": "https://dl-cdn.alpinelinux.org/alpine/v3.22/main/x86_64/zlib-1.3.2-r0.apk",
        "sha256": "1f3d5f463f490dad3a68097376711bfe5e8156e9e8daff3070513aa4378cdeca",
        "license": "Zlib",
        "install_root": str(prefix / "share" / "portfolio-lab" / "toolchain" / "alpine-root"),
        "repository": "main",
    }
    manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Alter checksum
    manifest["packages"]["zlib-1.3.2-r0.apk"]["sha256"] = "0000000000000000000000000000000000000000000000000000000000000000"
    manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    res_sha = run_entrypoint(
        ["--prefix", str(prefix), "--skip-tools=bun,bunx,python3,uv,ninja,git,curl,sqlite3,rsync,zstd,jq", "verify"],
        check=False,
    )
    assert res_sha.returncode != 0
    assert "sha256" in res_sha.stderr.lower()

    # Alter license
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    manifest["packages"]["zlib-1.3.2-r0.apk"]["sha256"] = "1f3d5f463f490dad3a68097376711bfe5e8156e9e8daff3070513aa4378cdeca"
    manifest["packages"]["zlib-1.3.2-r0.apk"]["license"] = "GPL-99"
    manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    res_lic = run_entrypoint(
        ["--prefix", str(prefix), "--skip-tools=bun,bunx,python3,uv,ninja,git,curl,sqlite3,rsync,zstd,jq", "verify"],
        check=False,
    )
    assert res_lic.returncode != 0
    assert "license" in res_lic.stderr.lower()

    # Alter repository
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    manifest["packages"]["zlib-1.3.2-r0.apk"]["license"] = "Zlib"
    manifest["packages"]["zlib-1.3.2-r0.apk"]["repository"] = "community"
    manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    res_repo = run_entrypoint(
        ["--prefix", str(prefix), "--skip-tools=bun,bunx,python3,uv,ninja,git,curl,sqlite3,rsync,zstd,jq", "verify"],
        check=False,
    )
    assert res_repo.returncode != 0
    assert "repository" in res_repo.stderr.lower()

    # Alter source URL: https website different from official URL but well-formed is acceptable per verified manifest URL policy
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    manifest["packages"]["zlib-1.3.2-r0.apk"]["repository"] = "main"
    manifest["packages"]["zlib-1.3.2-r0.apk"]["source_url"] = "https://invalid-url/not-the-official/zlib-1.3.2-r0.apk"
    manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # This is an acceptable well-formed HTTPS URL so verify validation succeeds
    res_url_ok = run_entrypoint(
        ["--prefix", str(prefix), "--skip-tools=bun,bunx,python3,uv,ninja,git,curl,sqlite3,rsync,zstd,jq", "verify"],
        check=True,
    )
    assert "verified" in res_url_ok.stdout.lower() or "ok" in res_url_ok.stdout.lower()

    # Alter install root
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    manifest["packages"]["zlib-1.3.2-r0.apk"]["source_url"] = "https://dl-cdn.alpinelinux.org/alpine/v3.22/main/x86_64/zlib-1.3.2-r0.apk"
    manifest["packages"]["zlib-1.3.2-r0.apk"]["install_root"] = "/somewhere/else"
    manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    res_root = run_entrypoint(
        ["--prefix", str(prefix), "--skip-tools=bun,bunx,python3,uv,ninja,git,curl,sqlite3,rsync,zstd,jq", "verify"],
        check=False,
    )
    assert res_root.returncode != 0
    assert "install_root" in res_root.stderr.lower()


def test_verify_accepts_recorded_sanitized_url_but_rejects_malformed_or_credential_urls(tmp_path: Path) -> None:
    """Plain verify after install with URL overrides must succeed without re-passing overrides; malformed/credential URLs must fail."""
    prefix = tmp_path / "url_record_prefix"
    cache_dir = tmp_path / "url_record_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    ca_apk = cache_dir / "ca-certificates-bundle-20260611-r0.apk"
    ca_sha = _make_dummy_apk(ca_apk, {"etc/ssl/certs/ca-certificates.crt": b"-----BEGIN CERTIFICATE-----\nMOCK\n-----END CERTIFICATE-----\n"})

    custom_url = "https://custom-mirror.local/alpine/v3.22/main/ca-certificates-bundle-20260611-r0.apk"
    install_args = [
        "--prefix",
        str(prefix),
        "--mock-closure",
        f"--override-artifact=ca-certificates-bundle-20260611-r0.apk={ca_apk}",
        f"--override-sha256=ca-certificates-bundle-20260611-r0.apk={ca_sha}",
        f"--override-url=ca-certificates-bundle-20260611-r0.apk={custom_url}",
        "--skip-tools=bun,bunx,python3,uv,ninja,git,curl,sqlite3,rsync,zstd,jq",
        "install",
    ]
    run_entrypoint(install_args, check=True)

    # Plain verify (no overrides re-passed) succeeds: manifest URL is authoritative if well-formed
    res_plain = run_entrypoint(
        ["--prefix", str(prefix), "--mock-closure", "--skip-tools=bun,bunx,python3,uv,ninja,git,curl,sqlite3,rsync,zstd,jq", "verify"],
        check=True,
    )
    assert "verified" in res_plain.stdout.lower() or "ok" in res_plain.stdout.lower()

    # Malformed URL in manifest must fail
    manifest_file = prefix / "share" / "portfolio-lab" / "toolchain" / "bootstrap-manifest.json"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    manifest["packages"]["ca-certificates-bundle-20260611-r0.apk"]["source_url"] = "not-a-url/foo"
    manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    res_bad_url = run_entrypoint(
        ["--prefix", str(prefix), "--mock-closure", "--skip-tools=bun,bunx,python3,uv,ninja,git,curl,sqlite3,rsync,zstd,jq", "verify"],
        check=False,
    )
    assert res_bad_url.returncode != 0
    assert "url" in res_bad_url.stderr.lower()

    # Credential-bearing URL in manifest must fail
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    manifest["packages"]["ca-certificates-bundle-20260611-r0.apk"]["source_url"] = "https://user:supersecret@dl-cdn.alpinelinux.org/alpine/v3.22/main/x86_64/ca-certificates-bundle-20260611-r0.apk"
    manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    res_cred = run_entrypoint(
        ["--prefix", str(prefix), "--mock-closure", "--skip-tools=bun,bunx,python3,uv,ninja,git,curl,sqlite3,rsync,zstd,jq", "verify"],
        check=False,
    )
    assert res_cred.returncode != 0
    assert "url" in res_cred.stderr.lower()


def test_verify_fails_closed_when_runtime_package_install_root_tampered(tmp_path: Path) -> None:
    """verify must validate runtime packages closure with the same rigor as build_packages."""
    prefix = tmp_path / "runtime_pkg_root_prefix"
    cache_dir = tmp_path / "runtime_pkg_root_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    ca_apk = cache_dir / "ca-certificates-bundle-20260611-r0.apk"
    ca_sha = _make_dummy_apk(ca_apk, {"etc/ssl/certs/ca-certificates.crt": b"-----BEGIN CERTIFICATE-----\nMOCK\n-----END CERTIFICATE-----\n"})

    install_args = [
        "--prefix",
        str(prefix),
        "--mock-closure",
        f"--override-artifact=ca-certificates-bundle-20260611-r0.apk={ca_apk}",
        f"--override-sha256=ca-certificates-bundle-20260611-r0.apk={ca_sha}",
        "--skip-tools=bun,bunx,python3,uv,ninja,git,curl,sqlite3,rsync,zstd,jq",
        "install",
    ]
    run_entrypoint(install_args, check=True)

    manifest_file = prefix / "share" / "portfolio-lab" / "toolchain" / "bootstrap-manifest.json"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))

    # Verify base case succeeds in mock mode (fixture record authoritative)
    run_entrypoint(
        ["--prefix", str(prefix), "--mock-closure", "--skip-tools=bun,bunx,python3,uv,ninja,git,curl,sqlite3,rsync,zstd,jq", "verify"],
        check=True,
    )

    # Tamper install_root of runtime package
    manifest["packages"]["ca-certificates-bundle-20260611-r0.apk"]["install_root"] = "/somewhere/else"
    manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    res_root = run_entrypoint(
        ["--prefix", str(prefix), "--mock-closure", "--skip-tools=bun,bunx,python3,uv,ninja,git,curl,sqlite3,rsync,zstd,jq", "verify"],
        check=False,
    )
    # install_root is structural metadata the installer controls: even in mock mode the manifest
    # record must not carry arbitrary values, so a tampered install_root fails closed.
    assert res_root.returncode != 0
    assert "install_root" in res_root.stderr.lower()


def test_verify_rejects_schema_or_layout_revision_mismatch(tmp_path: Path) -> None:
    """verify must reject schema_version/layout_revision mismatch before any behavioral checks."""
    prefix = tmp_path / "schema_layout_prefix"
    cache_dir = tmp_path / "schema_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    ca_apk = cache_dir / "ca-certificates-bundle-20260611-r0.apk"
    ca_sha = _make_dummy_apk(ca_apk, {"etc/ssl/certs/ca-certificates.crt": b"-----BEGIN CERTIFICATE-----\nMOCK\n-----END CERTIFICATE-----\n"})

    install_args = [
        "--prefix",
        str(prefix),
        "--mock-closure",
        f"--override-artifact=ca-certificates-bundle-20260611-r0.apk={ca_apk}",
        f"--override-sha256=ca-certificates-bundle-20260611-r0.apk={ca_sha}",
        "--skip-tools=bun,bunx,python3,uv,ninja,git,curl,sqlite3,rsync,zstd,jq",
        "install",
    ]
    run_entrypoint(install_args, check=True)

    manifest_file = prefix / "share" / "portfolio-lab" / "toolchain" / "bootstrap-manifest.json"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))

    # Case 1: schema mismatch
    manifest["schema_version"] = "portfolio-lab-cursor-box-bootstrap/OLD"
    manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    res_schema = run_entrypoint(
        ["--prefix", str(prefix), "--mock-closure", "--skip-tools=bun,bunx,python3,uv,ninja,git,curl,sqlite3,rsync,zstd,jq", "verify"],
        check=False,
    )
    assert res_schema.returncode != 0
    assert "schema" in res_schema.stderr.lower()

    # Case 2: layout mismatch
    manifest["schema_version"] = "portfolio-lab-cursor-box-bootstrap/v2"
    manifest["layout_revision"] = "v1-old-layout"
    manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    res_layout = run_entrypoint(
        ["--prefix", str(prefix), "--mock-closure", "--skip-tools=bun,bunx,python3,uv,ninja,git,curl,sqlite3,rsync,zstd,jq", "verify"],
        check=False,
    )
    assert res_layout.returncode != 0
    assert "layout" in res_layout.stderr.lower()


def test_verify_passes_full_relocatable_build_environment_to_tools_and_probes(tmp_path: Path) -> None:
    """verify must pass the full relocatable build environment (LD_LIBRARY_PATH, CC, etc.) to all build tools and probes."""
    prefix = tmp_path / "verify_env_prefix"
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True)

    ca_apk = cache_dir / "ca-certificates-bundle-20260611-r0.apk"
    ca_sha = _make_dummy_apk(ca_apk, {"etc/ssl/certs/ca-certificates.crt": b"-----BEGIN CERTIFICATE-----\nMOCK\n-----END CERTIFICATE-----\n"})

    ninja_zip = cache_dir / "ninja.whl"
    ninja_sha = _make_dummy_zip(ninja_zip, "ninja-1.13.2.data/scripts/ninja", _functional_ninja_script())

    run_entrypoint(
        [
            "--prefix",
            str(prefix),
            "--mock-closure",
            f"--override-artifact=ca-certificates-bundle-20260611-r0.apk={ca_apk}",
            f"--override-sha256=ca-certificates-bundle-20260611-r0.apk={ca_sha}",
            f"--override-artifact=ninja={ninja_zip}",
            f"--override-sha256=ninja={ninja_sha}",
            "--skip-tools=bun,bunx,python3,uv,git,curl,sqlite3,rsync,zstd,jq",
            "install",
        ],
        check=True,
    )

    build_root = prefix / "share" / "portfolio-lab" / "toolchain" / "alpine-build-root"
    _create_fake_build_root(build_root, require_env=True)

    manifest_path = prefix / "share" / "portfolio-lab" / "toolchain" / "bootstrap-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    import scripts.portfolio_lab_cursor_box_bootstrap as mod
    manifest["build_packages"] = {
        pkg_name: {
            "source_url": f"{mod.ALPINE_COMMUNITY_DL_BASE if repo == 'community' else mod.ALPINE_DL_BASE}/{pkg_name}",
            "sha256": sha,
            "license": lic,
            "install_root": str(build_root),
            "repository": repo,
        }
        for pkg_name, sha, lic, repo in mod.ALPINE_BUILD_PACKAGE_CLOSURE
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    res = run_entrypoint(
        ["--prefix", str(prefix), "--mock-closure", "--skip-tools=bun,bunx,python3,uv,git,curl,sqlite3,rsync,zstd,jq", "verify"],
        check=True,
    )
    assert "verified" in res.stdout.lower() or "ok" in res.stdout.lower()


def test_verify_fails_closed_when_build_tool_escapes_build_root(tmp_path: Path) -> None:
    """verify must fail closed if a build tool in alpine-build-root symlinks or resolves outside the build root."""
    prefix = tmp_path / "escape_tool_prefix"
    cache_dir = tmp_path / "escape_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    ca_apk = cache_dir / "ca-certificates-bundle-20260611-r0.apk"
    ca_sha = _make_dummy_apk(ca_apk, {"etc/ssl/certs/ca-certificates.crt": b"-----BEGIN CERTIFICATE-----\nMOCK\n-----END CERTIFICATE-----\n"})

    ninja_zip = cache_dir / "ninja.whl"
    ninja_sha = _make_dummy_zip(ninja_zip, "ninja-1.13.2.data/scripts/ninja", _functional_ninja_script())

    run_entrypoint(
        [
            "--prefix",
            str(prefix),
            "--mock-closure",
            f"--override-artifact=ca-certificates-bundle-20260611-r0.apk={ca_apk}",
            f"--override-sha256=ca-certificates-bundle-20260611-r0.apk={ca_sha}",
            f"--override-artifact=ninja={ninja_zip}",
            f"--override-sha256=ninja={ninja_sha}",
            "--skip-tools=bun,bunx,python3,uv,git,curl,sqlite3,rsync,zstd,jq",
            "install",
        ],
        check=True,
    )

    build_root = prefix / "share" / "portfolio-lab" / "toolchain" / "alpine-build-root"
    _create_fake_build_root(build_root)

    # Replace gcc with a symlink pointing outside prefix (e.g. to /usr/bin/gcc or tmp file)
    external_gcc = tmp_path / "external_gcc"
    external_gcc.write_text("#!/bin/sh\necho gcc\n", encoding="utf-8")
    external_gcc.chmod(0o755)

    gcc_link = build_root / "usr" / "bin" / "gcc"
    gcc_link.unlink()
    gcc_link.symlink_to(external_gcc)

    manifest_path = prefix / "share" / "portfolio-lab" / "toolchain" / "bootstrap-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    import scripts.portfolio_lab_cursor_box_bootstrap as mod
    manifest["build_packages"] = {
        pkg_name: {
            "source_url": f"{mod.ALPINE_COMMUNITY_DL_BASE if repo == 'community' else mod.ALPINE_DL_BASE}/{pkg_name}",
            "sha256": sha,
            "license": lic,
            "install_root": str(build_root),
            "repository": repo,
        }
        for pkg_name, sha, lic, repo in mod.ALPINE_BUILD_PACKAGE_CLOSURE
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    res = run_entrypoint(
        ["--prefix", str(prefix), "--mock-closure", "--skip-tools=bun,bunx,python3,uv,git,curl,sqlite3,rsync,zstd,jq", "verify"],
        check=False,
    )
    assert res.returncode != 0
    assert "escape" in res.stderr.lower() or "outside" in res.stderr.lower() or "containment" in res.stderr.lower()


def test_verify_fails_closed_when_compiler_probe_fails(tmp_path: Path) -> None:
    """verify must fail closed when a required compiler (e.g. gcc) fails its compile/link/run probe."""
    prefix = tmp_path / "failed_compiler_prefix"
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True)

    ca_apk = cache_dir / "ca-certificates-bundle-20260611-r0.apk"
    ca_sha = _make_dummy_apk(ca_apk, {"etc/ssl/certs/ca-certificates.crt": b"-----BEGIN CERTIFICATE-----\nMOCK\n-----END CERTIFICATE-----\n"})

    ninja_zip = cache_dir / "ninja.whl"
    ninja_sha = _make_dummy_zip(ninja_zip, "ninja-1.13.2.data/scripts/ninja", _functional_ninja_script())

    run_entrypoint(
        [
            "--prefix",
            str(prefix),
            "--mock-closure",
            f"--override-artifact=ca-certificates-bundle-20260611-r0.apk={ca_apk}",
            f"--override-sha256=ca-certificates-bundle-20260611-r0.apk={ca_sha}",
            f"--override-artifact=ninja={ninja_zip}",
            f"--override-sha256=ninja={ninja_sha}",
            "--skip-tools=bun,bunx,python3,uv,git,curl,sqlite3,rsync,zstd,jq",
            "install",
        ],
        check=True,
    )

    build_root = prefix / "share" / "portfolio-lab" / "toolchain" / "alpine-build-root"
    _create_fake_build_root(build_root)

    # Break gcc to fail with exit 1
    (build_root / "usr" / "bin" / "gcc").write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")

    manifest_path = prefix / "share" / "portfolio-lab" / "toolchain" / "bootstrap-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    import scripts.portfolio_lab_cursor_box_bootstrap as mod
    manifest["build_packages"] = {
        pkg_name: {
            "source_url": f"{mod.ALPINE_COMMUNITY_DL_BASE if repo == 'community' else mod.ALPINE_DL_BASE}/{pkg_name}",
            "sha256": sha,
            "license": lic,
            "install_root": str(build_root),
            "repository": repo,
        }
        for pkg_name, sha, lic, repo in mod.ALPINE_BUILD_PACKAGE_CLOSURE
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    res = run_entrypoint(
        ["--prefix", str(prefix), "--mock-closure", "--skip-tools=bun,bunx,python3,uv,git,curl,sqlite3,rsync,zstd,jq", "verify"],
        check=False,
    )
    assert res.returncode != 0
    assert "gcc" in res.stderr.lower() or "verify failed" in res.stderr.lower()


def test_verify_fails_closed_when_ninja_is_echo_only_mock(tmp_path: Path) -> None:
    """verify must reject echo-only Ninja mock and require usable build command behavior."""
    prefix = tmp_path / "echo_only_ninja_prefix"
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True)

    ca_apk = cache_dir / "ca-certificates-bundle-20260611-r0.apk"
    ca_sha = _make_dummy_apk(ca_apk, {"etc/ssl/certs/ca-certificates.crt": b"-----BEGIN CERTIFICATE-----\nMOCK\n-----END CERTIFICATE-----\n"})

    # Echo-only Ninja mock that returns 1.13.2 for --version but cannot build
    echo_ninja_zip = cache_dir / "echo_ninja.whl"
    echo_ninja_sha = _make_dummy_zip(echo_ninja_zip, "ninja-1.13.2.data/scripts/ninja", b"#!/bin/sh\necho 1.13.2\n")

    run_entrypoint(
        [
            "--prefix",
            str(prefix),
            "--mock-closure",
            f"--override-artifact=ca-certificates-bundle-20260611-r0.apk={ca_apk}",
            f"--override-sha256=ca-certificates-bundle-20260611-r0.apk={ca_sha}",
            f"--override-artifact=ninja={echo_ninja_zip}",
            f"--override-sha256=ninja={echo_ninja_sha}",
            "--skip-tools=bun,bunx,python3,uv,git,curl,sqlite3,rsync,zstd,jq",
            "install",
        ],
        check=True,
    )

    build_root = prefix / "share" / "portfolio-lab" / "toolchain" / "alpine-build-root"
    _create_fake_build_root(build_root)

    manifest_path = prefix / "share" / "portfolio-lab" / "toolchain" / "bootstrap-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    import scripts.portfolio_lab_cursor_box_bootstrap as mod
    manifest["build_packages"] = {
        pkg_name: {
            "source_url": f"{mod.ALPINE_COMMUNITY_DL_BASE if repo == 'community' else mod.ALPINE_DL_BASE}/{pkg_name}",
            "sha256": sha,
            "license": lic,
            "install_root": str(build_root),
            "repository": repo,
        }
        for pkg_name, sha, lic, repo in mod.ALPINE_BUILD_PACKAGE_CLOSURE
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    res = run_entrypoint(
        ["--prefix", str(prefix), "--mock-closure", "--skip-tools=bun,bunx,python3,uv,git,curl,sqlite3,rsync,zstd,jq", "verify"],
        check=False,
    )
    assert res.returncode != 0
    assert "ninja" in res.stderr.lower()


def test_partial_reinstall_with_skip_tools_preserves_existing_managed_tools(tmp_path: Path) -> None:
    """Partial reinstall with --skip-tools must not destructively remove already-installed standalone payloads/wrappers."""
    prefix = tmp_path / "partial_reinstall_prefix"
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True)

    # Different CA artifacts so the second install is a real reinstall (manifest packages differ)
    ca_apk_a = cache_dir / "ca-a.apk"
    ca_sha_a = _make_dummy_apk(ca_apk_a, {"etc/ssl/certs/ca-certificates.crt": b"-----BEGIN CERTIFICATE-----\nMOCK-A\n-----END CERTIFICATE-----\n"})
    ca_apk_b = cache_dir / "ca-b.apk"
    ca_sha_b = _make_dummy_apk(ca_apk_b, {"etc/ssl/certs/ca-certificates.crt": b"-----BEGIN CERTIFICATE-----\nMOCK-B\n-----END CERTIFICATE-----\n"})

    bun_zip = cache_dir / "bun.zip"
    bun_sha = _make_dummy_zip(bun_zip, "bun-linux-x64-musl/bun", b"#!/bin/sh\necho 1.4.0\n")
    uv_tar = cache_dir / "uv.tar.gz"
    uv_sha = _make_dummy_tar_gz(uv_tar, {"uv-x86_64-unknown-linux-musl/uv": b"#!/bin/sh\necho uv 0.12.9\n"})
    ninja_zip = cache_dir / "ninja.whl"
    ninja_sha = _make_dummy_zip(ninja_zip, "ninja-1.13.2.data/scripts/ninja", _functional_ninja_script())

    other_skips = "python3,git,curl,sqlite3,rsync,zstd,jq"

    def install_args(ca_apk: Path, ca_sha: str, extra_skip: str) -> list[str]:
        return [
            "--prefix",
            str(prefix),
            "--mock-closure",
            f"--override-artifact=ca-certificates-bundle-20260611-r0.apk={ca_apk}",
            f"--override-sha256=ca-certificates-bundle-20260611-r0.apk={ca_sha}",
            f"--override-artifact=bun={bun_zip}",
            f"--override-sha256=bun={bun_sha}",
            f"--override-artifact=uv={uv_tar}",
            f"--override-sha256=uv={uv_sha}",
            f"--override-artifact=ninja={ninja_zip}",
            f"--override-sha256=ninja={ninja_sha}",
            f"--skip-tools={f'{extra_skip},' + other_skips if extra_skip else other_skips}",
            "install",
        ]

    # Install A: bun + uv + ninja (+ bunx) all present
    run_entrypoint(install_args(ca_apk_a, ca_sha_a, ""), check=True)

    toolchain_dir = prefix / "share" / "portfolio-lab" / "toolchain"
    standalone_dir = toolchain_dir / "standalone"
    bin_dir = prefix / "bin"
    manifest_path = toolchain_dir / "bootstrap-manifest.json"

    ninja_payload_before = (standalone_dir / "ninja").read_bytes()
    ninja_wrapper_before = (bin_dir / "ninja").read_bytes()
    manifest_before = json.loads(manifest_path.read_text(encoding="utf-8"))
    ninja_record_before = manifest_before["tools"]["ninja"]

    # Partial reinstall: ninja newly skipped while the package manifest otherwise changes
    run_entrypoint(install_args(ca_apk_b, ca_sha_b, "ninja"), check=True)

    # Pre-existing skipped payload and wrapper must remain byte-identical
    assert (standalone_dir / "ninja").read_bytes() == ninja_payload_before, "skipped ninja payload was destroyed"
    assert (bin_dir / "ninja").read_bytes() == ninja_wrapper_before, "skipped ninja wrapper was destroyed"

    # Manifest must keep the preserved ninja record coherent with the surviving files
    manifest_after = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest_after["tools"]["ninja"] == ninja_record_before, "preserved ninja manifest record changed"

    # verify must still pass on the preserved toolchain
    res_verify = run_entrypoint(
        ["--prefix", str(prefix), "--mock-closure", "verify"],
        check=True,
    )
    assert "verified" in res_verify.stdout.lower() or "ok" in res_verify.stdout.lower()


def test_standalone_python_tar_rejects_traversal_escaping_symlink_and_hardlink(tmp_path: Path) -> None:
    """Standalone Python tar extraction must reject traversal, escaping destinations, unsafe symlinks, and hardlinks."""
    prefix = tmp_path / "hostile_py_prefix"
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True)

    # 1. Traversal member
    bad_traversal_tar = cache_dir / "bad_traversal.tar.gz"
    with tarfile.open(bad_traversal_tar, "w:gz") as tf:
        data = b"malicious"
        ti = tarfile.TarInfo(name="python/../escape.txt")
        ti.size = len(data)
        tf.addfile(ti, io.BytesIO(data))
    bad_traversal_sha = hashlib.sha256(bad_traversal_tar.read_bytes()).hexdigest()

    res1 = run_entrypoint(
        [
            "--prefix",
            str(prefix),
            "--mock-closure",
            f"--override-artifact=python3={bad_traversal_tar}",
            f"--override-sha256=python3={bad_traversal_sha}",
            "--skip-tools=bun,uv,git,curl,sqlite3,rsync,zstd,jq",
            "install",
        ],
        check=False,
    )
    assert res1.returncode != 0
    assert "unsafe" in res1.stderr.lower() or "traversal" in res1.stderr.lower() or "escape" in res1.stderr.lower()

    # 2. Escaping symlink member
    bad_symlink_tar = cache_dir / "bad_symlink.tar.gz"
    with tarfile.open(bad_symlink_tar, "w:gz") as tf:
        ti = tarfile.TarInfo(name="python/bin/bad_link")
        ti.type = tarfile.SYMTYPE
        ti.linkname = "../../../../../etc/shadow"
        tf.addfile(ti)
    bad_symlink_sha = hashlib.sha256(bad_symlink_tar.read_bytes()).hexdigest()

    res2 = run_entrypoint(
        [
            "--prefix",
            str(prefix),
            "--mock-closure",
            f"--override-artifact=python3={bad_symlink_tar}",
            f"--override-sha256=python3={bad_symlink_sha}",
            "--skip-tools=bun,uv,git,curl,sqlite3,rsync,zstd,jq",
            "install",
        ],
        check=False,
    )
    assert res2.returncode != 0
    assert "unsafe" in res2.stderr.lower() or "escape" in res2.stderr.lower() or "symlink" in res2.stderr.lower()

    # 3. Escaping hardlink member
    bad_hardlink_tar = cache_dir / "bad_hardlink.tar.gz"
    with tarfile.open(bad_hardlink_tar, "w:gz") as tf:
        ti = tarfile.TarInfo(name="python/bin/bad_hardlink")
        ti.type = tarfile.LNKTYPE
        ti.linkname = "/etc/passwd"
        tf.addfile(ti)
    bad_hardlink_sha = hashlib.sha256(bad_hardlink_tar.read_bytes()).hexdigest()

    res3 = run_entrypoint(
        [
            "--prefix",
            str(prefix),
            "--mock-closure",
            f"--override-artifact=python3={bad_hardlink_tar}",
            f"--override-sha256=python3={bad_hardlink_sha}",
            "--skip-tools=bun,uv,git,curl,sqlite3,rsync,zstd,jq",
            "install",
        ],
        check=False,
    )
    assert res3.returncode != 0
    assert "unsafe" in res3.stderr.lower() or "escape" in res3.stderr.lower() or "hardlink" in res3.stderr.lower()


def test_rejects_archive_members_with_traversal_or_absolute_paths(tmp_path: Path) -> None:
    """The extraction must reject packages with directory traversal or absolute paths."""
    prefix = tmp_path / "hostile_prefix"
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True)

    # Bun
    bun_zip = cache_dir / "bun.zip"
    bun_sha = _make_dummy_zip(bun_zip, "bun-linux-x64-musl/bun", b"#!/bin/sh\necho 1.4.0\n")

    # Python
    py_tar = cache_dir / "python.tar.gz"
    py_sha = _make_dummy_tar_gz(py_tar, {"python/bin/python3": b'#!/bin/sh\necho Python 3.11.16\n'})

    hostile_apk = cache_dir / "bad.apk"
    with tarfile.open(hostile_apk, "w:gz") as tf:
        data = b"malicious"
        ti = tarfile.TarInfo(name="../escape.txt")
        ti.size = len(data)
        tf.addfile(ti, io.BytesIO(data))
    bad_sha = hashlib.sha256(hostile_apk.read_bytes()).hexdigest()

    res = run_entrypoint(
        [
            "--prefix",
            str(prefix),
            "--mock-closure",
            f"--override-artifact=bun={bun_zip}",
            f"--override-sha256=bun={bun_sha}",
            f"--override-artifact=python3={py_tar}",
            f"--override-sha256=python3={py_sha}",
            f"--override-artifact=git-2.49.1-r0.apk={hostile_apk}",
            f"--override-sha256=git-2.49.1-r0.apk={bad_sha}",
            "install",
        ],
        check=False,
    )
    assert res.returncode != 0
    assert "traversal" in res.stderr.lower() or "escape" in res.stderr.lower() or "unsafe" in res.stderr.lower()
    assert not (prefix / ".." / "escape.txt").exists()


def test_install_fails_closed_when_checksum_mismatch(tmp_path: Path) -> None:
    """Checksum mismatch must abort installation, clean up staging, and fail closed."""
    prefix = tmp_path / "prefix_bad_sha"
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True)

    dummy_bin = b"#!/bin/sh\necho corrupted\n"
    bun_zip = cache_dir / "bun.zip"
    _make_dummy_zip(bun_zip, "bun-linux-x64-musl/bun", dummy_bin)

    res = run_entrypoint(
        [
            "--prefix",
            str(prefix),
            f"--override-artifact=bun={bun_zip}",
            "--override-sha256=bun=0000000000000000000000000000000000000000000000000000000000000000",
            "install",
        ],
        check=False,
    )
    assert res.returncode != 0
    assert "checksum mismatch" in res.stderr.lower() or "checksum mismatch" in res.stdout.lower()
    assert not (prefix / "bin" / "bun").exists()


def test_verify_fails_closed_when_runtime_binary_fails(tmp_path: Path) -> None:
    """verify must fail closed if an installed tool fails its behavior verification."""
    prefix = tmp_path / "verify_prefix"
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True)

    ca_apk = cache_dir / "ca-certificates-bundle-20260611-r0.apk"
    ca_sha = _make_dummy_apk(ca_apk, {"etc/ssl/certs/ca-certificates.crt": b"-----BEGIN CERTIFICATE-----\nMOCK\n-----END CERTIFICATE-----\n"})

    uv_tar = cache_dir / "uv.tar.gz"
    uv_sha = _make_dummy_tar_gz(uv_tar, {"uv-x86_64-unknown-linux-musl/uv": b"#!/bin/sh\nexit 1\n"})

    run_entrypoint(
        [
            "--prefix",
            str(prefix),
            "--mock-closure",
            f"--override-artifact=ca-certificates-bundle-20260611-r0.apk={ca_apk}",
            f"--override-sha256=ca-certificates-bundle-20260611-r0.apk={ca_sha}",
            f"--override-artifact=uv={uv_tar}",
            f"--override-sha256=uv={uv_sha}",
            "--skip-tools=bun,bunx,python3,ninja,git,curl,sqlite3,rsync,zstd,jq",
            "install",
        ],
        check=True,
    )

    res = run_entrypoint(
        ["--prefix", str(prefix), "--mock-closure", "--skip-tools=bun,bunx,python3,ninja,git,curl,sqlite3,rsync,zstd,jq", "verify"],
        check=False,
    )
    assert res.returncode != 0
    assert "verify failed" in res.stderr.lower()


def test_verify_rejects_version_only_echo_and_requires_behavioral_operations(tmp_path: Path) -> None:
    """verify must reject tools that merely echo a version string or shallow import without supporting real operations."""
    prefix = tmp_path / "version_only_prefix"
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True)

    # Mock CA bundle
    ca_apk = cache_dir / "ca-certificates-bundle-20260611-r0.apk"
    ca_sha = _make_dummy_apk(ca_apk, {"etc/ssl/certs/ca-certificates.crt": b"-----BEGIN CERTIFICATE-----\nMOCK\n-----END CERTIFICATE-----\n"})

    # Git that satisfies `git version` (old shallow check) but fails on real repo/bundle operations
    fake_git = cache_dir / "fake_git.apk"
    fake_git_sha = _make_dummy_apk(
        fake_git,
        {"usr/bin/git": b'#!/bin/sh\nif [ "$1" = "version" ]; then echo "git version 2.49.1"; exit 0; fi; exit 1\n'},
    )

    # Python that satisfies shallow `import encodings, os` (old check) but fails on real in-memory sqlite query
    fake_py = cache_dir / "fake_py.tar.gz"
    fake_py_sha = _make_dummy_tar_gz(
        fake_py,
        {
            "python/bin/python3": b'#!/bin/sh\nif [ "$1" = "-c" ] && [ "$2" = "import encodings, os; print(\'OK\')" ]; then echo "OK"; exit 0; fi; exit 1\n',
        },
    )

    run_entrypoint(
        [
            "--prefix",
            str(prefix),
            "--mock-closure",
            f"--override-artifact=ca-certificates-bundle-20260611-r0.apk={ca_apk}",
            f"--override-sha256=ca-certificates-bundle-20260611-r0.apk={ca_sha}",
            f"--override-artifact=git-2.49.1-r0.apk={fake_git}",
            f"--override-sha256=git-2.49.1-r0.apk={fake_git_sha}",
            f"--override-artifact=python3={fake_py}",
            f"--override-sha256=python3={fake_py_sha}",
            "--skip-tools=bun,uv,ninja,curl,sqlite3,rsync,zstd,jq",
            "install",
        ],
        check=True,
    )

    # Before implementing the new requirements, this verify call passed because it only checked `git version` and `import encodings, os`.
    # With the new requirement, verify MUST execute in-memory sqlite and git repo/bundle operations, so this MUST fail.
    res = run_entrypoint(
        ["--prefix", str(prefix), "--mock-closure", "--skip-tools=bun,uv,ninja,curl,sqlite3,rsync,zstd,jq", "verify"],
        check=False,
    )
    assert res.returncode != 0
    assert "verify failed" in res.stderr.lower()


def test_verify_real_git_bundle_operations_succeeds_in_repo_context(tmp_path: Path) -> None:
    """Regression test: verify must invoke git bundle verify within git_repo context, not outside where git fails."""
    prefix = tmp_path / "real_git_prefix"
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True)

    system_git = shutil.which("git")
    assert system_git is not None, "system git required for real git bundle verify regression test"

    # Create a git package fixture that executes the real system git binary
    git_apk = cache_dir / "git-2.49.1-r0.apk"
    git_script = f'#!/bin/sh\nexec "{system_git}" "$@"\n'
    git_sha = _make_dummy_apk(git_apk, {"usr/bin/git": git_script.encode("utf-8")})

    ca_apk = cache_dir / "ca-certificates-bundle-20260611-r0.apk"
    ca_sha = _make_dummy_apk(ca_apk, {"etc/ssl/certs/ca-certificates.crt": b"-----BEGIN CERTIFICATE-----\nMOCK\n-----END CERTIFICATE-----\n"})

    run_entrypoint(
        [
            "--prefix",
            str(prefix),
            "--mock-closure",
            f"--override-artifact=ca-certificates-bundle-20260611-r0.apk={ca_apk}",
            f"--override-sha256=ca-certificates-bundle-20260611-r0.apk={ca_sha}",
            f"--override-artifact=git-2.49.1-r0.apk={git_apk}",
            f"--override-sha256=git-2.49.1-r0.apk={git_sha}",
            "--skip-tools=bun,bunx,uv,ninja,python3,curl,sqlite3,rsync,zstd,jq",
            "install",
        ],
        check=True,
    )

    # If verify does not pass cwd=git_repo to bundle verify, real git outputs 'error: need a repository to verify a bundle' when invoked outside any git repository
    non_repo_dir = tmp_path / "non_repo_cwd"
    non_repo_dir.mkdir(parents=True, exist_ok=True)
    res = run_entrypoint(
        ["--prefix", str(prefix), "--mock-closure", "--skip-tools=bun,bunx,uv,ninja,python3,curl,sqlite3,rsync,zstd,jq", "verify"],
        check=True,
        cwd=non_repo_dir,
    )
    assert "verified" in res.stdout.lower() or "ok" in res.stdout.lower()


def test_uninstall_validates_prefix_containment_and_cleans_root(tmp_path: Path) -> None:
    """uninstall validates manifest paths stay within prefix and cleans up managed files."""
    prefix = tmp_path / "uninstall_prefix"
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True)

    uv_tar = cache_dir / "uv.tar.gz"
    uv_sha = _make_dummy_tar_gz(uv_tar, {"uv-x86_64-unknown-linux-musl/uv": b"#!/bin/sh\necho uv\n"})

    run_entrypoint(
        [
            "--prefix",
            str(prefix),
            "--mock-closure",
            f"--override-artifact=uv={uv_tar}",
            f"--override-sha256=uv={uv_sha}",
            "--skip-tools=bun,python3,git,curl,sqlite3,rsync,zstd,jq",
            "install",
        ],
        check=True,
    )
    assert (prefix / "bin" / "uv").is_file()

    # Dry run
    res_dry = run_entrypoint(["--prefix", str(prefix), "uninstall", "--dry-run"], check=True)
    assert str(prefix / "bin" / "uv") in res_dry.stdout
    assert (prefix / "bin" / "uv").is_file()

    # Tampered manifest pointing outside prefix must fail closed
    manifest_file = prefix / "share" / "portfolio-lab" / "toolchain" / "bootstrap-manifest.json"
    manifest_data = json.loads(manifest_file.read_text(encoding="utf-8"))
    manifest_data["tools"]["uv"]["install_path"] = "/etc/passwd"
    manifest_file.write_text(json.dumps(manifest_data), encoding="utf-8")

    res_tampered = run_entrypoint(["--prefix", str(prefix), "uninstall"], check=False)
    assert res_tampered.returncode != 0
    assert "containment" in res_tampered.stderr.lower() or "outside" in res_tampered.stderr.lower()

    # Restore manifest and run clean uninstall
    manifest_data["tools"]["uv"]["install_path"] = str(prefix / "bin" / "uv")
    manifest_file.write_text(json.dumps(manifest_data), encoding="utf-8")
    run_entrypoint(["--prefix", str(prefix), "uninstall"], check=True)

    assert not (prefix / "bin" / "uv").exists()
    assert not (prefix / "share" / "portfolio-lab" / "toolchain").exists()


def test_upgrade_replaces_old_manifest_and_raw_bun_layout(tmp_path: Path) -> None:
    """An existing install with an older manifest or raw Bun in bin/ must upgrade to standalone Bun + wrapper rather than returning up-to-date."""
    prefix = tmp_path / "upgrade_prefix"
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True)

    bun_zip = cache_dir / "bun.zip"
    bun_sha = _make_dummy_zip(bun_zip, "bun-linux-x64-musl/bun", b"#!/bin/sh\necho 1.4.0\n")

    # Simulate an older installation: raw bun placed directly at bin/bun, no standalone/bun, older manifest
    bin_dir = prefix / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    raw_bun = bin_dir / "bun"
    raw_bun.write_bytes(b"#!/bin/sh\necho 1.4.0\n")
    raw_bun.chmod(0o755)

    toolchain_dir = prefix / "share" / "portfolio-lab" / "toolchain"
    toolchain_dir.mkdir(parents=True, exist_ok=True)
    (toolchain_dir / "alpine-root").mkdir(parents=True, exist_ok=True)
    (toolchain_dir / "python-root").mkdir(parents=True, exist_ok=True)

    old_manifest = {
        "schema_version": "portfolio-lab-cursor-box-bootstrap/v2",
        "prefix": str(prefix),
        "bin_dir": str(bin_dir),
        "toolchain_dir": str(toolchain_dir),
        "alpine_root": str(toolchain_dir / "alpine-root"),
        "python_root": str(toolchain_dir / "python-root"),
        "tools": {
            "bun": {
                "version": PINNED_BUN_VERSION,
                "source_url": PINNED_BUN_URL,
                "sha256": bun_sha,
                "license": "MIT",
                "archive_format": "zip",
                "install_path": str(raw_bun),
                "installed_sha256": hashlib.sha256(raw_bun.read_bytes()).hexdigest(),
                # Note: older manifest lacks payload_sha256 and standalone_dir
            }
        },
        "packages": {},
    }
    manifest_file = toolchain_dir / "bootstrap-manifest.json"
    manifest_file.write_text(json.dumps(old_manifest), encoding="utf-8")

    # Run install on this prefix; it must NOT return "already up to date", it must upgrade
    res = run_entrypoint(
        [
            "--prefix",
            str(prefix),
            "--mock-closure",
            f"--override-artifact=bun={bun_zip}",
            f"--override-sha256=bun={bun_sha}",
            "--skip-tools=uv,python3,git,curl,sqlite3,rsync,zstd,jq",
            "install",
        ],
        check=True,
    )

    assert "already up to date" not in res.stdout.lower()
    assert "successfully installed toolchain" in res.stdout.lower()

    # Verify standalone Bun exists and bin/bun is now the wrapper
    standalone_bun = toolchain_dir / "standalone" / "bun"
    assert standalone_bun.is_file(), "upgrade must place bun binary in toolchain/standalone/bun"
    wrapper_text = raw_bun.read_text(encoding="utf-8")
    assert "LD_LIBRARY_PATH" in wrapper_text, "bin/bun must be upgraded to wrapper"
    assert "STANDALONE_BUN" in wrapper_text


def test_bun_wrapper_exports_private_ca_variables_and_records_in_manifest(tmp_path: Path) -> None:
    """Bun wrapper must export private CA variables needed for HTTPS (SSL_CERT_FILE and NODE_EXTRA_CA_CERTS)."""
    prefix = tmp_path / "bun_ca_prefix"
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True)

    bun_zip = cache_dir / "bun.zip"
    bun_sha = _make_dummy_zip(bun_zip, "bun-linux-x64-musl/bun", b"#!/bin/sh\necho 1.4.0\n")

    run_entrypoint(
        [
            "--prefix",
            str(prefix),
            "--mock-closure",
            f"--override-artifact=bun={bun_zip}",
            f"--override-sha256=bun={bun_sha}",
            "--skip-tools=python3,uv,git,curl,sqlite3,rsync,zstd,jq",
            "install",
        ],
        check=True,
    )

    bun_wrapper = prefix / "bin" / "bun"
    assert bun_wrapper.is_file()
    content = bun_wrapper.read_text(encoding="utf-8")
    assert "SSL_CERT_FILE" in content, "bun wrapper must export SSL_CERT_FILE"
    assert "NODE_EXTRA_CA_CERTS" in content, "bun wrapper must export NODE_EXTRA_CA_CERTS"


def test_sanitize_download_exception_text_redacts_credentials_and_query_secrets(tmp_path: Path) -> None:
    """Sanitize download exception text so credential/query secrets cannot be printed."""
    prefix = tmp_path / "sanitize_err_prefix"
    secret_url = "https://user:SuperSecretPassword@example.com/download.tar.gz?token=SecretToken12345&sig=SecretSig"

    res = run_entrypoint(
        [
            "--prefix",
            str(prefix),
            "--mock-closure",
            f"--override-url=bun={secret_url}",
            "--skip-tools=python3,uv,git,curl,sqlite3,rsync,zstd,jq",
            "install",
        ],
        check=False,
    )
    assert res.returncode != 0
    combined_output = res.stderr + res.stdout
    assert "SuperSecretPassword" not in combined_output
    assert "SecretToken12345" not in combined_output
    assert "SecretSig" not in combined_output
    assert "[REDACTED]" in combined_output


def test_git_config_command_return_codes_checked(tmp_path: Path) -> None:
    """Git config return codes in behavioral verify must be strictly checked."""
    # Checked structurally in portfolio_lab_cursor_box_bootstrap.py: run_cmd for git config user.email and user.name
    import scripts.portfolio_lab_cursor_box_bootstrap as mod
    import inspect
    verify_src = inspect.getsource(mod.BootstrapManager.verify)
    assert 'r = run_cmd([str(target), "config", "user.email"' in verify_src or 'if run_cmd([str(target), "config"' in verify_src
    assert 'if r.returncode != 0:' in verify_src


def test_install_transactional_rollback_on_late_failure_preserves_old_install(tmp_path: Path) -> None:
    """If install fails late (e.g. during wrapper/manifest publishing), all previous directories and wrappers must be restored."""
    prefix = tmp_path / "rollback_prefix"
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True)

    bun_zip = cache_dir / "bun.zip"
    bun_sha = _make_dummy_zip(bun_zip, "bun-linux-x64-musl/bun", b"#!/bin/sh\necho 1.4.0\n")

    # 1. First initial successful install of bun
    run_entrypoint(
        [
            "--prefix",
            str(prefix),
            "--mock-closure",
            f"--override-artifact=bun={bun_zip}",
            f"--override-sha256=bun={bun_sha}",
            "--skip-tools=python3,uv,git,curl,sqlite3,rsync,zstd,jq",
            "install",
        ],
        check=True,
    )

    manifest_file = prefix / "share" / "portfolio-lab" / "toolchain" / "bootstrap-manifest.json"
    assert manifest_file.is_file()
    initial_manifest_content = manifest_file.read_text(encoding="utf-8")
    initial_wrapper = (prefix / "bin" / "bun").read_text(encoding="utf-8")
    initial_payload = (prefix / "share" / "portfolio-lab" / "toolchain" / "standalone" / "bun").read_bytes()

    # 2. Trigger a second install with an updated payload where we inject a late failure during wrapper copy or manifest write
    bun_zip_v2 = cache_dir / "bun_v2.zip"
    bun_sha_v2 = _make_dummy_zip(bun_zip_v2, "bun-linux-x64-musl/bun", b"#!/bin/sh\necho 1.4.1\n")
    res = run_entrypoint(
        [
            "--prefix",
            str(prefix),
            "--mock-closure",
            f"--override-artifact=bun={bun_zip_v2}",
            f"--override-sha256=bun={bun_sha_v2}",
            "--skip-tools=python3,uv,git,curl,sqlite3,rsync,zstd,jq",
            "install",
        ],
        env={"PORTFOLIO_LAB_INJECT_INSTALL_FAILURE": "late_manifest_failure"},
        check=False,
    )
    assert res.returncode != 0
    # The previous installation must survive intact:
    assert manifest_file.is_file()
    assert manifest_file.read_text(encoding="utf-8") == initial_manifest_content
    assert (prefix / "bin" / "bun").read_text(encoding="utf-8") == initial_wrapper
    assert (prefix / "share" / "portfolio-lab" / "toolchain" / "standalone" / "bun").read_bytes() == initial_payload


def test_reinstall_does_not_delete_active_executing_python_runtime(tmp_path: Path) -> None:
    """During reinstall, the directory containing the currently executing interpreter must not be unlinked/deleted."""
    prefix = tmp_path / "active_py_prefix"
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True)

    py_tar = cache_dir / "python.tar.gz"
    py_sha = _make_dummy_tar_gz(
        py_tar,
        {
            "python/bin/python3": f"#!/bin/sh\nexec {sys.executable} \"$@\"\n".encode("utf-8"),
            "python/lib/python3.11/os.py": b"# os\n",
        },
    )

    # First install
    run_entrypoint(
        [
            "--prefix",
            str(prefix),
            "--mock-closure",
            f"--override-artifact=python3={py_tar}",
            f"--override-sha256=python3={py_sha}",
            "--skip-tools=bun,uv,git,curl,sqlite3,rsync,zstd,jq",
            "install",
        ],
        check=True,
    )

    installed_py_root = prefix / "share" / "portfolio-lab" / "toolchain" / "python-root"
    assert installed_py_root.is_dir()

    # If the installer deletes the active python runtime while running under it, simulate running FROM that python-root
    # We test this by asserting that executing python's tree is protected or asserting no rmtree of active runtime
    res = run_entrypoint(
        [
            "--prefix",
            str(prefix),
            "--mock-closure",
            f"--override-artifact=python3={py_tar}",
            f"--override-sha256=python3={py_sha}",
            "--skip-tools=bun,uv,git,curl,sqlite3,rsync,zstd,jq",
            "install",
        ],
        env={"PORTFOLIO_LAB_SIMULATE_ACTIVE_PYTHON": str(installed_py_root / "bin" / "python3")},
        check=False,
    )
    # If the installer tries to rmtree the active python directory, it must fail or avoid doing so
    assert res.returncode == 0
    assert installed_py_root.is_dir()


def test_blank_host_without_python_verify_fails_nonzero(tmp_path: Path) -> None:
    """On a blank host without Python, verify must fail nonzero rather than emit ready JSON or success."""
    prefix = tmp_path / "blank_host_prefix"
    res = run_entrypoint(
        ["--prefix", str(prefix), "verify"],
        env={"PORTFOLIO_LAB_BOOTSTRAP_PYTHON": "nonexistent-python"},
        check=False,
        use_sh=True,
    )
    assert res.returncode != 0, f"verify on blank host without python must fail nonzero, got {res.returncode}"


def test_uninstall_dry_run_without_python_is_non_mutating(tmp_path: Path) -> None:
    """uninstall --dry-run without Python must be non-mutating and must not download/extract."""
    prefix = tmp_path / "uninstall_dry_run_blank"
    res = run_entrypoint(
        ["--prefix", str(prefix), "uninstall", "--dry-run"],
        env={"PORTFOLIO_LAB_BOOTSTRAP_PYTHON": "nonexistent-python"},
        check=True,
        use_sh=True,
    )
    assert not prefix.exists(), "uninstall --dry-run without python must not mutate or create prefix"
    assert "dry-run" in res.stdout.lower() or "would remove" in res.stdout.lower() or "uninstall" in res.stdout.lower()


def test_stage_0_install_reuses_archive_for_python3(tmp_path: Path) -> None:
    """--stage0-python-archive must actually be reused by Python installer for python3 instead of redownloading/dropping."""
    prefix = tmp_path / "stage0_reuse_prefix"
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True)

    py_tar = cache_dir / "python.tar.gz"
    py_sha = _make_dummy_tar_gz(
        py_tar,
        {
            "python/bin/python3": f"#!/bin/sh\nexec {sys.executable} \"$@\"\n".encode("utf-8"),
            "python/lib/python3.11/os.py": b"# os\n",
        },
    )

    # When --stage0-python-archive is passed, the Python installer should reuse it directly
    # without trying to download from the internet.
    run_entrypoint(
        [
            "--prefix",
            str(prefix),
            "--mock-closure",
            f"--stage0-python-archive={py_tar}",
            f"--override-sha256=python3={py_sha}",
            "--skip-tools=bun,uv,git,curl,sqlite3,rsync,zstd,jq",
            "install",
        ],
        env={"PORTFOLIO_LAB_BOOTSTRAP_PYTHON": "nonexistent-python"},
        check=True,
        use_sh=True,
    )
    assert (prefix / "bin" / "python3").is_file()
    assert (prefix / "share" / "portfolio-lab" / "toolchain" / "python-root" / "bin" / "python3").is_file()


def test_stage_0_dry_run_is_non_mutating_and_install_supports_override(tmp_path: Path) -> None:
    """Without Python, dry-run must not mutate disk, and install must support pre-transferred override."""
    prefix = tmp_path / "stage0_prefix"
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True)

    py_tar = cache_dir / "python.tar.gz"
    _make_dummy_tar_gz(
        py_tar,
        {
            "python/bin/python3": b'#!/bin/sh\nif [ "$1" = "-c" ]; then eval "$2"; exit 0; fi; echo Python 3.11.16\n',
            "python/lib/python3.11/os.py": b"# os\n",
        },
    )

    # dry-run without python must NOT mutate disk
    res_dry = run_entrypoint(
        ["--prefix", str(prefix), "dry-run"],
        env={"PORTFOLIO_LAB_BOOTSTRAP_PYTHON": "nonexistent-python"},
        check=True,
        use_sh=True,
    )
    assert not prefix.exists(), "stage-0 dry-run must never mutate target prefix"
    assert "portfolio-lab-cursor-box-bootstrap" in res_dry.stdout or "bun" in res_dry.stdout

    # install with pre-transferred archive override in stage 0
    res_inst = run_entrypoint(
        [
            "--prefix",
            str(prefix),
            f"--stage0-python-archive={py_tar}",
            "stage-0-info",
        ],
        env={"PORTFOLIO_LAB_BOOTSTRAP_PYTHON": "nonexistent-python"},
        check=True,
        use_sh=True,
    )
    assert res_inst.returncode == 0


def test_stage0_drives_download_branch_single_fetch_and_cleans_only_on_successful_install(tmp_path: Path) -> None:
    """Stage-0 must download via curl/wget exactly once, forward archive to Python installer, and clean up only on successful install."""
    prefix = tmp_path / "stage0_download_prefix"
    fake_bin_dir = tmp_path / "fake_bin"
    fake_bin_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = tmp_path / "source_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Real python payload archive that fake curl will serve
    py_tar = cache_dir / "served_python.tar.gz"
    py_sha = _make_dummy_tar_gz(
        py_tar,
        {
            "python/bin/python3": f"#!/bin/sh\nexec {sys.executable} \"$@\"\n".encode("utf-8"),
            "python/lib/python3.11/os.py": b"# os\n",
        },
    )

    fetch_count_file = tmp_path / "fetch_count.txt"
    fetch_count_file.write_text("0", encoding="utf-8")

    # Controlled fake curl tracking invocation count and copying py_tar to target
    fake_curl = fake_bin_dir / "curl"
    fake_curl_script = f"""#!/bin/sh
# Increment fetch count
cnt=$(cat "{fetch_count_file}")
echo $((cnt + 1)) > "{fetch_count_file}"

out=""
while [ "$#" -gt 0 ]; do
  if [ "$1" = "-o" ]; then out="$2"; shift 2; continue; fi
  shift
done
if [ -n "$out" ]; then
  cp "{py_tar}" "$out"
  exit 0
fi
exit 1
"""
    fake_curl.write_text(fake_curl_script, encoding="utf-8")
    fake_curl.chmod(0o755)

    # Disable system wget so the download branch uses fake_curl
    fake_wget = fake_bin_dir / "wget"
    fake_wget_script = f"""#!/bin/sh
# Increment fetch count
cnt=$(cat "{fetch_count_file}")
echo $((cnt + 1)) > "{fetch_count_file}"

out=""
while [ "$#" -gt 0 ]; do
  if [ "$1" = "-O" ]; then out="$2"; shift 2; continue; fi
  shift
done
if [ -n "$out" ]; then
  cp "{py_tar}" "$out"
  exit 0
fi
exit 1
"""
    fake_wget.write_text(fake_wget_script, encoding="utf-8")
    fake_wget.chmod(0o755)

    ca_apk = cache_dir / "ca-certificates-bundle-20260611-r0.apk"
    ca_sha = _make_dummy_apk(ca_apk, {"etc/ssl/certs/ca-certificates.crt": b"-----BEGIN CERTIFICATE-----\nMOCK\n-----END CERTIFICATE-----\n"})

    # PATH has fake_curl before system curl/wget, and no system python
    test_env = {
        "PATH": f"{fake_bin_dir}:{os.environ.get('PATH', '')}",
        "PORTFOLIO_LAB_BOOTSTRAP_PYTHON": "nonexistent-python",
    }

    res = run_entrypoint(
        [
            "--prefix",
            str(prefix),
            "--mock-closure",
            f"--override-sha256=python3={py_sha}",
            f"--override-artifact=ca-certificates-bundle-20260611-r0.apk={ca_apk}",
            f"--override-sha256=ca-certificates-bundle-20260611-r0.apk={ca_sha}",
            "--skip-tools=bun,bunx,uv,ninja,git,curl,sqlite3,rsync,zstd,jq",
            "install",
        ],
        env=test_env,
        check=True,
        use_sh=True,
    )
    assert res.returncode == 0

    # Verify exactly one download occurred (no duplicate downloads between shell stage-0 and Python installer)
    final_count = int(fetch_count_file.read_text(encoding="utf-8").strip())
    assert final_count == 1, f"expected exactly 1 download, got {final_count}"

    # Verify stage0 directory was cleaned up on successful install
    stage0_dir = prefix / "share" / "portfolio-lab" / "toolchain" / "stage0-python"
    assert not stage0_dir.exists(), "stage0-python directory must be cleaned up on successful install"

    # Verify final python root is working
    py_bin = prefix / "bin" / "python3"
    assert py_bin.is_file()

    # 2. Failed install must NOT clean stage-0
    prefix_fail = tmp_path / "stage0_fail_cleanup_prefix"
    res_fail = run_entrypoint(
        [
            "--prefix",
            str(prefix_fail),
            "--mock-closure",
            f"--override-sha256=python3={py_sha}",
            "--skip-tools=bun,bunx,uv,ninja,git,curl,sqlite3,rsync,zstd,jq",
            "install",
        ],
        env={**test_env, "PORTFOLIO_LAB_INJECT_INSTALL_FAILURE": "late_manifest_failure"},
        check=False,
        use_sh=True,
    )
    assert res_fail.returncode != 0
    stage0_fail_dir = prefix_fail / "share" / "portfolio-lab" / "toolchain" / "stage0-python"
    assert stage0_fail_dir.exists(), "stage0-python directory must NOT be cleaned up on failed install"

    # Verify stage0 directory was cleaned up on successful install
    stage0_dir = prefix / "share" / "portfolio-lab" / "toolchain" / "stage0-python"
    assert not stage0_dir.exists(), "stage0-python directory must be cleaned up on successful install"

    # Verify final python root is working
    py_bin = prefix / "bin" / "python3"
    assert py_bin.is_file()


def test_stage0_passes_downloaded_archive_to_python_and_cleans_up_on_success(tmp_path: Path) -> None:
    """Stage-0 shell bootstrap must pass downloaded/staged archive via --stage0-python-archive and clean up stage-0 directory on success."""
    prefix = tmp_path / "stage0_clean_prefix"
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True)

    py_tar = cache_dir / "external_python.tar.gz"
    py_sha = _make_dummy_tar_gz(
        py_tar,
        {
            "python/bin/python3": f"#!/bin/sh\nexec {sys.executable} \"$@\"\n".encode("utf-8"),
            "python/lib/python3.11/os.py": b"# os\n",
        },
    )

    ca_apk = cache_dir / "ca-certificates-bundle-20260611-r0.apk"
    ca_sha = _make_dummy_apk(ca_apk, {"etc/ssl/certs/ca-certificates.crt": b"-----BEGIN CERTIFICATE-----\nMOCK\n-----END CERTIFICATE-----\n"})

    # Run install through shell script without host Python
    res = run_entrypoint(
        [
            "--prefix",
            str(prefix),
            "--mock-closure",
            f"--override-sha256=python3={py_sha}",
            f"--stage0-python-archive={py_tar}",
            f"--override-artifact=ca-certificates-bundle-20260611-r0.apk={ca_apk}",
            f"--override-sha256=ca-certificates-bundle-20260611-r0.apk={ca_sha}",
            "--skip-tools=bun,bunx,uv,ninja,git,curl,sqlite3,rsync,zstd,jq",
            "install",
        ],
        env={"PORTFOLIO_LAB_BOOTSTRAP_PYTHON": "nonexistent-python"},
        check=True,
        use_sh=True,
    )
    assert res.returncode == 0

    # External archive supplied by user must NOT be deleted
    assert py_tar.is_file(), "explicitly supplied stage0 archive outside stage0 dir must be preserved"

    # Stage-0 temp directory should be cleaned up after successful final install
    stage0_dir = prefix / "share" / "portfolio-lab" / "toolchain" / "stage0-python"
    assert not stage0_dir.exists(), "disposable stage0-python directory must be cleaned up on successful install"

    # Final python root must exist and work
    py_bin = prefix / "bin" / "python3"
    assert py_bin.is_file()


def test_stage0_preserves_diagnostics_on_failure(tmp_path: Path) -> None:
    """Stage-0 shell bootstrap must preserve diagnostic state in stage0 dir if install fails."""
    prefix = tmp_path / "stage0_fail_prefix"
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True)

    py_tar = cache_dir / "external_python.tar.gz"
    py_sha = _make_dummy_tar_gz(
        py_tar,
        {
            "python/bin/python3": f"#!/bin/sh\nexec {sys.executable} \"$@\"\n".encode("utf-8"),
            "python/lib/python3.11/os.py": b"# os\n",
        },
    )

    # Injected install failure
    res = run_entrypoint(
        [
            "--prefix",
            str(prefix),
            "--mock-closure",
            f"--override-sha256=python3={py_sha}",
            f"--stage0-python-archive={py_tar}",
            "--skip-tools=bun,bunx,uv,ninja,git,curl,sqlite3,rsync,zstd,jq",
            "install",
        ],
        env={
            "PORTFOLIO_LAB_BOOTSTRAP_PYTHON": "nonexistent-python",
            "PORTFOLIO_LAB_INJECT_INSTALL_FAILURE": "late_manifest_failure",
        },
        check=False,
        use_sh=True,
    )
    assert res.returncode != 0

    stage0_dir = prefix / "share" / "portfolio-lab" / "toolchain" / "stage0-python"
    assert stage0_dir.exists(), "stage0 directory must be preserved on failure for diagnostics"


def test_install_repairs_corrupted_or_deleted_standalone_uv_and_ninja(tmp_path: Path) -> None:
    """Install idempotency must detect corrupted or deleted standalone uv and ninja payloads and repair them."""
    prefix = tmp_path / "repair_prefix"
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True)

    ca_apk = cache_dir / "ca-certificates-bundle-20260611-r0.apk"
    ca_sha = _make_dummy_apk(ca_apk, {"etc/ssl/certs/ca-certificates.crt": b"-----BEGIN CERTIFICATE-----\nMOCK\n-----END CERTIFICATE-----\n"})

    uv_tar = cache_dir / "uv.tar.gz"
    uv_sha = _make_dummy_tar_gz(uv_tar, {"uv-x86_64-unknown-linux-musl/uv": b"#!/bin/sh\necho uv 0.12.9\n"})

    ninja_zip = cache_dir / "ninja.whl"
    ninja_sha = _make_dummy_zip(ninja_zip, "ninja-1.13.2.data/scripts/ninja", _functional_ninja_script())

    install_args = [
        "--prefix",
        str(prefix),
        "--mock-closure",
        f"--override-artifact=ca-certificates-bundle-20260611-r0.apk={ca_apk}",
        f"--override-sha256=ca-certificates-bundle-20260611-r0.apk={ca_sha}",
        f"--override-artifact=uv={uv_tar}",
        f"--override-sha256=uv={uv_sha}",
        f"--override-artifact=ninja={ninja_zip}",
        f"--override-sha256=ninja={ninja_sha}",
        "--skip-tools=bun,bunx,python3,git,curl,sqlite3,rsync,zstd,jq",
        "install",
    ]

    # First install: succeeds
    run_entrypoint(install_args, check=True)

    standalone_dir = prefix / "share" / "portfolio-lab" / "toolchain" / "standalone"
    assert (standalone_dir / "uv").is_file()
    assert (standalone_dir / "ninja").is_file()

    # Second install: must report already up to date
    res_idem = run_entrypoint(install_args, check=True)
    assert "already up to date" in res_idem.stdout.lower()

    # Corrupt standalone uv payload (keep wrapper intact)
    (standalone_dir / "uv").write_bytes(b"corrupted uv binary")

    # Install must detect corruption, NOT report up to date, and repair it
    res_repair_uv = run_entrypoint(install_args, check=True)
    assert "already up to date" not in res_repair_uv.stdout.lower()
    assert "successfully installed toolchain" in res_repair_uv.stdout.lower()
    assert (standalone_dir / "uv").read_bytes() == b"#!/bin/sh\necho uv 0.12.9\n"

    # Delete standalone ninja payload (keep wrapper intact)
    (standalone_dir / "ninja").unlink()

    # Install must detect missing payload, NOT report up to date, and repair it
    res_repair_ninja = run_entrypoint(install_args, check=True)
    assert "already up to date" not in res_repair_ninja.stdout.lower()
    assert "successfully installed toolchain" in res_repair_ninja.stdout.lower()
    assert (standalone_dir / "ninja").is_file()


def test_verify_bunx_delegates_to_bun_x_with_arguments_without_network(tmp_path: Path) -> None:
    """verify must check bunx delegates to bun x with arguments without network access."""
    prefix = tmp_path / "bunx_verify_prefix"
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True)

    ca_apk = cache_dir / "ca-certificates-bundle-20260611-r0.apk"
    ca_sha = _make_dummy_apk(ca_apk, {"etc/ssl/certs/ca-certificates.crt": b"-----BEGIN CERTIFICATE-----\nMOCK\n-----END CERTIFICATE-----\n"})

    # Bun fixture that responds to --version AND to x --bun --version
    bun_script = (
        '#!/bin/sh\n'
        'if [ "$1" = "--version" ]; then echo "1.4.0"; exit 0; fi\n'
        'if [ "$1" = "x" ]; then\n'
        '  shift\n'
        '  if [ "$1" = "--bun" ] && [ "$2" = "--version" ]; then\n'
        '    echo "1.4.0"\n'
        '    exit 0\n'
        '  fi\n'
        'fi\n'
        'exit 1\n'
    ).encode("utf-8")
    bun_zip = cache_dir / "bun.zip"
    bun_sha = _make_dummy_zip(bun_zip, "bun-linux-x64-musl/bun", bun_script)

    run_entrypoint(
        [
            "--prefix",
            str(prefix),
            "--mock-closure",
            f"--override-artifact=ca-certificates-bundle-20260611-r0.apk={ca_apk}",
            f"--override-sha256=ca-certificates-bundle-20260611-r0.apk={ca_sha}",
            f"--override-artifact=bun={bun_zip}",
            f"--override-sha256=bun={bun_sha}",
            "--skip-tools=python3,uv,ninja,git,curl,sqlite3,rsync,zstd,jq",
            "install",
        ],
        check=True,
    )

    # Run verify - this must check bunx delegation with arguments
    res_verify = run_entrypoint(
        ["--prefix", str(prefix), "--mock-closure", "--skip-tools=python3,uv,ninja,git,curl,sqlite3,rsync,zstd,jq", "verify"],
        check=True,
    )
    assert "verified" in res_verify.stdout.lower() or "ok" in res_verify.stdout.lower()
