"""Agent-compatibility cold path: clean uv, README, frontend node fallback."""

from __future__ import annotations

import json
import os
import stat
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AGENT_UV = PROJECT_ROOT / "scripts" / "agent_uv.sh"
README = PROJECT_ROOT / "README.md"
PACKAGE_JSON = PROJECT_ROOT / "package.json"
MAKEFILE = PROJECT_ROOT / "Makefile"


def _write_exec(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def test_agent_uv_script_is_executable_and_refuses_wrapper_without_payload(tmp_path: Path) -> None:
    assert AGENT_UV.is_file()
    assert os.access(AGENT_UV, os.X_OK)

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    wrapper = _write_exec(
        bin_dir / "uv",
        """#!/bin/sh
# Portfolio Lab user-owned wrapper for uv (relocatable)
export LD_LIBRARY_PATH="/alpine/lib:${LD_LIBRARY_PATH:-}"
STANDALONE_UV="/missing/standalone/uv"
exec echo WRAPPER_RAN
""",
    )
    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env.pop("PORTFOLIO_LAB_UV", None)
    env.pop("PORTFOLIO_LAB_TOOLCHAIN_ROOT", None)
    # HOME without a standalone payload so PATH wrapper is the only candidate.
    env["HOME"] = str(tmp_path)

    result = subprocess.run(
        [str(AGENT_UV), "run", "pytest", "-q"],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 127, result.stdout + result.stderr
    assert "WRAPPER_RAN" not in result.stdout
    assert "wrapper" in result.stderr.lower() or "standalone" in result.stderr.lower()
    assert wrapper.is_file()


def test_agent_uv_prefers_standalone_payload_over_path_wrapper(tmp_path: Path) -> None:
    prefix = tmp_path / ".local"
    standalone_dir = prefix / "share" / "portfolio-lab" / "toolchain" / "standalone"
    standalone_dir.mkdir(parents=True)
    calls = tmp_path / "calls.txt"
    _write_exec(
        standalone_dir / "uv",
        f"""#!/bin/sh
printf '%s\\n' "$@" > "{calls}"
printf '%s\\n' "${{LD_LIBRARY_PATH-UNSET}}" > "{tmp_path / "ld.txt"}"
exit 0
""",
    )
    bin_dir = prefix / "bin"
    bin_dir.mkdir(parents=True)
    _write_exec(
        bin_dir / "uv",
        """#!/bin/sh
# Portfolio Lab user-owned wrapper for uv (relocatable)
export LD_LIBRARY_PATH="/alpine/musl/lib"
STANDALONE_UV="$(cd "$(dirname "$0")/.." && pwd)/share/portfolio-lab/toolchain/standalone/uv"
exec echo WRAPPER_RAN
""",
    )

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["HOME"] = str(tmp_path)
    env.pop("PORTFOLIO_LAB_UV", None)
    env.pop("PORTFOLIO_LAB_TOOLCHAIN_ROOT", None)

    result = subprocess.run(
        [str(AGENT_UV), "run", "pytest", "tests/foo.py"],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert calls.read_text().splitlines() == ["run", "pytest", "tests/foo.py"]
    assert "WRAPPER_RAN" not in result.stdout


def test_agent_uv_honors_portfolio_lab_uv_override(tmp_path: Path) -> None:
    calls = tmp_path / "calls.txt"
    clean = _write_exec(
        tmp_path / "clean-uv",
        f"""#!/bin/sh
printf '%s\\n' "$@" > "{calls}"
exit 0
""",
    )
    env = os.environ.copy()
    env["PORTFOLIO_LAB_UV"] = str(clean)
    env["PATH"] = f"{tmp_path}:{env['PATH']}"

    result = subprocess.run(
        [str(AGENT_UV), "--version"],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert calls.read_text().splitlines() == ["--version"]


def test_makefile_test_gate_uses_agent_uv() -> None:
    text = MAKEFILE.read_text(encoding="utf-8")
    assert "UV := $(PROJECT_DIR)/scripts/agent_uv.sh" in text
    fast = text.split("test-fast:", 1)[1].split("\ntest-unit:", 1)[0]
    assert "$(UV) run pytest" in fast
    assert "uv run pytest" not in fast.replace("$(UV) run pytest", "")


def test_readme_cold_path_covers_uv_tests_side_tasker_and_rolldown() -> None:
    text = README.read_text(encoding="utf-8")
    assert "scripts/agent_uv.sh" in text
    assert "make test-gate" in text
    assert "TASKER_DISABLE_SCHEDULER=1" in text
    assert "--no-scheduler" in text
    assert "8010" in text or "non-8000" in text
    assert "tasker-side.lock" in text
    assert "@rolldown/binding-linux-x64-gnu" in text
    assert "dev:node" in text
    assert "/root/projects/portfolio-lab" not in text or "Do not assume" in text


def test_package_json_has_node_vite_fallback() -> None:
    data = json.loads(PACKAGE_JSON.read_text(encoding="utf-8"))
    assert data["scripts"]["dev:node"] == "node node_modules/vite/bin/vite.js"
    assert data["scripts"]["build:node"] == "node node_modules/vite/bin/vite.js build"
