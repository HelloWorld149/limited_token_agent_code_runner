"""Tests for agent/tools.py — output truncation, command normalization, workspace root."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess

import pytest

from agent.tools import (
    _build_tool_env,
    _execute_shell_command_impl,
    _extract_ctest_summary,
    _extract_error_hint,
    _normalize_command_for_platform,
    _truncate_output,
    get_workspace_root,
    set_tool_runtime_policy,
    set_workspace_root,
)


# ---------------------------------------------------------------------------
# _truncate_output
# ---------------------------------------------------------------------------

class TestTruncateOutput:
    def test_short_output_unchanged(self) -> None:
        text = "line1\nline2\nline3"
        assert _truncate_output(text) == text

    def test_long_output_truncated(self) -> None:
        lines = [f"line {i}" for i in range(500)]
        text = "\n".join(lines)
        result = _truncate_output(text, max_chars=3000)
        assert len(result) <= 3000 + 50  # allow small overhead from hard truncation marker

    def test_critical_patterns_preserved(self) -> None:
        head = [f"head {i}" for i in range(50)]
        middle = ["some noise"] * 100
        critical = ["CMake Error at CMakeLists.txt:10"]
        tail = [f"tail {i}" for i in range(50)]
        text = "\n".join(head + middle + critical + tail)
        result = _truncate_output(text)
        assert "CMake Error" in result

    def test_hard_truncation_marker(self) -> None:
        text = "x" * 10000
        result = _truncate_output(text, max_chars=500)
        assert "hard truncated" in result


# ---------------------------------------------------------------------------
# _normalize_command_for_platform
# ---------------------------------------------------------------------------

class TestNormalizeCommandForPlatform:
    def test_no_change_on_non_windows(self) -> None:
        if os.name == "nt":
            pytest.skip("Only runs on non-Windows")
        assert _normalize_command_for_platform("./build.sh") == "./build.sh"

    def test_strip_dot_slash_on_windows(self) -> None:
        if os.name != "nt":
            pytest.skip("Only runs on Windows")
        result = _normalize_command_for_platform("./my_script")
        assert not result.startswith("./")

    def test_export_to_set(self) -> None:
        if os.name != "nt":
            pytest.skip("Only runs on Windows")
        result = _normalize_command_for_platform("export FOO=bar")
        assert "set FOO=bar" in result

    def test_rm_rf_to_rmdir(self) -> None:
        if os.name != "nt":
            pytest.skip("Only runs on Windows")
        result = _normalize_command_for_platform("rm -rf build")
        assert "rmdir" in result


# ---------------------------------------------------------------------------
# _extract_ctest_summary
# ---------------------------------------------------------------------------

class TestExtractCtestSummary:
    def test_valid_summary(self) -> None:
        stdout = "100% tests passed, 0 tests failed out of 42"
        result = _extract_ctest_summary(stdout, "")
        assert result is not None
        assert "42" in result
        assert "0" in result

    def test_failed_summary(self) -> None:
        stdout = "95% tests passed, 2 tests failed out of 40"
        result = _extract_ctest_summary(stdout, "")
        assert result is not None
        assert "2 tests failed" in result

    def test_no_summary(self) -> None:
        assert _extract_ctest_summary("no test output", "") is None


# ---------------------------------------------------------------------------
# _extract_error_hint
# ---------------------------------------------------------------------------

class TestExtractErrorHint:
    def test_finds_error(self) -> None:
        result = _extract_error_hint("", "fatal error: file not found")
        assert result is not None
        assert "fatal" in result.lower()

    def test_no_error(self) -> None:
        assert _extract_error_hint("all good", "no issues") is None


# ---------------------------------------------------------------------------
# set_workspace_root / get_workspace_root
# ---------------------------------------------------------------------------

class TestWorkspaceRoot:
    def test_set_and_get(self, tmp_path: Path) -> None:
        set_workspace_root(tmp_path)
        assert get_workspace_root() == tmp_path.resolve()

    def test_raises_when_unset(self) -> None:
        # Reset to None via module internals for test isolation
        import agent.tools as tools_mod
        old = tools_mod._workspace_root
        tools_mod._workspace_root = None
        try:
            with pytest.raises(RuntimeError, match="workspace root is not configured"):
                get_workspace_root()
        finally:
            tools_mod._workspace_root = old


class TestExecuteShellCommandSafety:
    def test_build_tool_env_filters_sensitive_variables(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "secret")
        monkeypatch.setenv("PATH", os.environ.get("PATH", ""))
        env = _build_tool_env()
        assert "OPENAI_API_KEY" not in env
        assert "PATH" in env

    def test_blocks_dangerous_commands_by_default(self) -> None:
        set_tool_runtime_policy(timeout_seconds=30, allow_dangerous_shell_commands=False)
        result = _execute_shell_command_impl("curl https://example.com")
        assert "[exit_code]=126" in result
        assert "blocked by safety policy" in result

    def test_rejects_non_allowlisted_executable(self, tmp_path: Path) -> None:
        set_workspace_root(tmp_path)
        set_tool_runtime_policy(timeout_seconds=30, allow_dangerous_shell_commands=False)
        result = _execute_shell_command_impl("unknown-tool --help")
        assert "[exit_code]=126" in result
        assert "not permitted by the safety allowlist" in result

    def test_safe_execution_uses_shell_false(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        import agent.tools as tools_mod

        set_workspace_root(tmp_path)
        set_tool_runtime_policy(timeout_seconds=30, allow_dangerous_shell_commands=False)

        observed: dict[str, object] = {}

        monkeypatch.setattr(tools_mod.shutil, "which", lambda *_args, **_kwargs: str(tmp_path / "python.exe"))

        def _fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
            observed["args"] = args[0]
            observed["shell"] = kwargs["shell"]
            return subprocess.CompletedProcess(args=args[0], returncode=0, stdout="ok", stderr="")

        monkeypatch.setattr(subprocess, "run", _fake_run)
        result = _execute_shell_command_impl("python --version")

        assert observed["shell"] is False
        assert isinstance(observed["args"], list)
        assert "[exit_code]=0" in result

    def test_reports_timeout(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        set_workspace_root(tmp_path)
        set_tool_runtime_policy(timeout_seconds=1, allow_dangerous_shell_commands=True)

        def _raise_timeout(*args: object, **kwargs: object) -> object:
            raise subprocess.TimeoutExpired(
                cmd="python -c \"print('x')\"",
                timeout=1,
                output="partial stdout",
                stderr="partial stderr",
            )

        monkeypatch.setattr(subprocess, "run", _raise_timeout)
        result = _execute_shell_command_impl("python -c \"print('x')\"")
        assert "[exit_code]=124" in result
        assert "[timed_out]=1" in result
        assert "command timed out after 1 seconds" in result
