from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
from typing import Iterable

from langchain_core.tools import tool

from agent.indexer import SKIP_DIRS, SKIP_EXTENSIONS


# ---------------------------------------------------------------------------
# Workspace root — set explicitly via set_workspace_root() at startup.
# Avoids dependency on os.chdir() / Path.cwd() which can break if the
# working directory is changed externally.
# ---------------------------------------------------------------------------
_workspace_root: Path | None = None
_TOOL_TIMEOUT_SECONDS = 120
_ALLOW_DANGEROUS_SHELL_COMMANDS = False

_BLOCKED_COMMAND_PATTERNS = [
    re.compile(r"(^|[\s;&|])(curl|wget|Invoke-WebRequest|iwr)\b", re.IGNORECASE),
    re.compile(r"(^|[\s;&|])(git\s+clone|scp|ssh|sftp|ftp)\b", re.IGNORECASE),
    re.compile(r"(^|[\s;&|])(Remove-Item|del|erase|rmdir\s+/s|rm\s+-rf|format|diskpart|shutdown|restart-computer)\b", re.IGNORECASE),
    re.compile(r"(^|[\s;&|])(Start-Process\s+powershell|powershell\s+-|cmd\s+/c\s+del)\b", re.IGNORECASE),
]


def set_workspace_root(path: Path) -> None:
    """Set the workspace root used by search_codebase and other tools."""
    global _workspace_root
    _workspace_root = path.resolve()


def set_tool_runtime_policy(
    timeout_seconds: int = 120,
    allow_dangerous_shell_commands: bool = False,
) -> None:
    """Configure command execution safety policy for shell-backed tools."""
    global _TOOL_TIMEOUT_SECONDS, _ALLOW_DANGEROUS_SHELL_COMMANDS
    _TOOL_TIMEOUT_SECONDS = max(1, int(timeout_seconds))
    _ALLOW_DANGEROUS_SHELL_COMMANDS = allow_dangerous_shell_commands


def get_workspace_root() -> Path:
    """Return the workspace root, falling back to cwd if not set explicitly."""
    if _workspace_root is not None:
        return _workspace_root
    return Path.cwd()


# ---------------------------------------------------------------------------
# Critical output patterns to preserve during truncation
# ---------------------------------------------------------------------------
_CRITICAL_PATTERNS = [
    re.compile(r"\d+% tests passed", re.IGNORECASE),
    re.compile(r"The following tests? FAILED", re.IGNORECASE),
    re.compile(r"\d+\s*-\s*\w+.*\((Failed|Timeout|Not Run)\)", re.IGNORECASE),
    re.compile(r"Errors while running CTest", re.IGNORECASE),
    re.compile(r"\[cmd\]="),
    re.compile(r"\[exit_code\]="),
    re.compile(r"error:.*fatal|error:.*undefined|error:.*not found", re.IGNORECASE),
    re.compile(r"CMake Error", re.IGNORECASE),
    re.compile(r"Build FAILED", re.IGNORECASE),
    re.compile(r"Could NOT find", re.IGNORECASE),
    re.compile(r"No rule to make target", re.IGNORECASE),
    re.compile(r"is not recognized as an internal or external command", re.IGNORECASE),
]

_CTEST_SUMMARY_RE = re.compile(
    r"(?P<pct>\d+)% tests passed,\s*(?P<failed>\d+)\s+tests? failed out of\s+(?P<total>\d+)",
    re.IGNORECASE,
)

_ERROR_HINT_PATTERNS = [
    re.compile(r"\berror\b", re.IGNORECASE),
    re.compile(r"\bfatal\b", re.IGNORECASE),
    re.compile(r"\bfailed\b", re.IGNORECASE),
    re.compile(r"not recognized as an internal or external command", re.IGNORECASE),
    re.compile(r"not a directory", re.IGNORECASE),
]


def _truncate_output(text: str, max_chars: int = 3000) -> str:
    """Smart truncation that preserves critical build/test output patterns."""
    lines = text.splitlines()
    if len(lines) <= 150 and len(text) <= max_chars:
        return text

    head = lines[:40]
    tail = lines[-40:]
    head_set = set(range(40))
    tail_set = set(range(max(0, len(lines) - 40), len(lines)))

    critical: list[str] = []
    for i, line in enumerate(lines):
        if i not in head_set and i not in tail_set:
            if any(p.search(line) for p in _CRITICAL_PATTERNS):
                critical.append(line)

    parts = head + ["\n... <truncated> ..."]
    if critical:
        parts += critical
        parts.append("... <end of critical lines> ...")
    parts += tail
    result = "\n".join(parts)
    if len(result) > max_chars:
        result = result[:max_chars] + "\n... <hard truncated>"
    return result


def _normalize_command_for_platform(cmd: str) -> str:
    """Normalize Unix-isms to Windows-compatible commands when running on Windows."""
    if os.name != "nt":
        return cmd

    normalized = cmd
    normalized = re.sub(r"(^|[\s;&|])\./", r"\1", normalized)
    normalized = normalized.replace("$(nproc)", "%NUMBER_OF_PROCESSORS%")
    normalized = re.sub(r"\bexport\s+(\w+)=", r"set \1=", normalized)
    normalized = re.sub(r"\brm\s+-rf\s+(\S+)", r"rmdir /s /q \1", normalized)
    normalized = re.sub(r"\brm\s+-f\s+(\S+)", r"del /f \1", normalized)
    return normalized


def _validate_command_policy(cmd: str) -> str | None:
    """Reject obviously dangerous or out-of-scope commands unless explicitly allowed."""
    if _ALLOW_DANGEROUS_SHELL_COMMANDS:
        return None

    stripped = cmd.strip()
    if not stripped:
        return "empty command"

    for pattern in _BLOCKED_COMMAND_PATTERNS:
        if pattern.search(stripped):
            return "command blocked by safety policy"
    return None


def _build_tool_env() -> dict[str, str]:
    """Provide a minimal, stable environment for build and test commands."""
    env = dict(os.environ)
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("CLICOLOR", "0")
    return env


def _format_command_result(
    normalized_cmd: str,
    exit_code: int,
    stdout: str,
    stderr: str,
    timed_out: bool = False,
) -> str:
    status = "PASS" if exit_code == 0 and not timed_out else "FAIL"
    ctest_summary = _extract_ctest_summary(stdout, stderr)
    error_hint = _extract_error_hint(stdout, stderr)

    metadata: list[str] = [
        f"command={normalized_cmd}",
        f"result={status} (exit_code={exit_code})",
    ]
    if timed_out:
        metadata.append(f"timed_out=true (timeout_seconds={_TOOL_TIMEOUT_SECONDS})")
    if ctest_summary:
        metadata.append(f"tests={ctest_summary}")
    if exit_code != 0 and error_hint:
        metadata.append(f"error_hint={error_hint}")

    combined = (
        "\n".join(metadata)
        + "\n"
        f"[cmd]={normalized_cmd}\n"
        f"[exit_code]={exit_code}\n"
        f"[timed_out]={1 if timed_out else 0}\n"
        f"[stdout]\n{stdout}\n"
        f"[stderr]\n{stderr}"
    )
    return _truncate_output(combined)


def _execute_shell_command_impl(cmd: str) -> str:
    normalized_cmd = _normalize_command_for_platform(cmd)
    policy_error = _validate_command_policy(normalized_cmd)
    if policy_error is not None:
        return _format_command_result(
            normalized_cmd=normalized_cmd,
            exit_code=126,
            stdout="",
            stderr=policy_error,
            timed_out=False,
        )

    try:
        completed = subprocess.run(
            normalized_cmd,
            shell=True,
            text=True,
            capture_output=True,
            cwd=str(get_workspace_root()),
            timeout=_TOOL_TIMEOUT_SECONDS,
            env=_build_tool_env(),
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = (exc.stderr or "")
        if stderr:
            stderr += "\n"
        stderr += f"command timed out after {_TOOL_TIMEOUT_SECONDS} seconds"
        return _format_command_result(
            normalized_cmd=normalized_cmd,
            exit_code=124,
            stdout=stdout,
            stderr=stderr,
            timed_out=True,
        )
    except Exception as exc:
        return _format_command_result(
            normalized_cmd=normalized_cmd,
            exit_code=125,
            stdout="",
            stderr=f"execution error: {type(exc).__name__}: {exc}",
            timed_out=False,
        )

    return _format_command_result(
        normalized_cmd=normalized_cmd,
        exit_code=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        timed_out=False,
    )


def _extract_ctest_summary(stdout: str, stderr: str) -> str | None:
    text = f"{stdout}\n{stderr}"
    match = _CTEST_SUMMARY_RE.search(text)
    if not match:
        return None
    pct = match.group("pct")
    failed = match.group("failed")
    total = match.group("total")
    passed = max(0, int(total) - int(failed))
    return f"{pct}% tests passed, {failed} tests failed out of {total} (passed={passed})"


def _extract_error_hint(stdout: str, stderr: str) -> str | None:
    for line in f"{stderr}\n{stdout}".splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if any(p.search(stripped) for p in _ERROR_HINT_PATTERNS):
            return stripped[:220]
    return None


@tool
def execute_shell_command(cmd: str) -> str:
    """Run a shell command and capture both stdout and stderr with truncation logic."""
    return _execute_shell_command_impl(cmd)


@tool
def list_directory(path: str, depth: int = 1) -> str:
    """List directory contents up to a bounded depth (max 3)."""
    base = Path(path)
    if depth < 0:
        return "depth must be >= 0"
    if depth > 3:
        return "depth too large; max depth is 3"
    if not base.exists() or not base.is_dir():
        return f"invalid directory: {path}"

    output: list[str] = []

    def walk(node: Path, remaining_depth: int, level: int) -> None:
        indent = "  " * level
        try:
            children = sorted(
                node.iterdir(),
                key=lambda c: (not c.is_dir(), c.name.lower()),
            )
        except PermissionError:
            output.append(f"{indent}<permission denied>")
            return
        for child in children:
            suffix = "/" if child.is_dir() else ""
            output.append(f"{indent}{child.name}{suffix}")
            if child.is_dir() and remaining_depth > 0:
                walk(child, remaining_depth - 1, level + 1)

    walk(base, depth, 0)
    return "\n".join(output) if output else "<empty directory>"


@tool
def read_file_chunk(filepath: str, start_line: int, end_line: int) -> str:
    """Read a file chunk by explicit line range (1-indexed, max 250 lines)."""
    if start_line <= 0 or end_line < start_line:
        return "invalid line range"
    if (end_line - start_line) > 250:
        return "line range too large; max chunk size is 250 lines"

    file_path = Path(filepath)
    if not file_path.exists() or not file_path.is_file():
        return f"invalid file: {filepath}"

    try:
        lines = file_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError as e:
        return f"read error: {e}"

    max_line = len(lines)
    start = min(start_line, max_line)
    end = min(end_line, max_line)
    selected = lines[start - 1 : end]

    prefixed = [f"{idx}: {line}" for idx, line in enumerate(selected, start=start)]
    return "\n".join(prefixed) if prefixed else "<no content>"


def _iter_text_files(root: Path, include_build: bool = False) -> Iterable[Path]:
    """Iterate text files under root, skipping binaries and non-essential dirs.

    Uses the shared SKIP_DIRS and SKIP_EXTENSIONS from indexer.py to stay consistent.
    """
    excluded_dirs = set(SKIP_DIRS)
    if include_build:
        excluded_dirs -= {"build", "build-mingw", "build-ninja"}
    for path in root.rglob("*"):
        if path.is_dir():
            continue
        if any(part in excluded_dirs for part in path.parts):
            continue
        if path.suffix.lower() in SKIP_EXTENSIONS:
            continue
        yield path


@tool
def search_codebase(regex_pattern: str) -> str:
    """Search source files with regex and return grep-like path:line:content matches."""
    try:
        pattern = re.compile(regex_pattern, re.IGNORECASE)
    except re.error as exc:
        return f"invalid regex: {exc}"

    root = get_workspace_root()
    matches: list[str] = []
    max_matches = 60

    for file_path in _iter_text_files(root):
        try:
            lines = file_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            continue
        try:
            rel = str(file_path.resolve().relative_to(root.resolve()))
        except ValueError:
            rel = str(file_path)
        for line_num, line in enumerate(lines, start=1):
            if pattern.search(line):
                trimmed_line = line.strip()
                if len(trimmed_line) > 200:
                    trimmed_line = trimmed_line[:200] + " ..."
                matches.append(f"{rel}:{line_num}:{trimmed_line}")
                if len(matches) >= max_matches:
                    return "\n".join(matches) + "\n... <match limit reached>"

    return "\n".join(matches) if matches else "<no matches>"


ALL_TOOLS = [
    execute_shell_command,
    list_directory,
    read_file_chunk,
    search_codebase,
]
