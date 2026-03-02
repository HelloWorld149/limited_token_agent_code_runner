from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
from typing import Iterable

from langchain_core.tools import tool


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


def _truncate_output(text: str) -> str:
    """Smart truncation that preserves critical build/test output patterns."""
    lines = text.splitlines()
    if len(lines) <= 150 and len(text) <= 3000:
        return text

    # Partition into head, tail, and critical lines from the middle
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
    return "\n".join(parts)


def _safe_relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except Exception:
        return str(path)


@tool
def execute_shell_command(cmd: str) -> str:
    """Run a shell command and capture both stdout and stderr with truncation logic."""
    normalized_cmd = _normalize_command_for_platform(cmd)
    try:
        completed = subprocess.run(
            normalized_cmd,
            shell=True,
            text=True,
            capture_output=True,
            timeout=300,  # 5 minute timeout
        )
    except subprocess.TimeoutExpired:
        return (
            f"[cmd]={normalized_cmd}\n"
            f"[exit_code]=124\n"
            f"[stdout]\n<command timed out after 300 seconds>\n"
            f"[stderr]\n<timeout>"
        )
    combined = (
        f"[cmd]={normalized_cmd}\n"
        f"[exit_code]={completed.returncode}\n"
        f"[stdout]\n{completed.stdout}\n"
        f"[stderr]\n{completed.stderr}"
    )
    return _truncate_output(combined)


def _normalize_command_for_platform(cmd: str) -> str:
    """Normalize Unix-isms to Windows-compatible commands when running on Windows."""
    if os.name != "nt":
        return cmd

    normalized = cmd
    # Strip ./ prefix (Windows doesn't need it)
    normalized = re.sub(r"(^|[\s;&|])\./", r"\1", normalized)
    # Replace $(nproc) with Windows equivalent
    normalized = normalized.replace("$(nproc)", "%NUMBER_OF_PROCESSORS%")
    # Replace 'export VAR=val' with 'set VAR=val'
    normalized = re.sub(r"\bexport\s+(\w+)=", r"set \1=", normalized)
    # Replace forward-slash paths in common build directories
    normalized = normalized.replace("build/tests/", "build\\tests\\")
    # Replace 'rm -rf' with 'rmdir /s /q' for directory removal
    normalized = re.sub(r"\brm\s+-rf\s+(\S+)", r"rmdir /s /q \1", normalized)
    # Replace 'rm -f' with 'del /f'
    normalized = re.sub(r"\brm\s+-f\s+(\S+)", r"del /f \1", normalized)
    return normalized


@tool
def list_directory(path: str, depth: int = 1) -> str:
    """List directory contents up to a bounded depth."""
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
        children = sorted(node.iterdir(), key=lambda child: (not child.is_dir(), child.name.lower()))
        for child in children:
            suffix = "/" if child.is_dir() else ""
            output.append(f"{indent}{child.name}{suffix}")
            if child.is_dir() and remaining_depth > 0:
                walk(child, remaining_depth - 1, level + 1)

    walk(base, depth, 0)
    return "\n".join(output) if output else "<empty directory>"


@tool
def read_file_chunk(filepath: str, start_line: int, end_line: int) -> str:
    """Read a file chunk by explicit line range."""
    if start_line <= 0 or end_line < start_line:
        return "invalid line range"
    if (end_line - start_line) > 250:
        return "line range too large; max chunk size is 250 lines"

    file_path = Path(filepath)
    if not file_path.exists() or not file_path.is_file():
        return f"invalid file: {filepath}"

    lines = file_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    max_line = len(lines)
    start = min(start_line, max_line)
    end = min(end_line, max_line)
    selected = lines[start - 1 : end]

    prefixed = [f"{index}: {line}" for index, line in enumerate(selected, start=start)]
    return "\n".join(prefixed) if prefixed else "<no content>"


def _iter_text_files(root: Path, include_build: bool = False) -> Iterable[Path]:
    excluded_dirs = {".git", "dist", "out", "node_modules", "__pycache__"}
    if not include_build:
        excluded_dirs.add("build")
    for path in root.rglob("*"):
        if path.is_dir():
            continue
        if any(part in excluded_dirs for part in path.parts):
            continue
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".pdf", ".zip", ".exe", ".dll", ".o", ".obj"}:
            continue
        yield path


@tool
def search_codebase(regex_pattern: str) -> str:
    """Search files with regex and return grep-like path:line:content matches."""
    try:
        pattern = re.compile(regex_pattern)
    except re.error as exc:
        return f"invalid regex: {exc}"

    root = Path.cwd()
    matches: list[str] = []
    max_matches = 80

    for file_path in _iter_text_files(root):
        try:
            lines = file_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            continue
        rel = _safe_relative(file_path, root)
        for line_num, line in enumerate(lines, start=1):
            if pattern.search(line):
                trimmed_line = line.strip()
                if len(trimmed_line) > 220:
                    trimmed_line = trimmed_line[:220] + " ..."
                matches.append(f"{rel}:{line_num}:{trimmed_line}")
                if len(matches) >= max_matches:
                    return "\n".join(matches) + "\n... <match limit reached>"

    return "\n".join(matches) if matches else "<no matches>"


@tool
def search_build_artifacts(regex_pattern: str) -> str:
    """Search cmake build artifacts (CMakeCache, error logs, test logs) for debugging."""
    try:
        pattern = re.compile(regex_pattern)
    except re.error as exc:
        return f"invalid regex: {exc}"

    root = Path.cwd()
    build_dir = root / "build"
    if not build_dir.exists():
        return "<no build directory found>"

    # Only search known useful build artifact files
    useful_patterns = [
        "CMakeCache.txt",
        "CMakeError.log",
        "CMakeOutput.log",
        "CMakeConfigureLog.yaml",
        "LastTest.log",
        "CTestTestfile.cmake",
    ]

    matches: list[str] = []
    max_matches = 100

    for path in build_dir.rglob("*"):
        if path.is_dir():
            continue
        if not any(path.name == p for p in useful_patterns):
            continue
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            continue
        rel = _safe_relative(path, root)
        for line_num, line in enumerate(lines, start=1):
            if pattern.search(line):
                matches.append(f"{rel}:{line_num}:{line}")
                if len(matches) >= max_matches:
                    return "\n".join(matches) + "\n... <match limit reached>"

    return "\n".join(matches) if matches else "<no matches in build artifacts>"


ALL_TOOLS = [
    execute_shell_command,
    list_directory,
    read_file_chunk,
    search_codebase,
    search_build_artifacts,
]
