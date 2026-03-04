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
            timeout=300,
        )
    except subprocess.TimeoutExpired:
        return f"[cmd]={normalized_cmd}\n[exit_code]=124\n[stderr]\nCommand timed out after 300s"
    combined = (
        f"[cmd]={normalized_cmd}\n"
        f"[exit_code]={completed.returncode}\n"
        f"[stdout]\n{completed.stdout}\n"
        f"[stderr]\n{completed.stderr}"
    )
    return _truncate_output(combined)


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
    """Iterate text files under root, skipping binaries and non-essential dirs."""
    excluded_dirs = {".git", "dist", "out", "node_modules", "__pycache__"}
    if not include_build:
        excluded_dirs.update({"build", "build-mingw", "build-ninja"})
    for path in root.rglob("*"):
        if path.is_dir():
            continue
        if any(part in excluded_dirs for part in path.parts):
            continue
        if path.suffix.lower() in {
            ".png", ".jpg", ".jpeg", ".gif", ".pdf", ".zip",
            ".exe", ".dll", ".o", ".obj", ".a", ".so",
        }:
            continue
        yield path


@tool
def search_codebase(regex_pattern: str) -> str:
    """Search source files with regex and return grep-like path:line:content matches."""
    try:
        pattern = re.compile(regex_pattern, re.IGNORECASE)
    except re.error as exc:
        return f"invalid regex: {exc}"

    root = Path.cwd()
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
