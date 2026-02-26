from __future__ import annotations

from pathlib import Path
import re
import subprocess
from typing import Iterable

from langchain_core.tools import tool


def _truncate_output(text: str) -> str:
    lines = text.splitlines()
    if len(lines) <= 100 and len(text) <= 1000:
        return text

    head = lines[:50]
    tail = lines[-50:] if len(lines) > 50 else []
    marker = ["... <output truncated> ..."]
    merged = head + marker + tail
    return "\n".join(merged)


def _safe_relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except Exception:
        return str(path)


@tool
def execute_shell_command(cmd: str) -> str:
    """Run a shell command and capture both stdout and stderr with truncation logic."""
    completed = subprocess.run(
        cmd,
        shell=True,
        text=True,
        capture_output=True,
    )
    combined = (
        f"[cmd]={cmd}\n"
        f"[exit_code]={completed.returncode}\n"
        f"[stdout]\n{completed.stdout}\n"
        f"[stderr]\n{completed.stderr}"
    )
    return _truncate_output(combined)


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


def _iter_text_files(root: Path) -> Iterable[Path]:
    excluded_dirs = {".git", "build", "dist", "out", "node_modules", "__pycache__"}
    for path in root.rglob("*"):
        if path.is_dir():
            continue
        if any(part in excluded_dirs for part in path.parts):
            continue
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".pdf", ".zip", ".exe", ".dll"}:
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
    max_matches = 200

    for file_path in _iter_text_files(root):
        try:
            lines = file_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            continue
        rel = _safe_relative(file_path, root)
        for line_num, line in enumerate(lines, start=1):
            if pattern.search(line):
                matches.append(f"{rel}:{line_num}:{line}")
                if len(matches) >= max_matches:
                    return "\n".join(matches) + "\n... <match limit reached>"

    return "\n".join(matches) if matches else "<no matches>"


ALL_TOOLS = [
    execute_shell_command,
    list_directory,
    read_file_chunk,
    search_codebase,
]
