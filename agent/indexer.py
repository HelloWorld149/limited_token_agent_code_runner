from __future__ import annotations

import os
import re
from pathlib import Path

from agent.state import CodebaseIndex, FileEntry, SymbolEntry


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------
_LANG_MAP: dict[str, str] = {
    ".h": "c++",
    ".hpp": "c++",
    ".cpp": "c++",
    ".cc": "c++",
    ".cxx": "c++",
    ".c": "c",
    ".py": "python",
    ".cmake": "cmake",
    ".txt": "text",
    ".md": "markdown",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".sh": "shell",
    ".bat": "batch",
    ".ps1": "powershell",
    ".swift": "swift",
    ".bzl": "bazel",
}

_SKIP_DIRS = {
    ".git",
    "build",
    "build-mingw",
    "build-ninja",
    "__pycache__",
    "node_modules",
    ".venv",
    "dist",
    "out",
}

_SKIP_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".pdf", ".zip",
    ".exe", ".dll", ".o", ".obj", ".a", ".so", ".dylib",
    ".woff", ".woff2", ".ttf", ".eot", ".ico",
}

# Regex patterns to extract C/C++ symbols
_CPP_FUNCTION_RE = re.compile(
    r"^\s*(?:[\w:*&<>\[\]]+\s+)+(\w+)\s*\([^)]*\)\s*(?:const\s*)?(?:noexcept\s*)?(?:override\s*)?[{;]",
    re.MULTILINE,
)
_CPP_CLASS_RE = re.compile(
    r"^\s*(?:template\s*<[^>]*>\s*)?(?:class|struct)\s+(\w+)",
    re.MULTILINE,
)
_CPP_MACRO_RE = re.compile(
    r"^\s*#\s*define\s+(\w+)",
    re.MULTILINE,
)


def build_codebase_index(workspace_path: Path) -> CodebaseIndex:
    """Walk the workspace and build a file manifest + symbol table."""
    root = str(workspace_path.resolve())
    files: list[FileEntry] = []
    symbols: list[SymbolEntry] = []

    for dirpath, dirnames, filenames in os.walk(workspace_path):
        # Prune skipped directories in-place
        dirnames[:] = [
            d for d in dirnames
            if d not in _SKIP_DIRS
        ]

        for filename in filenames:
            filepath = Path(dirpath) / filename
            ext = filepath.suffix.lower()

            if ext in _SKIP_EXTENSIONS:
                continue

            try:
                stat = filepath.stat()
            except OSError:
                continue

            # Skip very large files (>500KB)
            if stat.st_size > 500_000:
                continue

            lang = _LANG_MAP.get(ext, "other")

            # Read first line for summary
            summary = ""
            try:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    first_line = f.readline().strip()
                    summary = first_line[:120] if first_line else ""
            except OSError:
                pass

            rel_path = str(filepath.relative_to(workspace_path))
            files.append(
                FileEntry(
                    path=rel_path,
                    language=lang,
                    size=stat.st_size,
                    summary=summary,
                )
            )

            # Extract symbols from C/C++ files
            if lang in ("c++", "c") and stat.st_size < 200_000:
                _extract_symbols(filepath, rel_path, symbols)

    return CodebaseIndex(root=root, files=files, symbols=symbols)


def _extract_symbols(
    filepath: Path, rel_path: str, symbols: list[SymbolEntry]
) -> None:
    """Extract function, class/struct, and macro symbols from a C/C++ file."""
    try:
        content = filepath.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return

    lines = content.split("\n")

    for match in _CPP_CLASS_RE.finditer(content):
        name = match.group(1)
        line = content[: match.start()].count("\n") + 1
        symbols.append(SymbolEntry(name=name, kind="class", file=rel_path, line=line))

    for match in _CPP_MACRO_RE.finditer(content):
        name = match.group(1)
        # Skip include guards and common boilerplate
        if name.startswith("_") and name.endswith("_"):
            continue
        if name in ("NULL", "TRUE", "FALSE"):
            continue
        line = content[: match.start()].count("\n") + 1
        symbols.append(SymbolEntry(name=name, kind="macro", file=rel_path, line=line))

    # Only extract top-level functions (heuristic: not indented)
    for match in _CPP_FUNCTION_RE.finditer(content):
        name = match.group(1)
        # Skip common false positives
        if name in ("if", "for", "while", "switch", "return", "catch", "sizeof", "alignof"):
            continue
        line = content[: match.start()].count("\n") + 1
        symbols.append(SymbolEntry(name=name, kind="function", file=rel_path, line=line))


def search_index(
    index: CodebaseIndex,
    query: str,
    max_results: int = 15,
) -> list[FileEntry | SymbolEntry]:
    """Keyword search across file manifest and symbol table."""
    query_lower = query.lower()
    keywords = query_lower.split()

    scored: list[tuple[float, FileEntry | SymbolEntry]] = []

    for fe in index.files:
        score = 0.0
        text = f"{fe.path} {fe.summary} {fe.language}".lower()
        for kw in keywords:
            if kw in text:
                score += 1.0
            if kw in fe.path.lower():
                score += 0.5  # bonus for path match
        if score > 0:
            scored.append((score, fe))

    for se in index.symbols:
        score = 0.0
        text = f"{se.name} {se.kind} {se.file}".lower()
        for kw in keywords:
            if kw in text:
                score += 1.5  # symbols are more relevant
            if kw == se.name.lower():
                score += 2.0  # exact name match
        if score > 0:
            scored.append((score, se))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in scored[:max_results]]


def format_file_manifest_summary(index: CodebaseIndex, max_entries: int = 30) -> str:
    """Format a compact manifest for injection into LLM context."""
    lines = [f"Workspace: {index.root}  ({len(index.files)} files, {len(index.symbols)} symbols)"]
    # Group by directory
    dirs: dict[str, list[FileEntry]] = {}
    for fe in index.files:
        d = str(Path(fe.path).parent)
        dirs.setdefault(d, []).append(fe)

    count = 0
    for d in sorted(dirs):
        if count >= max_entries:
            lines.append(f"  ... and {len(index.files) - count} more files")
            break
        entries = dirs[d]
        for fe in entries[:5]:
            lines.append(f"  {fe.path} ({fe.language}, {fe.size}B)")
            count += 1
        if len(entries) > 5:
            lines.append(f"  ... +{len(entries)-5} more in {d}/")
            count += len(entries) - 5

    return "\n".join(lines)
