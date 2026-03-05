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
    ".cppm": "c++",
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


SKIP_DIRS = {
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

SKIP_EXTENSIONS = {
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

# Patterns for file purpose detection
_MODULE_EXPORT_RE = re.compile(r"^\s*export\s+module\s+([\w.:]+)", re.MULTILINE)
_MODULE_IMPORT_RE = re.compile(r"^\s*(?:export\s+)?import\s+([\w.:]+)", re.MULTILINE)
_MODULE_DECL_RE = re.compile(r"^\s*module\s+([\w.:]+)", re.MULTILINE)
_PRAGMA_ONCE_RE = re.compile(r"^\s*#\s*pragma\s+once", re.MULTILINE)
_INCLUDE_GUARD_RE = re.compile(r"^\s*#\s*ifndef\s+(\w+_H[_P]*)\b", re.MULTILINE)
_MAIN_RE = re.compile(r"\bint\s+main\s*\(", re.MULTILINE)
_TEST_CASE_RE = re.compile(r"\b(?:TEST_CASE|TEST_F|TEST|SECTION|CATCH_TEST_CASE)\s*\(", re.MULTILINE)
_CMAKE_PROJECT_RE = re.compile(r"\bproject\s*\(", re.MULTILINE | re.IGNORECASE)
_CMAKE_ADD_LIB_RE = re.compile(r"\badd_library\s*\(", re.MULTILINE | re.IGNORECASE)
_CMAKE_ADD_EXE_RE = re.compile(r"\badd_executable\s*\(", re.MULTILINE | re.IGNORECASE)
_INCLUDE_RE = re.compile(r'^\s*#\s*include\s*[<"]([^>"]+)[>"]', re.MULTILINE)
_BRIEF_RE = re.compile(r"(?:@brief|\\brief)\s+(.+)", re.MULTILINE)


def build_codebase_index(workspace_path: Path) -> CodebaseIndex:
    """Walk the workspace and build a file manifest + symbol table + file purpose map."""
    root = str(workspace_path.resolve())
    files: list[FileEntry] = []
    symbols: list[SymbolEntry] = []

    for dirpath, dirnames, filenames in os.walk(workspace_path):
        # Prune skipped directories in-place
        dirnames[:] = [
            d for d in dirnames
            if d not in SKIP_DIRS
        ]

        for filename in filenames:
            filepath = Path(dirpath) / filename
            ext = filepath.suffix.lower()

            if ext in SKIP_EXTENSIONS:
                continue

            try:
                stat = filepath.stat()
            except OSError:
                continue

            # Skip very large files (>500KB)
            if stat.st_size > 500_000:
                continue

            lang = _LANG_MAP.get(ext, "other")

            # Read file content for summary, purpose, and declarations
            content = ""
            try:
                content = filepath.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                pass

            # Build rich summary and purpose
            summary = _build_rich_summary(content, filepath, lang)
            purpose = _detect_file_purpose(content, filepath, lang)
            declarations = _extract_declarations(content, lang)

            rel_path = str(filepath.relative_to(workspace_path))
            files.append(
                FileEntry(
                    path=rel_path,
                    language=lang,
                    size=stat.st_size,
                    summary=summary,
                    purpose=purpose,
                    declarations=declarations,
                )
            )

            # Extract symbols from C/C++ files
            if lang in ("c++", "c") and stat.st_size < 200_000 and content:
                _extract_symbols_from_content(content, rel_path, symbols)

    return CodebaseIndex(root=root, files=files, symbols=symbols)


# ---------------------------------------------------------------------------
# Rich summary builder (replaces "first line only")
# ---------------------------------------------------------------------------

def _build_rich_summary(content: str, filepath: Path, lang: str) -> str:
    """Build a multi-sentence summary by parsing key declarations."""
    parts: list[str] = []

    # First non-empty line
    for line in content.splitlines()[:5]:
        stripped = line.strip()
        if stripped and not stripped.startswith("//") and not stripped.startswith("/*"):
            parts.append(stripped[:120])
            break

    # @brief docstrings
    brief_match = _BRIEF_RE.search(content[:2000])
    if brief_match:
        parts.append(brief_match.group(1).strip()[:120])

    # C++20 module export
    mod_export = _MODULE_EXPORT_RE.search(content)
    if mod_export:
        parts.append(f"Exports module: {mod_export.group(1)}")

    # Key includes (first 5)
    includes = _INCLUDE_RE.findall(content[:3000])[:5]
    if includes:
        parts.append(f"Includes: {', '.join(includes)}")

    # Class/struct declarations (first 3)
    classes = _CPP_CLASS_RE.findall(content)[:3]
    if classes:
        parts.append(f"Declares: {', '.join(classes)}")

    # CMake project info
    if filepath.name.lower() == "cmakelists.txt":
        if _CMAKE_PROJECT_RE.search(content):
            parts.append("CMake project definition")
        if _CMAKE_ADD_LIB_RE.search(content):
            parts.append("Defines library target")
        if _CMAKE_ADD_EXE_RE.search(content):
            parts.append("Defines executable target")

    return " | ".join(parts)[:300] if parts else filepath.name


# ---------------------------------------------------------------------------
# File purpose detection
# ---------------------------------------------------------------------------

def _detect_file_purpose(content: str, filepath: Path, lang: str) -> str:
    """Compute a purpose tag for the file based on structural patterns."""
    purposes: list[str] = []

    name = filepath.name.lower()

    # C++20 module interface
    if _MODULE_EXPORT_RE.search(content):
        mod = _MODULE_EXPORT_RE.search(content)
        purposes.append(f"C++20 module interface ({mod.group(1)})")
    elif _MODULE_DECL_RE.search(content):
        purposes.append("C++20 module implementation")

    # Header file
    if _PRAGMA_ONCE_RE.search(content) or _INCLUDE_GUARD_RE.search(content):
        purposes.append("header file")

    # Entry point
    if _MAIN_RE.search(content):
        purposes.append("executable entry point")

    # Test file
    if _TEST_CASE_RE.search(content):
        purposes.append("test file")

    # CMake
    if name == "cmakelists.txt":
        if _CMAKE_PROJECT_RE.search(content):
            purposes.append("CMake project root")
        else:
            purposes.append("CMake build script")

    # Build files
    if name in ("makefile", "meson.build", "build.bazel", "module.bazel", "package.swift"):
        purposes.append("build configuration")

    # Documentation
    if lang == "markdown":
        purposes.append("documentation")

    if not purposes:
        if lang in ("c++", "c"):
            purposes.append("source file")
        else:
            purposes.append(f"{lang} file")

    return "; ".join(purposes)


# ---------------------------------------------------------------------------
# Declaration extraction
# ---------------------------------------------------------------------------

def _extract_declarations(content: str, lang: str) -> list[str]:
    """Extract top-level declaration names for the purpose map."""
    if lang not in ("c++", "c"):
        return []

    decls: list[str] = []

    # Module exports
    for m in _MODULE_EXPORT_RE.finditer(content):
        decls.append(f"export module {m.group(1)}")

    # Classes/structs
    for m in _CPP_CLASS_RE.finditer(content):
        decls.append(f"class/struct {m.group(1)}")

    # Functions (top-level only, first 10)
    count = 0
    for m in _CPP_FUNCTION_RE.finditer(content):
        name = m.group(1)
        if name in ("if", "for", "while", "switch", "return", "catch", "sizeof", "alignof"):
            continue
        decls.append(f"function {name}")
        count += 1
        if count >= 10:
            break

    return decls[:20]  # cap at 20 declarations


def _extract_symbols_from_content(
    content: str, rel_path: str, symbols: list[SymbolEntry]
) -> None:
    """Extract function, class/struct, and macro symbols from file content."""
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


# ---------------------------------------------------------------------------
# Path-aware file reference detection (used by retrieve_context in nodes.py)
# ---------------------------------------------------------------------------

# Regex to detect file-like references in user text
_FILE_REF_PATTERNS = [
    # Full or partial paths with slashes: src/modules/json.cppm
    re.compile(r"(?:[\w./\\-]+/[\w./\\-]+\.[\w]+)"),
    # Bare filenames with extensions: json.cppm, CMakeLists.txt
    re.compile(r"\b([\w-]+\.(?:cpp|cppm|hpp|h|c|cc|cxx|cmake|txt|json|py|md|yaml|yml|swift|bzl))\b", re.IGNORECASE),
]


def detect_file_references(user_input: str, index: CodebaseIndex) -> list[FileEntry]:
    """Detect explicit file path/name references in user input and match to index entries.

    Returns matching FileEntry objects from the index.
    """
    candidates: set[str] = set()

    for pattern in _FILE_REF_PATTERNS:
        for match in pattern.finditer(user_input):
            ref = match.group(0).strip().replace("\\", "/")
            candidates.add(ref)

    if not candidates:
        return []

    matched: list[FileEntry] = []
    seen: set[str] = set()

    for ref in candidates:
        ref_lower = ref.lower()
        for fe in index.files:
            if fe.path in seen:
                continue
            path_lower = fe.path.lower().replace("\\", "/")
            # Exact path match
            if path_lower == ref_lower or path_lower.endswith("/" + ref_lower):
                matched.append(fe)
                seen.add(fe.path)
            # Bare filename match
            elif ref_lower == Path(fe.path).name.lower():
                matched.append(fe)
                seen.add(fe.path)

    return matched


def detect_directory_references(user_input: str, index: CodebaseIndex) -> list[FileEntry]:
    """Detect directory path references in user input and return all files in those directories.

    E.g. "src/modules" -> return all files under src/modules/.
    """
    # Look for directory-like references (paths without file extensions)
    dir_pattern = re.compile(r"(?:[\w./\\-]+/[\w./\\-]+)(?!\.\w)")
    candidates: set[str] = set()

    for match in dir_pattern.finditer(user_input):
        ref = match.group(0).strip().replace("\\", "/").rstrip("/")
        # Ignore if it looks like a file (has a dot-extension at the end)
        if re.search(r"\.\w{1,5}$", ref):
            continue
        candidates.add(ref)

    if not candidates:
        return []

    matched: list[FileEntry] = []
    seen: set[str] = set()

    for ref in candidates:
        ref_lower = ref.lower()
        for fe in index.files:
            if fe.path in seen:
                continue
            parent = str(Path(fe.path).parent).replace("\\", "/").lower()
            if parent == ref_lower or parent.endswith("/" + ref_lower):
                matched.append(fe)
                seen.add(fe.path)

    return matched


# ---------------------------------------------------------------------------
# Enhanced search with fuzzy path matching
# ---------------------------------------------------------------------------

def search_index(
    index: CodebaseIndex,
    query: str,
    max_results: int = 15,
) -> list[FileEntry | SymbolEntry]:
    """Enhanced keyword search with fuzzy path matching and purpose-aware scoring."""
    query_lower = query.lower()
    keywords = query_lower.split()

    # Extract potential filename tokens for fuzzy matching
    filename_tokens = [
        kw for kw in keywords
        if "." in kw or "/" in kw or "\\" in kw
    ]

    scored: list[tuple[float, FileEntry | SymbolEntry]] = []

    for fe in index.files:
        score = 0.0
        path_lower = fe.path.lower().replace("\\", "/")
        text = f"{fe.path} {fe.summary} {fe.language} {fe.purpose}".lower()

        for kw in keywords:
            if kw in text:
                score += 1.0
            if kw in path_lower:
                score += 0.5  # bonus for path match

        # Fuzzy path matching: filename substring match
        for ft in filename_tokens:
            ft_clean = ft.replace("\\", "/").lower()
            # Direct substring in path
            if ft_clean in path_lower:
                score += 3.0
            # Bare filename match (e.g. "json.cppm" matches "src/modules/json.cppm")
            elif ft_clean == Path(fe.path).name.lower():
                score += 4.0
            # Stem match (e.g. "json" matches "json.cppm")
            elif ft_clean == Path(fe.path).stem.lower():
                score += 2.0

        # Bonus for purpose matches
        if fe.purpose:
            purpose_lower = fe.purpose.lower()
            for kw in keywords:
                if kw in purpose_lower:
                    score += 1.0

        # Bonus for declaration matches
        for decl in fe.declarations:
            decl_lower = decl.lower()
            for kw in keywords:
                if kw in decl_lower:
                    score += 1.5

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
            purpose_tag = f" [{fe.purpose}]" if fe.purpose else ""
            lines.append(f"  {fe.path} ({fe.language}, {fe.size}B){purpose_tag}")
            count += 1
        if len(entries) > 5:
            lines.append(f"  ... +{len(entries)-5} more in {d}/")
            count += len(entries) - 5

    return "\n".join(lines)
