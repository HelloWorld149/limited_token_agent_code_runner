from __future__ import annotations

import gzip
import json
import os
import re
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from agent.state import ChunkEntry, CodebaseIndex, FileEntry, SymbolEntry


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
    ".png", ".jpg", ".jpeg", ".gif", ".tif", ".tiff", ".bmp", ".pdf", ".zip",
    ".exe", ".dll", ".o", ".obj", ".a", ".so", ".dylib",
    ".woff", ".woff2", ".ttf", ".eot", ".ico",
}

_INDEX_CACHE_VERSION = 2
_INDEX_CACHE_FILE_NAME = ".codebase_index_cache_v2.json.gz"
_MAX_INDEX_FILE_BYTES = 25_000_000
_STREAMING_INDEX_THRESHOLD_BYTES = 2_000_000
_MAX_SYMBOL_SCAN_BYTES = 1_500_000
_MIN_CHUNK_LINES = 24

_MAX_CHUNK_LINES_BY_LANG: dict[str, int] = {
    "c++": 140,
    "c": 140,
    "python": 120,
    "markdown": 80,
    "cmake": 90,
    "json": 100,
    "yaml": 100,
    "text": 100,
}

# Regex patterns to extract C/C++ symbols
_CPP_FUNCTION_RE = re.compile(
    r"^\s*(?:[\w:*&<>\[\]~,]+\s+)+(\w+)\s*\([^)]*\)\s*(?:const\s*)?(?:noexcept\s*)?(?:override\s*)?(?:->\s*[\w:<>*&\s]+)?\s*[{;]",
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
_CPP_NAMESPACE_RE = re.compile(r"^\s*namespace\s+([\w:]+)", re.MULTILINE)
_CPP_BOUNDARY_RE = re.compile(
    r"^(?:#\s*(?:include|define|if|ifdef|ifndef|elif|else|endif|pragma)\b"
    r"|(?:export\s+module|module)\b"
    r"|(?:template\s*<[^>]*>\s*)?(?:class|struct|namespace|enum|union)\s+\w+"
    r"|NLOHMANN_JSON_NAMESPACE_BEGIN\b|NLOHMANN_JSON_NAMESPACE_END\b)",
)
_CPP_FUNCTION_LINE_RE = re.compile(
    r"^(?:template\s*<[^>]*>\s*)?(?:(?:inline|constexpr|static|virtual|explicit|friend|extern)\s+)*"
    r"[\w:<>~*&\s,]+\b([A-Za-z_]\w*)\s*\([^;]*\)\s*(?:const\s*)?(?:noexcept\s*)?(?:override\s*)?(?:->\s*[\w:<>*&\s]+)?\s*(?:\{|$)"
)

_PY_DECL_RE = re.compile(r"^(class|def)\s+([A-Za-z_]\w*)\b")
_MARKDOWN_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")
_CMAKE_COMMAND_RE = re.compile(
    r"^(project|add_library|add_executable|target_[a-z_]+|find_package|include|set|option|if|elseif|else|endif|function|macro|foreach|install)\s*\(",
    re.IGNORECASE,
)
_YAML_HEADING_RE = re.compile(r"^([A-Za-z0-9_.-]+):\s*(?:#.*)?$")

# Patterns for file purpose detection
_MODULE_EXPORT_RE = re.compile(r"^\s*export\s+module\s+([\w.:]+)", re.MULTILINE)
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
_CPP_FALSE_POSITIVES = {"if", "for", "while", "switch", "return", "catch", "sizeof", "alignof"}


def build_codebase_index(
    workspace_path: Path,
    use_persistent_cache: bool = True,
    cache_directory: Path | None = None,
) -> CodebaseIndex:
    """Walk the workspace and build file, symbol, chunk, and summary indexes."""
    resolved_workspace = workspace_path.resolve()
    root = str(resolved_workspace)
    files: list[FileEntry] = []
    symbols: list[SymbolEntry] = []
    chunks: list[ChunkEntry] = []
    chunks_by_file: dict[str, list[int]] = {}

    cached_records = (
        _load_cached_records(resolved_workspace, cache_directory)
        if use_persistent_cache
        else {}
    )
    records_to_cache: dict[str, dict[str, object]] = {}

    for filepath in _iter_workspace_files(resolved_workspace):
        ext = filepath.suffix.lower()
        lang = _LANG_MAP.get(ext, "other")

        try:
            stat = filepath.stat()
        except OSError:
            continue

        if stat.st_size > _MAX_INDEX_FILE_BYTES:
            continue

        rel_path = str(filepath.relative_to(resolved_workspace)).replace("\\", "/")
        fingerprint = _file_fingerprint(stat)
        cached_record = cached_records.get(rel_path)
        if cached_record is not None and cached_record.get("fingerprint") == fingerprint:
            file_entry, file_chunks, file_symbols = _deserialize_cached_record(cached_record)
        else:
            file_entry, file_chunks, file_symbols = _index_single_file(
                filepath=filepath,
                rel_path=rel_path,
                stat=stat,
                lang=lang,
            )
        records_to_cache[rel_path] = _serialize_cached_record(
            fingerprint=fingerprint,
            file_entry=file_entry,
            file_chunks=file_chunks,
            file_symbols=file_symbols,
        )
        files.append(file_entry)
        symbols.extend(file_symbols)

        chunk_indexes: list[int] = []
        for chunk in file_chunks:
            chunk_indexes.append(len(chunks))
            chunks.append(chunk)
        if chunk_indexes:
            chunks_by_file[rel_path] = chunk_indexes

    files.sort(key=lambda file_entry: file_entry.path)
    symbols.sort(key=lambda symbol: (symbol.file, symbol.line, symbol.name))
    repository_summary = _build_repository_summary(files)

    if use_persistent_cache:
        _save_cached_records(resolved_workspace, cache_directory, repository_summary, records_to_cache)

    return CodebaseIndex(
        root=root,
        files=files,
        symbols=symbols,
        chunks=chunks,
        chunks_by_file=chunks_by_file,
        repository_summary=repository_summary,
    )


def _iter_workspace_files(workspace_path: Path) -> Iterable[Path]:
    for dirpath, dirnames, filenames in os.walk(workspace_path):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        for filename in sorted(filenames):
            if filename == _INDEX_CACHE_FILE_NAME:
                continue
            filepath = Path(dirpath) / filename
            if filepath.suffix.lower() in SKIP_EXTENSIONS:
                continue
            yield filepath


def _cache_file_path(workspace_path: Path, cache_directory: Path | None) -> Path:
    if cache_directory is not None:
        return cache_directory.resolve() / _INDEX_CACHE_FILE_NAME
    return workspace_path.parent / ".cache" / _INDEX_CACHE_FILE_NAME


def _file_fingerprint(stat: os.stat_result) -> dict[str, int]:
    return {
        "size": int(stat.st_size),
        "mtime_ns": int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1_000_000_000))),
    }


def _load_cached_records(
    workspace_path: Path,
    cache_directory: Path | None,
) -> dict[str, dict[str, object]]:
    cache_path = _cache_file_path(workspace_path, cache_directory)
    if not cache_path.exists():
        return {}

    try:
        with gzip.open(cache_path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}

    if payload.get("version") != _INDEX_CACHE_VERSION:
        return {}

    records = payload.get("records")
    if not isinstance(records, dict):
        return {}
    return {str(path): record for path, record in records.items() if isinstance(record, dict)}


def _save_cached_records(
    workspace_path: Path,
    cache_directory: Path | None,
    repository_summary: str,
    records: dict[str, dict[str, object]],
) -> None:
    cache_path = _cache_file_path(workspace_path, cache_directory)
    temp_path = cache_path.with_name(f"{cache_path.name}.tmp")
    payload = {
        "version": _INDEX_CACHE_VERSION,
        "repository_summary": repository_summary,
        "records": records,
    }
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(temp_path, "wt", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False)
        os.replace(temp_path, cache_path)
    except (OSError, TypeError, ValueError):
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass
        return


def _serialize_cached_record(
    fingerprint: dict[str, int],
    file_entry: FileEntry,
    file_chunks: list[ChunkEntry],
    file_symbols: list[SymbolEntry],
) -> dict[str, object]:
    return {
        "fingerprint": fingerprint,
        "file_entry": asdict(file_entry),
        "chunks": [asdict(chunk) for chunk in file_chunks],
        "symbols": [asdict(symbol) for symbol in file_symbols],
    }


def _deserialize_cached_record(
    record: dict[str, object],
) -> tuple[FileEntry, list[ChunkEntry], list[SymbolEntry]]:
    raw_file_entry = record.get("file_entry") or {}
    raw_chunks = record.get("chunks") or []
    raw_symbols = record.get("symbols") or []

    file_entry = FileEntry(**raw_file_entry)
    file_chunks = [ChunkEntry(**chunk) for chunk in raw_chunks if isinstance(chunk, dict)]
    file_symbols = [SymbolEntry(**symbol) for symbol in raw_symbols if isinstance(symbol, dict)]
    return file_entry, file_chunks, file_symbols


def _index_single_file(
    filepath: Path,
    rel_path: str,
    stat: os.stat_result,
    lang: str,
) -> tuple[FileEntry, list[ChunkEntry], list[SymbolEntry]]:
    if stat.st_size >= _STREAMING_INDEX_THRESHOLD_BYTES:
        return _index_large_text_file(filepath, rel_path, stat, lang)

    try:
        content = filepath.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        content = ""

    purpose = _detect_file_purpose(content, filepath, lang)
    file_chunks = _chunk_file_content(rel_path, content, lang)
    declarations = _collect_file_declarations(file_chunks, content, lang)
    summary = _build_file_summary(
        file_chunks=file_chunks,
        content=content,
        filepath=filepath,
        lang=lang,
        purpose=purpose,
        declarations=declarations,
    )

    file_entry = FileEntry(
        path=rel_path,
        language=lang,
        size=stat.st_size,
        summary=summary,
        purpose=purpose,
        declarations=declarations,
        chunk_count=len(file_chunks),
    )

    file_symbols: list[SymbolEntry] = []
    if content and stat.st_size <= _MAX_SYMBOL_SCAN_BYTES:
        if lang in ("c++", "c"):
            _extract_symbols_from_content(content, rel_path, file_symbols)
        elif lang == "python":
            _extract_python_symbols_from_content(content, rel_path, file_symbols)

    return file_entry, file_chunks, file_symbols


def _index_large_text_file(
    filepath: Path,
    rel_path: str,
    stat: os.stat_result,
    lang: str,
) -> tuple[FileEntry, list[ChunkEntry], list[SymbolEntry]]:
    content_preview, file_chunks = _chunk_large_text_file_streaming(filepath, rel_path, lang)
    purpose = _detect_file_purpose(content_preview, filepath, lang)
    declarations = _collect_file_declarations(file_chunks, content_preview, lang)
    summary = _build_file_summary(
        file_chunks=file_chunks,
        content=content_preview,
        filepath=filepath,
        lang=lang,
        purpose=purpose,
        declarations=declarations,
    )

    file_entry = FileEntry(
        path=rel_path,
        language=lang,
        size=stat.st_size,
        summary=summary,
        purpose=purpose,
        declarations=declarations,
        chunk_count=len(file_chunks),
    )
    return file_entry, file_chunks, []


def _chunk_large_text_file_streaming(
    filepath: Path,
    rel_path: str,
    lang: str,
    preview_char_budget: int = 20_000,
) -> tuple[str, list[ChunkEntry]]:
    max_chunk_lines = _max_chunk_lines(lang)
    max_buffer_lines = max_chunk_lines + _MIN_CHUNK_LINES
    preview_parts: list[str] = []
    preview_chars = 0
    buffer_lines: list[str] = []
    buffer_start_line = 1
    chunks: list[ChunkEntry] = []

    for line in _read_text_lines_streaming(filepath):
        if preview_chars < preview_char_budget:
            remaining_chars = preview_char_budget - preview_chars
            preview_fragment = line[:remaining_chars]
            preview_parts.append(preview_fragment)
            preview_chars += len(preview_fragment) + 1

        buffer_lines.append(line)
        while len(buffer_lines) >= max_buffer_lines:
            flush_count = _select_streaming_flush_line_count(buffer_lines, lang, max_chunk_lines)
            _append_streaming_chunk(
                chunks=chunks,
                rel_path=rel_path,
                lang=lang,
                start_line=buffer_start_line,
                lines=buffer_lines[:flush_count],
            )
            buffer_lines = buffer_lines[flush_count:]
            buffer_start_line += flush_count

    if buffer_lines:
        if chunks:
            previous_chunk = chunks[-1]
            previous_chunk_length = previous_chunk.end_line - previous_chunk.start_line + 1
            combined_length = previous_chunk_length + len(buffer_lines)
            if len(buffer_lines) < _MIN_CHUNK_LINES and combined_length <= max_chunk_lines + (_MIN_CHUNK_LINES // 2):
                chunks.pop()
                _append_streaming_chunk(
                    chunks=chunks,
                    rel_path=rel_path,
                    lang=lang,
                    start_line=previous_chunk.start_line,
                    lines=previous_chunk.text.splitlines() + buffer_lines,
                )
            else:
                _append_streaming_chunk(
                    chunks=chunks,
                    rel_path=rel_path,
                    lang=lang,
                    start_line=buffer_start_line,
                    lines=buffer_lines,
                )
        else:
            _append_streaming_chunk(
                chunks=chunks,
                rel_path=rel_path,
                lang=lang,
                start_line=buffer_start_line,
                lines=buffer_lines,
            )

    return "\n".join(preview_parts), chunks


def _read_text_lines_streaming(filepath: Path) -> Iterable[str]:
    try:
        with filepath.open("r", encoding="utf-8", errors="ignore") as handle:
            for raw_line in handle:
                yield raw_line.rstrip("\n\r")
    except OSError:
        return


def _select_streaming_flush_line_count(
    lines: list[str],
    lang: str,
    max_chunk_lines: int,
) -> int:
    candidate_lines = lines[: max_chunk_lines + _MIN_CHUNK_LINES]
    boundaries = _detect_semantic_boundaries(candidate_lines, lang)
    split_at = _find_split_point(candidate_lines, 1, max_chunk_lines, boundaries)
    if split_at < _MIN_CHUNK_LINES:
        return max_chunk_lines
    return min(split_at, max_chunk_lines)


def _append_streaming_chunk(
    chunks: list[ChunkEntry],
    rel_path: str,
    lang: str,
    start_line: int,
    lines: list[str],
) -> None:
    if not lines:
        return

    chunk_text = "\n".join(lines)
    if not chunk_text.strip():
        return

    local_boundaries = _detect_semantic_boundaries(lines, lang)
    declarations = _extract_declarations(chunk_text, lang)
    symbol_names = _extract_chunk_symbol_names(chunk_text, lang)
    heading = _chunk_heading(local_boundaries, lines, 1, len(lines))
    summary = _build_chunk_summary(chunk_text, Path(rel_path), lang, heading, declarations, symbol_names)
    chunks.append(
        ChunkEntry(
            file_path=rel_path,
            language=lang,
            start_line=start_line,
            end_line=start_line + len(lines) - 1,
            summary=summary,
            heading=heading,
            symbol_names=symbol_names,
            declarations=declarations,
            text=chunk_text,
        )
    )


def _preview_from_lines(lines: list[str], max_chars: int = 20000) -> str:
    if not lines:
        return ""

    preview_parts: list[str] = []
    total_chars = 0
    for line in lines:
        preview_parts.append(line)
        total_chars += len(line) + 1
        if total_chars >= max_chars:
            break
    return "\n".join(preview_parts)


# ---------------------------------------------------------------------------
# Chunking and hierarchical summaries
# ---------------------------------------------------------------------------

def _chunk_file_content(rel_path: str, content: str, lang: str) -> list[ChunkEntry]:
    """Split file content into semantic chunks with cached local summaries."""
    if not content.strip():
        return []

    lines = content.splitlines()
    return _chunk_lines(rel_path, lines, lang)


def _chunk_lines(rel_path: str, lines: list[str], lang: str) -> list[ChunkEntry]:
    if not lines:
        return []

    boundaries = _detect_semantic_boundaries(lines, lang)
    ranges = _build_chunk_ranges(lines, boundaries, _max_chunk_lines(lang))
    chunks: list[ChunkEntry] = []

    for start_line, end_line in ranges:
        chunk_text = "\n".join(lines[start_line - 1 : end_line])
        if not chunk_text.strip():
            continue
        heading = _chunk_heading(boundaries, lines, start_line, end_line)
        declarations = _extract_declarations(chunk_text, lang)
        symbol_names = _extract_chunk_symbol_names(chunk_text, lang)
        summary = _build_chunk_summary(chunk_text, Path(rel_path), lang, heading, declarations, symbol_names)
        chunks.append(
            ChunkEntry(
                file_path=rel_path,
                language=lang,
                start_line=start_line,
                end_line=end_line,
                summary=summary,
                heading=heading,
                symbol_names=symbol_names,
                declarations=declarations,
                text=chunk_text,
            )
        )

    if not chunks:
        content = "\n".join(lines)
        summary = _build_chunk_summary(content, Path(rel_path), lang, "", [], [])
        chunks.append(
            ChunkEntry(
                file_path=rel_path,
                language=lang,
                start_line=1,
                end_line=len(lines),
                summary=summary,
                declarations=_extract_declarations(content, lang),
                symbol_names=_extract_chunk_symbol_names(content, lang),
                text=content,
            )
        )

    return chunks


def _detect_semantic_boundaries(lines: list[str], lang: str) -> dict[int, str]:
    """Return start-line -> heading labels for semantic sections."""
    boundaries: dict[int, str] = {1: _normalize_heading(lines[0]) if lines else ""}

    for idx, raw_line in enumerate(lines, start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue

        if lang == "markdown":
            match = _MARKDOWN_HEADING_RE.match(stripped)
            if match:
                boundaries[idx] = f"{match.group(1)} {match.group(2).strip()}"
            continue

        if lang == "cmake":
            match = _CMAKE_COMMAND_RE.match(stripped)
            if match:
                boundaries[idx] = f"{match.group(1).lower()}()"
            continue

        if lang == "python":
            if raw_line == raw_line.lstrip():
                match = _PY_DECL_RE.match(stripped)
                if match:
                    boundaries[idx] = f"{match.group(1)} {match.group(2)}"
            continue

        if lang in ("c++", "c"):
            indent = len(raw_line) - len(raw_line.lstrip())
            if stripped.startswith("#"):
                boundaries[idx] = _normalize_heading(stripped)
            elif indent <= 4 and _CPP_BOUNDARY_RE.match(stripped):
                boundaries[idx] = _normalize_heading(stripped)
            elif indent <= 2 and _CPP_FUNCTION_LINE_RE.match(stripped):
                boundaries[idx] = _normalize_heading(stripped)
            continue

        if lang in ("yaml", "json", "text"):
            match = _YAML_HEADING_RE.match(stripped)
            if match and len(match.group(1)) > 1:
                boundaries[idx] = match.group(1)

    return boundaries


def _build_chunk_ranges(
    lines: list[str],
    boundaries: dict[int, str],
    max_chunk_lines: int,
) -> list[tuple[int, int]]:
    total_lines = len(lines)
    if total_lines == 0:
        return []

    anchor_lines = sorted({line for line in boundaries if 1 <= line <= total_lines} | {1})
    ranges: list[tuple[int, int]] = []

    for idx, start_line in enumerate(anchor_lines):
        next_start = anchor_lines[idx + 1] if idx + 1 < len(anchor_lines) else total_lines + 1
        segment_end = next_start - 1
        if segment_end < start_line:
            continue
        ranges.extend(
            _split_large_range(
                lines=lines,
                start_line=start_line,
                end_line=segment_end,
                max_chunk_lines=max_chunk_lines,
                boundaries=boundaries,
            )
        )

    return _merge_short_ranges(ranges, max_chunk_lines)


def _split_large_range(
    lines: list[str],
    start_line: int,
    end_line: int,
    max_chunk_lines: int,
    boundaries: dict[int, str],
) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    cursor = start_line

    while cursor <= end_line:
        candidate_end = min(end_line, cursor + max_chunk_lines - 1)
        if candidate_end >= end_line:
            ranges.append((cursor, end_line))
            break

        split_at = _find_split_point(lines, cursor, candidate_end, boundaries)
        if split_at < cursor:
            split_at = candidate_end

        ranges.append((cursor, split_at))
        cursor = split_at + 1

    return ranges


def _find_split_point(
    lines: list[str],
    start_line: int,
    candidate_end: int,
    boundaries: dict[int, str],
) -> int:
    min_line = start_line + _MIN_CHUNK_LINES - 1

    for boundary_line in sorted(boundaries, reverse=True):
        if min_line < boundary_line <= candidate_end:
            return boundary_line - 1

    for line_no in range(candidate_end, min_line - 1, -1):
        stripped = lines[line_no - 1].strip()
        if not stripped:
            return line_no
        if stripped in ("}", "};"):
            return line_no

    return candidate_end


def _merge_short_ranges(
    ranges: list[tuple[int, int]],
    max_chunk_lines: int,
) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []

    for start_line, end_line in ranges:
        if not merged:
            merged.append((start_line, end_line))
            continue

        prev_start, prev_end = merged[-1]
        prev_len = prev_end - prev_start + 1
        curr_len = end_line - start_line + 1
        combined_len = end_line - prev_start + 1

        if curr_len < _MIN_CHUNK_LINES and combined_len <= max_chunk_lines + (_MIN_CHUNK_LINES // 2):
            merged[-1] = (prev_start, end_line)
        elif prev_len < _MIN_CHUNK_LINES and combined_len <= max_chunk_lines + (_MIN_CHUNK_LINES // 2):
            merged[-1] = (prev_start, end_line)
        else:
            merged.append((start_line, end_line))

    return merged


def _chunk_heading(
    boundaries: dict[int, str],
    lines: list[str],
    start_line: int,
    end_line: int,
) -> str:
    if start_line in boundaries and boundaries[start_line]:
        return boundaries[start_line][:120]

    for line_no in range(start_line, min(end_line, start_line + 6) + 1):
        label = boundaries.get(line_no, "")
        if label:
            return label[:120]

    for raw_line in lines[start_line - 1 : min(end_line, start_line + 5)]:
        stripped = raw_line.strip()
        if stripped and not stripped.startswith(("//", "/*", "*", "#")):
            return _normalize_heading(stripped)
    return ""


def _build_chunk_summary(
    content: str,
    filepath: Path,
    lang: str,
    heading: str,
    declarations: list[str],
    symbol_names: list[str],
) -> str:
    parts: list[str] = []

    if heading:
        parts.append(heading)

    brief_match = _BRIEF_RE.search(content[:2000])
    if brief_match:
        parts.append(brief_match.group(1).strip()[:120])

    includes = _INCLUDE_RE.findall(content[:3000])[:4]
    if includes:
        parts.append(f"Includes: {', '.join(includes)}")

    if declarations:
        parts.append(f"Declares: {', '.join(declarations[:4])}")
    elif symbol_names:
        parts.append(f"Symbols: {', '.join(symbol_names[:4])}")

    first_line = _first_meaningful_line(content)
    if first_line:
        parts.append(first_line[:120])

    summary = " | ".join(_unique(parts))[:280]
    return summary or filepath.name


def _build_file_summary(
    file_chunks: list[ChunkEntry],
    content: str,
    filepath: Path,
    lang: str,
    purpose: str,
    declarations: list[str],
) -> str:
    if not file_chunks:
        return _build_rich_summary(content, filepath, lang)

    parts: list[str] = []
    if purpose:
        parts.append(purpose)
    if declarations:
        parts.append(f"Declares: {', '.join(declarations[:5])}")
    for chunk in file_chunks[:4]:
        if chunk.summary:
            parts.append(chunk.summary)
    if len(file_chunks) > 4:
        parts.append(f"{len(file_chunks)} semantic chunks")
    return " | ".join(_unique(parts))[:420]


def _collect_file_declarations(
    file_chunks: list[ChunkEntry],
    content: str,
    lang: str,
) -> list[str]:
    declarations: list[str] = []
    for chunk in file_chunks:
        declarations.extend(chunk.declarations)
    declarations = _unique(declarations)[:20]
    if declarations:
        return declarations
    return _extract_declarations(content, lang)[:20]


def _build_repository_summary(files: list[FileEntry]) -> str:
    """Build a repository-level digest from file summaries."""
    if not files:
        return "empty workspace"

    dir_counts: dict[str, int] = {}
    lang_counts: dict[str, int] = {}
    chunked_files = sorted((fe for fe in files if fe.chunk_count > 1), key=lambda fe: fe.chunk_count, reverse=True)

    for file_entry in files:
        top_dir = file_entry.path.split("/", 1)[0] if "/" in file_entry.path else "."
        dir_counts[top_dir] = dir_counts.get(top_dir, 0) + 1
        lang_counts[file_entry.language] = lang_counts.get(file_entry.language, 0) + 1

    top_dirs = ", ".join(f"{name}={count}" for name, count in sorted(dir_counts.items(), key=lambda item: item[1], reverse=True)[:4])
    top_langs = ", ".join(f"{name}={count}" for name, count in sorted(lang_counts.items(), key=lambda item: item[1], reverse=True)[:4])
    notable = ", ".join(f"{Path(file_entry.path).name}({file_entry.chunk_count})" for file_entry in chunked_files[:3])

    parts = [f"dirs: {top_dirs}", f"languages: {top_langs}"]
    if notable:
        parts.append(f"chunked: {notable}")
    return " | ".join(parts)[:420]


# ---------------------------------------------------------------------------
# Summary builders and declaration extraction
# ---------------------------------------------------------------------------

def _build_rich_summary(content: str, filepath: Path, lang: str) -> str:
    """Build a fallback summary for files that do not produce semantic chunks."""
    parts: list[str] = []

    first_line = _first_meaningful_line(content)
    if first_line:
        parts.append(first_line[:120])

    brief_match = _BRIEF_RE.search(content[:2000])
    if brief_match:
        parts.append(brief_match.group(1).strip()[:120])

    mod_export = _MODULE_EXPORT_RE.search(content)
    if mod_export:
        parts.append(f"Exports module: {mod_export.group(1)}")

    includes = _INCLUDE_RE.findall(content[:3000])[:5]
    if includes:
        parts.append(f"Includes: {', '.join(includes)}")

    classes = _CPP_CLASS_RE.findall(content)[:3]
    if classes:
        parts.append(f"Declares: {', '.join(classes)}")

    if filepath.name.lower() == "cmakelists.txt":
        if _CMAKE_PROJECT_RE.search(content):
            parts.append("CMake project definition")
        if _CMAKE_ADD_LIB_RE.search(content):
            parts.append("Defines library target")
        if _CMAKE_ADD_EXE_RE.search(content):
            parts.append("Defines executable target")

    return " | ".join(_unique(parts))[:300] if parts else filepath.name


def _detect_file_purpose(content: str, filepath: Path, lang: str) -> str:
    """Compute a purpose tag for the file based on structural patterns."""
    purposes: list[str] = []

    name = filepath.name.lower()

    if _MODULE_EXPORT_RE.search(content):
        mod = _MODULE_EXPORT_RE.search(content)
        if mod:
            purposes.append(f"C++20 module interface ({mod.group(1)})")
    elif _MODULE_DECL_RE.search(content):
        purposes.append("C++20 module implementation")

    if _PRAGMA_ONCE_RE.search(content) or _INCLUDE_GUARD_RE.search(content):
        purposes.append("header file")

    if _MAIN_RE.search(content):
        purposes.append("executable entry point")

    if _TEST_CASE_RE.search(content):
        purposes.append("test file")

    if name == "cmakelists.txt":
        if _CMAKE_PROJECT_RE.search(content):
            purposes.append("CMake project root")
        else:
            purposes.append("CMake build script")

    if name in ("makefile", "meson.build", "build.bazel", "module.bazel", "package.swift"):
        purposes.append("build configuration")

    if lang == "markdown":
        purposes.append("documentation")

    if not purposes:
        if lang in ("c++", "c"):
            purposes.append("source file")
        else:
            purposes.append(f"{lang} file")

    return "; ".join(_unique(purposes))


def _extract_declarations(content: str, lang: str) -> list[str]:
    """Extract top-level declaration names for the purpose map."""
    if lang in ("c++", "c"):
        decls: list[str] = []
        for match in _MODULE_EXPORT_RE.finditer(content):
            decls.append(f"export module {match.group(1)}")
        for match in _CPP_NAMESPACE_RE.finditer(content):
            decls.append(f"namespace {match.group(1)}")
        for match in _CPP_CLASS_RE.finditer(content):
            decls.append(f"class/struct {match.group(1)}")
        count = 0
        for match in _CPP_FUNCTION_RE.finditer(content):
            name = match.group(1)
            if name in _CPP_FALSE_POSITIVES:
                continue
            decls.append(f"function {name}")
            count += 1
            if count >= 10:
                break
        return _unique(decls)[:20]

    if lang == "python":
        decls = [f"{kind} {name}" for kind, name in _extract_python_decl_pairs(content)]
        return _unique(decls)[:20]

    if lang == "markdown":
        headings = [f"heading {match.group(2).strip()}" for match in _MARKDOWN_HEADING_RE.finditer(content)]
        return _unique(headings)[:12]

    if lang == "cmake":
        commands = [f"command {match.group(1).lower()}" for match in _CMAKE_COMMAND_RE.finditer(content)]
        return _unique(commands)[:20]

    if lang in ("yaml", "json", "text"):
        headings = [match.group(1) for match in _YAML_HEADING_RE.finditer(content)]
        return _unique(headings)[:12]

    return []


def _extract_chunk_symbol_names(content: str, lang: str) -> list[str]:
    if lang in ("c++", "c"):
        names = [match.group(1) for match in _CPP_CLASS_RE.finditer(content)]
        names.extend(match.group(1) for match in _CPP_FUNCTION_RE.finditer(content) if match.group(1) not in _CPP_FALSE_POSITIVES)
        names.extend(match.group(1) for match in _CPP_MACRO_RE.finditer(content))
        return _unique(names)[:12]

    if lang == "python":
        return _unique(name for _, name in _extract_python_decl_pairs(content))[:12]

    if lang == "markdown":
        return _unique(match.group(2).strip() for match in _MARKDOWN_HEADING_RE.finditer(content))[:8]

    if lang == "cmake":
        return _unique(match.group(1).lower() for match in _CMAKE_COMMAND_RE.finditer(content))[:12]

    return []


def _extract_python_decl_pairs(content: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for raw_line in content.splitlines():
        if raw_line != raw_line.lstrip():
            continue
        match = _PY_DECL_RE.match(raw_line.strip())
        if match:
            pairs.append((match.group(1), match.group(2)))
    return pairs


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
        if name.startswith("_") and name.endswith("_"):
            continue
        if name in ("NULL", "TRUE", "FALSE"):
            continue
        line = content[: match.start()].count("\n") + 1
        symbols.append(SymbolEntry(name=name, kind="macro", file=rel_path, line=line))

    for match in _CPP_FUNCTION_RE.finditer(content):
        name = match.group(1)
        if name in _CPP_FALSE_POSITIVES:
            continue
        line = content[: match.start()].count("\n") + 1
        symbols.append(SymbolEntry(name=name, kind="function", file=rel_path, line=line))


def _extract_python_symbols_from_content(
    content: str, rel_path: str, symbols: list[SymbolEntry]
) -> None:
    lines = content.splitlines()
    for kind, name in _extract_python_decl_pairs(content):
        marker = f"{kind} {name}"
        line = next(
            (idx for idx, raw_line in enumerate(lines, start=1) if raw_line.strip().startswith(marker)),
            1,
        )
        symbols.append(SymbolEntry(name=name, kind=kind, file=rel_path, line=line))


# ---------------------------------------------------------------------------
# Path-aware file reference detection (used by retrieve_context in nodes.py)
# ---------------------------------------------------------------------------

_FILE_REF_PATTERNS = [
    re.compile(r"(?:[\w./\\-]+/[\w./\\-]+\.[\w]+)"),
    re.compile(r"\b([\w-]+\.(?:cpp|cppm|hpp|h|c|cc|cxx|cmake|txt|json|py|md|yaml|yml|swift|bzl))\b", re.IGNORECASE),
]


def detect_file_references(user_input: str, index: CodebaseIndex) -> list[FileEntry]:
    """Detect explicit file path/name references in user input and match to index entries."""
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
        for file_entry in index.files:
            if file_entry.path in seen:
                continue
            path_lower = file_entry.path.lower().replace("\\", "/")
            if path_lower == ref_lower or path_lower.endswith("/" + ref_lower):
                matched.append(file_entry)
                seen.add(file_entry.path)
            elif ref_lower == Path(file_entry.path).name.lower():
                matched.append(file_entry)
                seen.add(file_entry.path)

    return matched


def detect_directory_references(user_input: str, index: CodebaseIndex) -> list[FileEntry]:
    """Detect directory path references in user input and return files in those directories."""
    dir_pattern = re.compile(r"(?:[\w./\\-]+/[\w./\\-]+)(?!\.\w)")
    candidates: set[str] = set()

    for match in dir_pattern.finditer(user_input):
        ref = match.group(0).strip().replace("\\", "/").rstrip("/")
        if re.search(r"\.\w{1,5}$", ref):
            continue
        candidates.add(ref)

    if not candidates:
        return []

    matched: list[FileEntry] = []
    seen: set[str] = set()

    for ref in candidates:
        ref_lower = ref.lower()
        for file_entry in index.files:
            if file_entry.path in seen:
                continue
            parent = str(Path(file_entry.path).parent).replace("\\", "/").lower()
            if parent == ref_lower or parent.endswith("/" + ref_lower):
                matched.append(file_entry)
                seen.add(file_entry.path)

    return matched


# ---------------------------------------------------------------------------
# Search helpers
# ---------------------------------------------------------------------------

def search_index(
    index: CodebaseIndex,
    query: str,
    max_results: int = 15,
) -> list[FileEntry | SymbolEntry]:
    """Enhanced keyword search with fuzzy path matching and purpose-aware scoring."""
    keywords = _query_keywords(query)
    filename_tokens = [kw for kw in keywords if "." in kw or "/" in kw or "\\" in kw]

    scored: list[tuple[float, FileEntry | SymbolEntry]] = []

    for file_entry in index.files:
        score = 0.0
        path_lower = file_entry.path.lower().replace("\\", "/")
        text = f"{file_entry.path} {file_entry.summary} {file_entry.language} {file_entry.purpose} {' '.join(file_entry.declarations)}".lower()

        for kw in keywords:
            if kw in text:
                score += 1.0
            if kw in path_lower:
                score += 0.75

        for filename_token in filename_tokens:
            ft_clean = filename_token.replace("\\", "/").lower()
            if ft_clean in path_lower:
                score += 3.0
            elif ft_clean == Path(file_entry.path).name.lower():
                score += 4.0
            elif ft_clean == Path(file_entry.path).stem.lower():
                score += 2.0

        if file_entry.chunk_count > 1:
            score += 0.25

        if score > 0:
            scored.append((score, file_entry))

    for symbol in index.symbols:
        score = 0.0
        text = f"{symbol.name} {symbol.kind} {symbol.file}".lower()
        for kw in keywords:
            if kw in text:
                score += 1.5
            if kw == symbol.name.lower():
                score += 2.5
        if score > 0:
            scored.append((score, symbol))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [item for _, item in scored[:max_results]]


def search_chunks(
    index: CodebaseIndex,
    query: str,
    max_results: int = 10,
    allowed_files: set[str] | None = None,
) -> list[ChunkEntry]:
    """Search semantic chunks using path hints, summaries, symbols, and lexical matches."""
    keywords = _query_keywords(query)
    if not keywords and allowed_files:
        keywords = [Path(path).name.lower() for path in allowed_files]

    symbol_hits_by_file: dict[str, list[SymbolEntry]] = {}
    for symbol in index.symbols:
        symbol_name = symbol.name.lower()
        if any(kw == symbol_name or kw in symbol_name for kw in keywords):
            symbol_hits_by_file.setdefault(symbol.file, []).append(symbol)

    file_lookup = {file_entry.path: file_entry for file_entry in index.files}
    scored: list[tuple[float, int]] = []

    for idx, chunk in enumerate(index.chunks):
        if allowed_files and chunk.file_path not in allowed_files:
            continue

        score = 0.0
        path_lower = chunk.file_path.lower()
        text = (
            f"{chunk.file_path} {chunk.heading} {chunk.summary} "
            f"{' '.join(chunk.symbol_names)} {' '.join(chunk.declarations)}"
        ).lower()
        lexical_body = chunk.text.lower()[:4000]
        file_entry = file_lookup.get(chunk.file_path)

        for kw in keywords:
            if kw in text:
                score += 1.5
            if kw in path_lower:
                score += 1.0
            if kw in lexical_body:
                score += 0.5
            if chunk.heading and kw in chunk.heading.lower():
                score += 1.0
            if any(kw == name.lower() for name in chunk.symbol_names):
                score += 3.0
            if any(kw in decl.lower() for decl in chunk.declarations):
                score += 1.5
            if file_entry and kw in file_entry.summary.lower():
                score += 0.5

        for symbol in symbol_hits_by_file.get(chunk.file_path, []):
            if chunk.start_line <= symbol.line <= chunk.end_line:
                score += 4.0

        if file_entry and file_entry.purpose and any(kw in file_entry.purpose.lower() for kw in keywords):
            score += 0.5

        if score > 0:
            scored.append((score, idx))

    scored.sort(key=lambda item: (item[0], -index.chunks[item[1]].start_line), reverse=True)
    return [index.chunks[idx] for _, idx in scored[:max_results]]


def get_file_chunks(index: CodebaseIndex, rel_path: str) -> list[ChunkEntry]:
    """Return cached semantic chunks for a file in line order."""
    return [index.chunks[idx] for idx in index.chunks_by_file.get(rel_path, []) if 0 <= idx < len(index.chunks)]


def expand_chunk_window(
    index: CodebaseIndex,
    seed_chunks: list[ChunkEntry],
    neighbor_depth: int = 1,
    max_chunks: int = 12,
) -> list[ChunkEntry]:
    """Add adjacent chunks around seed hits to capture cross-boundary answers."""
    selected: list[ChunkEntry] = []
    seen: set[tuple[str, int, int]] = set()

    def add(chunk: ChunkEntry) -> None:
        key = (chunk.file_path, chunk.start_line, chunk.end_line)
        if key in seen or len(selected) >= max_chunks:
            return
        seen.add(key)
        selected.append(chunk)

    for chunk in seed_chunks:
        add(chunk)
        file_chunks = get_file_chunks(index, chunk.file_path)
        try:
            pos = next(
                i for i, candidate in enumerate(file_chunks)
                if candidate.start_line == chunk.start_line and candidate.end_line == chunk.end_line
            )
        except StopIteration:
            continue

        for delta in range(1, neighbor_depth + 1):
            if pos - delta >= 0:
                add(file_chunks[pos - delta])
            if pos + delta < len(file_chunks):
                add(file_chunks[pos + delta])

    selected.sort(key=lambda chunk: (chunk.file_path, chunk.start_line))
    return selected[:max_chunks]


def format_file_outline(index: CodebaseIndex, file_entry: FileEntry, max_chunks: int = 6) -> str:
    """Format file-level outline using cached chunk summaries and line ranges."""
    chunks = get_file_chunks(index, file_entry.path)
    purpose_tag = f" [{file_entry.purpose}]" if file_entry.purpose else ""
    lines = [
        (
            f"{file_entry.path}{purpose_tag} "
            f"({file_entry.language}, {file_entry.size}B, chunks={file_entry.chunk_count})"
        ),
        f"Summary: {file_entry.summary}",
    ]
    for chunk in chunks[:max_chunks]:
        heading = f" {chunk.heading}" if chunk.heading else ""
        lines.append(
            f"  - lines {chunk.start_line}-{chunk.end_line}:{heading} {chunk.summary}"
        )
    if len(chunks) > max_chunks:
        lines.append(f"  ... +{len(chunks) - max_chunks} more chunks")
    return "\n".join(lines)


def format_file_manifest_summary(index: CodebaseIndex, max_entries: int = 30) -> str:
    """Format a compact manifest for injection into LLM context."""
    lines = [
        (
            f"Workspace: {index.root} "
            f"({len(index.files)} files, {len(index.symbols)} symbols, {len(index.chunks)} chunks)"
        )
    ]
    if index.repository_summary:
        lines.append(f"Repository summary: {index.repository_summary}")

    dirs: dict[str, list[FileEntry]] = {}
    for file_entry in index.files:
        directory = str(Path(file_entry.path).parent)
        dirs.setdefault(directory, []).append(file_entry)

    count = 0
    for directory in sorted(dirs):
        if count >= max_entries:
            lines.append(f"  ... and {len(index.files) - count} more files")
            break
        entries = dirs[directory]
        for file_entry in entries[:5]:
            purpose_tag = f" [{file_entry.purpose}]" if file_entry.purpose else ""
            lines.append(
                f"  {file_entry.path} ({file_entry.language}, {file_entry.size}B, chunks={file_entry.chunk_count}){purpose_tag}"
            )
            count += 1
        if len(entries) > 5:
            lines.append(f"  ... +{len(entries) - 5} more in {directory}/")
            count += len(entries) - 5

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _max_chunk_lines(lang: str) -> int:
    return _MAX_CHUNK_LINES_BY_LANG.get(lang, 110)


def _normalize_heading(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())[:120]


def _first_meaningful_line(content: str) -> str:
    for line in content.splitlines()[:12]:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(("//", "/*", "*")):
            continue
        return stripped
    return ""


def _query_keywords(query: str) -> list[str]:
    return [token.lower() for token in re.findall(r"[A-Za-z0-9_./:-]+", query) if len(token) > 1]


def _unique(items: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        normalized = item.strip()
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(normalized)
    return result
