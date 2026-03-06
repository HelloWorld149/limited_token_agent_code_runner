"""Tests for chunk-aware indexing and large-file retrieval."""

from __future__ import annotations

from hashlib import sha1
from pathlib import Path

from agent.config import AgentConfig
from agent.indexer import (
    build_codebase_index,
    expand_chunk_window,
    format_file_outline,
    get_file_chunks,
    search_chunks,
)
from agent.nodes import retrieve_context
from agent.state import BuildState


MODEL = "gpt-4o-mini"


def _cache_dir_for(workspace_path: Path) -> Path:
    return workspace_path.parent / f"{workspace_path.name}-cache"


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


class TestChunkAwareIndexing:
    def test_large_text_file_is_chunked_instead_of_skipped(self, tmp_path: Path) -> None:
        sections: list[str] = []
        for idx in range(60):
            body = "\n".join(
                f"line {idx}-{inner} semantic content repeated to force chunked indexing behavior"
                for inner in range(220)
            )
            sections.append(f"## Section {idx}\n{body}\n")
        large_markdown = "# Large Doc\n\n" + "\n".join(sections)
        target = tmp_path / "docs" / "large.md"
        _write(target, large_markdown)

        index = build_codebase_index(tmp_path, use_persistent_cache=False)

        file_entry = next(file for file in index.files if file.path == "docs/large.md")
        assert file_entry.size > 500_000
        assert file_entry.chunk_count > 5
        assert index.repository_summary
        assert get_file_chunks(index, file_entry.path)

    def test_streamed_oversized_text_file_is_indexed(self, tmp_path: Path) -> None:
        repeated_line = "oversized semantic payload for streamed indexing coverage\n"
        large_payload = "# Streamed Doc\n\n" + (repeated_line * 45_000)
        target = tmp_path / "docs" / "oversized.md"
        _write(target, large_payload)

        index = build_codebase_index(tmp_path, use_persistent_cache=False)

        file_entry = next(file for file in index.files if file.path == "docs/oversized.md")
        assert file_entry.size > 2_000_000
        assert file_entry.chunk_count > 10
        assert get_file_chunks(index, file_entry.path)

    def test_persistent_cache_reuses_unchanged_files(self, tmp_path: Path, monkeypatch) -> None:
        _write(tmp_path / "docs" / "cached.md", "# Cached\n\nalpha\nbeta\n")
        cache_dir = _cache_dir_for(tmp_path)

        import agent.indexer as indexer_mod

        original = indexer_mod._index_single_file
        calls: list[str] = []

        def wrapped(*args, **kwargs):
            rel_path = kwargs.get("rel_path")
            if isinstance(rel_path, str):
                calls.append(rel_path)
            return original(*args, **kwargs)

        monkeypatch.setattr(indexer_mod, "_index_single_file", wrapped)

        first_index = build_codebase_index(tmp_path, use_persistent_cache=True, cache_directory=cache_dir)
        assert any(file.path == "docs/cached.md" for file in first_index.files)
        assert calls == ["docs/cached.md"]

        calls.clear()
        second_index = build_codebase_index(tmp_path, use_persistent_cache=True, cache_directory=cache_dir)
        assert any(file.path == "docs/cached.md" for file in second_index.files)
        assert calls == []
        assert (cache_dir / ".codebase_index_cache_v2.json.gz").exists()
        assert not (tmp_path / ".codebase_index_cache_v2.json.gz").exists()

    def test_persistent_cache_invalidates_changed_files(self, tmp_path: Path, monkeypatch) -> None:
        target = tmp_path / "docs" / "invalidate.md"
        cache_dir = _cache_dir_for(tmp_path)
        _write(target, "# Version 1\n\ninitial\n")
        build_codebase_index(tmp_path, use_persistent_cache=True, cache_directory=cache_dir)

        import agent.indexer as indexer_mod

        original = indexer_mod._index_single_file
        calls: list[str] = []

        def wrapped(*args, **kwargs):
            rel_path = kwargs.get("rel_path")
            if isinstance(rel_path, str):
                calls.append(rel_path)
            return original(*args, **kwargs)

        monkeypatch.setattr(indexer_mod, "_index_single_file", wrapped)

        updated_payload = "# Version 2\n\n" + sha1(b"changed").hexdigest() + "\n"
        _write(target, updated_payload)
        updated_index = build_codebase_index(tmp_path, use_persistent_cache=True, cache_directory=cache_dir)

        assert any(file.path == "docs/invalidate.md" for file in updated_index.files)
        assert calls == ["docs/invalidate.md"]

    def test_failed_cache_write_preserves_existing_cache(self, tmp_path: Path, monkeypatch) -> None:
        cache_dir = _cache_dir_for(tmp_path)
        target = tmp_path / "docs" / "cached.md"
        _write(target, "# Cached\n\nalpha\n")

        build_codebase_index(tmp_path, use_persistent_cache=True, cache_directory=cache_dir)
        cache_file = cache_dir / ".codebase_index_cache_v2.json.gz"
        original_payload = cache_file.read_bytes()

        import agent.indexer as indexer_mod

        def _raise_on_dump(*args, **kwargs):
            raise OSError("simulated cache write failure")

        monkeypatch.setattr(indexer_mod.json, "dump", _raise_on_dump)
        _write(target, "# Cached\n\nbeta\n")

        build_codebase_index(tmp_path, use_persistent_cache=True, cache_directory=cache_dir)

        assert cache_file.read_bytes() == original_payload
        assert not cache_file.with_name(f"{cache_file.name}.tmp").exists()

    def test_search_chunks_and_neighbor_expansion_preserve_section_context(self, tmp_path: Path) -> None:
        content = "\n".join(
            [
                "# Overview",
                *[f"overview line {idx}" for idx in range(40)],
                "## Alpha",
                *[f"alpha line {idx}" for idx in range(50)],
                "## Beta",
                *[f"beta unique needle line {idx}" for idx in range(50)],
                "## Gamma",
                *[f"gamma line {idx}" for idx in range(50)],
            ]
        )
        _write(tmp_path / "docs" / "guide.md", content)

        index = build_codebase_index(tmp_path, use_persistent_cache=False)
        hits = search_chunks(index, "needle beta", max_results=2)

        assert hits
        assert hits[0].file_path == "docs/guide.md"
        assert "Beta" in hits[0].summary or "Beta" in hits[0].heading

        expanded = expand_chunk_window(index, hits, neighbor_depth=1, max_chunks=4)
        headings = {chunk.heading for chunk in expanded if chunk.heading}
        assert any("Beta" in heading for heading in headings)
        assert any("Alpha" in heading or "Gamma" in heading for heading in headings)

    def test_file_outline_uses_chunk_summaries_and_line_ranges(self, tmp_path: Path) -> None:
        content = "\n".join(
            [
                "# Intro",
                *[f"intro line {idx}" for idx in range(30)],
                "## Details",
                *[f"detail line {idx}" for idx in range(50)],
            ]
        )
        _write(tmp_path / "README.md", content)

        index = build_codebase_index(tmp_path, use_persistent_cache=False)
        file_entry = next(file for file in index.files if file.path == "README.md")
        outline = format_file_outline(index, file_entry)

        assert "Summary:" in outline
        assert "lines" in outline
        assert "README.md" in outline


class TestChunkAwareRetrieval:
    def test_retrieve_context_uses_matching_late_chunk_not_file_prefix(self, tmp_path: Path) -> None:
        content = "\n".join(
            [
                "# Overview",
                *[f"intro line {idx}" for idx in range(120)],
                "## Parser Details",
                *[f"parser internals line {idx}" for idx in range(40)],
                "## Serialization Window",
                *[f"late unique token line {idx}" for idx in range(60)],
            ]
        )
        _write(tmp_path / "docs" / "architecture.md", content)
        index = build_codebase_index(tmp_path, use_persistent_cache=False)

        config = AgentConfig(
            model_name=MODEL,
            classifier_model=MODEL,
            subagent_model=MODEL,
            workspace_path=tmp_path,
            use_retrieval_subagent=False,
            use_tool_summarizer=False,
            use_conversation_compressor=False,
            use_multi_hop=False,
        )
        state = {
            "messages": [],
            "summary_of_knowledge": "",
            "codebase_index": index,
            "current_intent": "QUESTION",
            "build_state": BuildState(),
            "turn_count": 0,
            "last_user_input": "Where is the late unique token documented?",
            "_retrieved_context": "",
            "_tool_iteration_count": 0,
            "_turn_subagent_count": 0,
            "_turn_debug_logs": [],
        }

        result = retrieve_context(state, config)
        context = result["_retrieved_context"]

        assert "late unique token" in context
        assert "architecture.md" in context
        assert "lines" in context
        assert "intro line 0" not in context
