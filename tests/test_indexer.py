"""Tests for chunk-aware indexing and large-file retrieval."""

from __future__ import annotations

import gzip
import json
import logging
from hashlib import sha1
from pathlib import Path
import time

import pytest

from agent.config import AgentConfig
from agent.indexer import (
    build_codebase_index,
    expand_chunk_window,
    format_file_outline,
    get_file_chunks,
    search_chunks,
    stop_background_reindexing,
)
from agent.nodes import index_workspace, retrieve_context
from agent.state import BuildState
from agent.token_utils import estimate_text_tokens


MODEL = "gpt-4o-mini"


def _cache_dir_for(workspace_path: Path) -> Path:
    return workspace_path.parent / f"{workspace_path.name}-cache"


def _cache_file_for(workspace_path: Path, cache_dir: Path) -> Path:
    workspace_key = sha1(str(workspace_path.resolve()).encode("utf-8")).hexdigest()[:12]
    return cache_dir / f"{workspace_path.name}-{workspace_key}" / ".codebase_index_cache_v3.json.gz"


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
        assert _cache_file_for(tmp_path, cache_dir).exists()
        assert not (tmp_path / ".codebase_index_cache_v3.json.gz").exists()

    def test_build_user_directory_is_skipped(self, tmp_path: Path) -> None:
        _write(tmp_path / "src" / "main.cpp", "int main() { return 0; }\n")
        _write(tmp_path / "build-user" / "generated.cpp", "int generated() { return 1; }\n")

        index = build_codebase_index(tmp_path, use_persistent_cache=False)

        assert any(file.path == "src/main.cpp" for file in index.files)
        assert not any(file.path.startswith("build-user/") for file in index.files)

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
        cache_file = _cache_file_for(tmp_path, cache_dir)
        original_payload = cache_file.read_bytes()

        import agent.indexer as indexer_mod

        def _raise_on_dump(*args, **kwargs):
            raise OSError("simulated cache write failure")

        monkeypatch.setattr(indexer_mod.json, "dump", _raise_on_dump)
        _write(target, "# Cached\n\nbeta\n")

        build_codebase_index(tmp_path, use_persistent_cache=True, cache_directory=cache_dir)

        assert cache_file.read_bytes() == original_payload
        assert not cache_file.with_name(f"{cache_file.name}.tmp").exists()

    def test_cache_persists_chunk_embeddings(self, tmp_path: Path, monkeypatch) -> None:
        _write(tmp_path / "docs" / "embedded.md", "# Embedded\n\ncar engine diagnostics\n")
        cache_dir = _cache_dir_for(tmp_path)

        import agent.indexer as indexer_mod

        class FakeEmbeddingBackend:
            backend_name = "fake"
            model_name = "fake-semantic"
            dimensions = 3
            signature = "fake:semantic:3"

            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                return [self.embed_query(text) for text in texts]

            def embed_query(self, text: str) -> list[float]:
                lowered = text.lower()
                if "car" in lowered or "engine" in lowered:
                    return [1.0, 0.0, 0.0]
                return [0.0, 1.0, 0.0]

        monkeypatch.setattr(
            indexer_mod,
            "_resolve_embedding_backend",
            lambda **_: FakeEmbeddingBackend(),
        )

        build_codebase_index(
            tmp_path,
            use_persistent_cache=True,
            cache_directory=cache_dir,
            use_embedding_retrieval=True,
        )

        cache_file = _cache_file_for(tmp_path, cache_dir)
        with gzip.open(cache_file, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)

        record = payload["records"]["docs/embedded.md"]
        assert record["embedding_signature"] == "fake:semantic:3"
        assert record["chunks"][0]["embedding"] == [1.0, 0.0, 0.0]

    def test_openai_provider_fails_fast_when_backend_unavailable(self, tmp_path: Path, monkeypatch) -> None:
        _write(tmp_path / "docs" / "embedded.md", "# Embedded\n\ncar engine diagnostics\n")

        import agent.indexer as indexer_mod

        monkeypatch.setattr(
            indexer_mod,
            "_try_create_openai_embedding_backend",
            lambda *args, **kwargs: None,
        )

        with pytest.raises(RuntimeError, match="OpenAI embedding provider is required"):
            build_codebase_index(
                tmp_path,
                use_persistent_cache=False,
                use_embedding_retrieval=True,
                embedding_provider="openai",
            )

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
    def test_search_chunks_uses_embedding_similarity_when_enabled(self, tmp_path: Path, monkeypatch) -> None:
        _write(tmp_path / "docs" / "vehicles.md", "# Vehicles\n\nThe car engine and drivetrain need regular maintenance.\n")
        _write(tmp_path / "docs" / "fruit.md", "# Fruit\n\nBanana smoothie preparation notes.\n")

        import agent.indexer as indexer_mod

        class FakeEmbeddingBackend:
            backend_name = "fake"
            model_name = "fake-semantic"
            dimensions = 3
            signature = "fake:semantic:3"

            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                return [self.embed_query(text) for text in texts]

            def embed_query(self, text: str) -> list[float]:
                lowered = text.lower()
                if any(term in lowered for term in ("car", "engine", "drivetrain", "automobile", "servicing")):
                    return [1.0, 0.0, 0.0]
                if "banana" in lowered or "fruit" in lowered:
                    return [0.0, 1.0, 0.0]
                return [0.0, 0.0, 1.0]

        monkeypatch.setattr(
            indexer_mod,
            "_resolve_embedding_backend",
            lambda **_: FakeEmbeddingBackend(),
        )

        index = build_codebase_index(
            tmp_path,
            use_persistent_cache=False,
            use_embedding_retrieval=True,
        )

        lexical_only = search_chunks(
            index,
            "automobile servicing",
            max_results=2,
            use_embedding_retrieval=False,
        )
        semantic_hits = search_chunks(
            index,
            "automobile servicing",
            max_results=2,
            use_embedding_retrieval=True,
        )

        assert lexical_only == []
        assert semantic_hits
        assert semantic_hits[0].file_path == "docs/vehicles.md"
        assert semantic_hits[0].embedding == [1.0, 0.0, 0.0]

    def test_embedding_requests_are_trimmed_to_5000_tokens(self, tmp_path: Path, monkeypatch) -> None:
        oversized_text = "alpha " * 8000
        _write(tmp_path / "docs" / "oversized.md", f"# Oversized\n\n{oversized_text}\n")

        import agent.indexer as indexer_mod

        captured_texts: list[str] = []

        class FakeEmbeddingBackend:
            backend_name = "fake"
            model_name = "text-embedding-3-large"
            dimensions = 3
            signature = "fake:semantic:3"

            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                captured_texts.extend(texts)
                return [[1.0, 0.0, 0.0] for _ in texts]

            def embed_query(self, text: str) -> list[float]:
                captured_texts.append(text)
                return [1.0, 0.0, 0.0]

        monkeypatch.setattr(
            indexer_mod,
            "_resolve_embedding_backend",
            lambda **_: FakeEmbeddingBackend(),
        )

        index = build_codebase_index(
            tmp_path,
            use_persistent_cache=False,
            use_embedding_retrieval=True,
        )
        search_chunks(
            index,
            "alpha concept",
            max_results=1,
            use_embedding_retrieval=True,
        )

        assert captured_texts
        assert all(
            estimate_text_tokens(text, "text-embedding-3-large") <= 5000
            for text in captured_texts
        )

    def test_embedding_population_failure_logs_warning(self, tmp_path: Path, monkeypatch, caplog: pytest.LogCaptureFixture) -> None:
        _write(tmp_path / "docs" / "embedded.md", "# Embedded\n\nsemantic payload\n")

        import agent.indexer as indexer_mod

        class FailingEmbeddingBackend:
            backend_name = "fake"
            model_name = "fake-semantic"
            dimensions = 3
            signature = "fake:failing-populate:3"

            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                raise RuntimeError("embedding service unavailable")

            def embed_query(self, text: str) -> list[float]:
                return [1.0, 0.0, 0.0]

        monkeypatch.setattr(
            indexer_mod,
            "_resolve_embedding_backend",
            lambda **_: FailingEmbeddingBackend(),
        )
        caplog.set_level(logging.WARNING)

        index = build_codebase_index(
            tmp_path,
            use_persistent_cache=False,
            use_embedding_retrieval=True,
        )

        assert any("Failed to populate chunk embeddings" in record.message for record in caplog.records)
        assert any(chunk.embedding == [] for chunk in index.chunks)

    def test_query_embedding_failure_logs_warning(self, tmp_path: Path, monkeypatch, caplog: pytest.LogCaptureFixture) -> None:
        _write(tmp_path / "docs" / "vehicles.md", "# Vehicles\n\nCar engine maintenance notes.\n")

        import agent.indexer as indexer_mod

        class FailingQueryEmbeddingBackend:
            backend_name = "fake"
            model_name = "fake-semantic"
            dimensions = 3
            signature = "fake:failing-query:3"

            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                return [[1.0, 0.0, 0.0] for _ in texts]

            def embed_query(self, text: str) -> list[float]:
                raise RuntimeError("query embedding outage")

        monkeypatch.setattr(
            indexer_mod,
            "_resolve_embedding_backend",
            lambda **_: FailingQueryEmbeddingBackend(),
        )
        caplog.set_level(logging.WARNING)

        index = build_codebase_index(
            tmp_path,
            use_persistent_cache=False,
            use_embedding_retrieval=True,
        )
        hits = search_chunks(
            index,
            "automobile servicing",
            max_results=2,
            use_embedding_retrieval=True,
        )

        assert hits == []
        assert any("Failed to compute query embedding" in record.message for record in caplog.records)

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

    def test_background_reindex_refreshes_retrieve_context(self, tmp_path: Path) -> None:
        live_file = tmp_path / "docs" / "live.md"
        _write(live_file, "# Live\n\ninitial release notes\n")

        config = AgentConfig(
            model_name=MODEL,
            classifier_model=MODEL,
            subagent_model=MODEL,
            workspace_path=tmp_path,
            use_embedding_retrieval=True,
            use_retrieval_subagent=False,
            use_tool_summarizer=False,
            use_conversation_compressor=False,
            use_multi_hop=False,
            background_reindex_enabled=True,
            background_reindex_interval_seconds=0.1,
        )
        state = {
            "messages": [],
            "summary_of_knowledge": "",
            "codebase_index": build_codebase_index(tmp_path, use_persistent_cache=False),
            "current_intent": "QUESTION",
            "build_state": BuildState(),
            "turn_count": 0,
            "last_user_input": "",
            "_retrieved_context": "",
            "_tool_iteration_count": 0,
            "_turn_subagent_count": 0,
            "_turn_debug_logs": [],
        }

        try:
            started = index_workspace(state, config)
            started_index = started["codebase_index"]
            _write(live_file, "# Live\n\nbeta rollout guide\n")

            poll_state = {
                **state,
                **started,
                "last_user_input": "Where is the beta rollout guide described?",
                "_turn_debug_logs": [],
            }

            deadline = time.time() + 5
            while time.time() < deadline:
                result = retrieve_context(poll_state, config)
                if "beta rollout guide" in result.get("_retrieved_context", ""):
                    refreshed_index = result.get("codebase_index", started_index)
                    assert refreshed_index.indexed_at_ns >= started_index.indexed_at_ns
                    assert any(
                        "background_reindex.refresh" in entry
                        for entry in result.get("_turn_debug_logs", [])
                    )
                    break
                time.sleep(0.15)
            else:
                raise AssertionError("background reindex did not refresh the live index in time")
        finally:
            stop_background_reindexing(tmp_path)

    def test_background_reindex_failure_reports_health_signal(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        live_file = tmp_path / "docs" / "live.md"
        _write(live_file, "# Live\n\ninitial release notes\n")

        config = AgentConfig(
            model_name=MODEL,
            classifier_model=MODEL,
            subagent_model=MODEL,
            workspace_path=tmp_path,
            use_embedding_retrieval=False,
            use_retrieval_subagent=False,
            use_tool_summarizer=False,
            use_conversation_compressor=False,
            use_multi_hop=False,
            background_reindex_enabled=True,
            background_reindex_interval_seconds=0.1,
        )
        state = {
            "messages": [],
            "summary_of_knowledge": "",
            "codebase_index": build_codebase_index(tmp_path, use_persistent_cache=False),
            "current_intent": "QUESTION",
            "build_state": BuildState(),
            "turn_count": 0,
            "last_user_input": "",
            "_retrieved_context": "",
            "_tool_iteration_count": 0,
            "_turn_subagent_count": 0,
            "_turn_debug_logs": [],
        }

        try:
            started = index_workspace(state, config)
            import agent.indexer as indexer_mod

            caplog.set_level(logging.WARNING)

            def _fail_rebuild(*args, **kwargs):
                raise RuntimeError("background refresh unavailable")

            monkeypatch.setattr(indexer_mod, "build_codebase_index", _fail_rebuild)
            _write(live_file, "# Live\n\nupdated release notes\n")

            poll_state = {
                **state,
                **started,
                "last_user_input": "Where are the updated release notes described?",
                "_turn_debug_logs": [],
            }

            deadline = time.time() + 5
            while time.time() < deadline:
                result = retrieve_context(poll_state, config)
                if any(
                    "background_reindex.health" in entry and "background refresh unavailable" in entry
                    for entry in result.get("_turn_debug_logs", [])
                ):
                    assert any("Background reindex failed" in record.message for record in caplog.records)
                    break
                time.sleep(0.15)
            else:
                raise AssertionError("background reindex failure health signal was not surfaced in time")
        finally:
            stop_background_reindexing(tmp_path)
