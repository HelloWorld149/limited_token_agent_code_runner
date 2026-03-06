from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage

from agent.config import AgentConfig
from agent.indexer import (
    build_codebase_index,
    configure_background_reindexing,
    detect_directory_references,
    detect_file_references,
    expand_chunk_window,
    format_file_outline,
    format_file_manifest_summary,
    get_live_codebase_index,
    get_file_chunks,
    search_chunks,
    search_index,
    stop_background_reindexing,
)
from agent.intent import classify_intent_sync
from agent.model_utils import build_chat_model, normalize_ai_message
from agent.prompts import INTENT_PROMPT_MAP
from agent.state import AgentState, BuildState, ChunkEntry, CodebaseIndex, FileEntry, SymbolEntry
from agent.subagents import (
    conversation_compressor_sync,
    is_complex_question,
    multi_hop_decomposer_sync,
    retrieval_subagent_sync,
    should_summarize_tool_output,
    tool_output_summarizer_sync,
)
from agent.token_utils import (
    estimate_text_tokens,
    fit_messages_to_budget,
    sanitize_tool_message_sequence,
    trim_text_to_token_budget,
)
from agent.tools import ALL_TOOLS, set_tool_runtime_policy, set_workspace_root


# ===================================================================
# Node: index_workspace — runs once at startup
# ===================================================================

def index_workspace(state: AgentState, config: AgentConfig) -> dict[str, Any]:
    """Verify workspace exists and build the in-memory codebase index."""
    ws = config.workspace_path.resolve()
    if not ws.exists() or not ws.is_dir():
        return {
            "summary_of_knowledge": (
                f"ERROR: workspace path '{ws}' does not exist. "
                "The agent requires a pre-downloaded copy of nlohmann/json."
            ),
            "codebase_index": CodebaseIndex(),
            "build_state": BuildState(),
            "turn_count": 0,
        }

    # Set the workspace root explicitly for tools (avoids Path.cwd() dependency)
    set_workspace_root(ws)
    set_tool_runtime_policy(
        timeout_seconds=config.shell_timeout_seconds,
        allow_dangerous_shell_commands=config.allow_dangerous_shell_commands,
    )

    # Build index (use resolved absolute path)
    index = build_codebase_index(
        ws,
        use_persistent_cache=config.index_cache_enabled,
        cache_directory=config.cache_directory,
        use_embedding_retrieval=config.use_embedding_retrieval,
        embedding_provider=config.embedding_provider,
        embedding_model=config.embedding_model,
        embedding_dimensions=config.embedding_dimensions,
    )
    configure_background_reindexing(
        workspace_path=ws,
        initial_index=index,
        enabled=config.background_reindex_enabled,
        interval_seconds=config.background_reindex_interval_seconds,
        use_persistent_cache=config.index_cache_enabled,
        cache_directory=config.cache_directory,
        use_embedding_retrieval=config.use_embedding_retrieval,
        embedding_provider=config.embedding_provider,
        embedding_model=config.embedding_model,
        embedding_dimensions=config.embedding_dimensions,
    )

    # Detect environment
    env_facts = _probe_environment(ws)
    env_facts.append(f"retrieval_embeddings={index.embedding_backend}")
    env_facts.append(
        "background_reindex="
        + (
            f"enabled:{config.background_reindex_interval_seconds:.0f}s"
            if config.background_reindex_enabled
            else "disabled"
        )
    )

    summary = (
        f"Workspace: {ws.resolve()} | "
        f"Files: {len(index.files)} | Symbols: {len(index.symbols)} | Chunks: {len(index.chunks)} | "
        f"Repo: {index.repository_summary} | "
        + " | ".join(env_facts)
    )

    return {
        "summary_of_knowledge": summary,
        "codebase_index": index,
        "build_state": BuildState(),
        "turn_count": 0,
        "current_intent": "QUESTION",
        "last_user_input": "",
    }


# ===================================================================
# Node: classify_and_prepare — intent classification + context prep
# ===================================================================

def classify_and_prepare(state: AgentState, config: AgentConfig) -> dict[str, Any]:
    """Classify user intent (via lightweight LLM) and prepare retrieval context.

    Uses a single context-aware classifier call.  The classifier receives
    the previous intent and a short summary of the last assistant message
    so it can resolve ambiguous follow-ups ("yes please", "do it") in one
    pass without a separate follow-up classifier.

    Also runs the conversation compressor subagent every 3 turns to keep
    the rolling summary fresh and compact.
    """
    user_input = state.get("last_user_input", "")
    previous_intent = state.get("current_intent", "QUESTION")
    messages = state.get("messages", [])

    # Defensive: if previous intent was EXIT the session should have ended.
    # If we're still here the classification was wrong — don't let it cascade.
    if previous_intent == "EXIT":
        previous_intent = "QUESTION"

    # Build a short summary of the last assistant message so the classifier
    # can understand what was offered / discussed.
    last_ai_summary = _last_ai_text(messages)[:300]

    # Single context-aware LLM call (~200 tokens, separate from main budget)
    intent = classify_intent_sync(
        user_input,
        config.classifier_model,
        previous_intent=previous_intent,
        last_ai_summary=last_ai_summary,
    )

    turn_count = state.get("turn_count", 0) + 1
    result: dict[str, Any] = {
        "current_intent": intent,
        "turn_count": turn_count,
        "_tool_iteration_count": 0,  # reset tool loop counter each turn
        "_turn_subagent_count": 0,
        "_turn_debug_logs": [f"intent={intent}"],
    }

    # --- Conversation Compressor Subagent ---
    # Every 3 turns, re-compress the summary using a subagent.
    # This produces a much tighter summary than naive string concatenation.
    if config.use_conversation_compressor and turn_count > 1 and turn_count % 3 == 0:
        old_summary = state.get("summary_of_knowledge", "")
        messages = state.get("messages", [])

        # Collect last few messages as text for the compressor
        recent_text_parts: list[str] = []
        for msg in messages[-6:]:  
            role = getattr(msg, "type", "unknown")
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            recent_text_parts.append(f"[{role}]: {content[:300]}")
        recent_text = "\n".join(recent_text_parts)

        if recent_text.strip():
            new_summary = conversation_compressor_sync(
                old_summary=old_summary,
                recent_messages_text=recent_text,
                model_name=config.subagent_model,
                max_input_tokens=3000,
                max_output_tokens=400,
            )
            result["summary_of_knowledge"] = new_summary
            result["_turn_subagent_count"] = 1
            result["_turn_debug_logs"] = list(result.get("_turn_debug_logs", [])) + [
                "subagent.conversation_compressor used=1",
                f"conversation.recent_messages={len(messages[-6:])}",
            ]

    return result


# ===================================================================
# Node: retrieve_context — search index and inject relevant snippets
# ===================================================================

def retrieve_context(state: AgentState, config: AgentConfig) -> dict[str, Any]:
    """Chunk-aware retrieval: file outlines -> semantic chunk search -> tool fallback.

    Large files are retrieved through cached semantic chunks with line provenance,
    not whole-file prefixes. File outlines provide map-level context, chunk search
    provides evidence-bearing code, and adjacent chunk expansion captures answers
    that span section boundaries.
    """
    user_input = state.get("last_user_input", "")
    original_index = state.get("codebase_index", CodebaseIndex())
    index = get_live_codebase_index(config.workspace_path, original_index)

    raw_code_chunks: list[str] = []
    debug_logs = list(state.get("_turn_debug_logs", []))
    turn_subagent_count = int(state.get("_turn_subagent_count", 0))
    file_lookup = {file_entry.path: file_entry for file_entry in index.files}

    index_updated = index.indexed_at_ns > 0 and index.indexed_at_ns != original_index.indexed_at_ns
    if index_updated:
        debug_logs.append(
            "background_reindex.refresh "
            f"chunks={len(index.chunks)} indexed_at_ns={index.indexed_at_ns}"
        )

    if config.use_retrieval_subagent:
        token_budget_for_raw = 3000
    else:
        token_budget_for_raw = 1800

    tokens_used = 0
    seen_chunk_keys: set[tuple[str, int, int]] = set()
    outline_loaded = 0
    direct_chunk_loaded = 0
    search_loaded = 0

    referenced_files = detect_file_references(user_input, index)
    dir_files = detect_directory_references(user_input, index)
    direct_files = referenced_files + [
        file_entry for file_entry in dir_files
        if file_entry.path not in {ref.path for ref in referenced_files}
    ]
    direct_paths = {file_entry.path for file_entry in direct_files}

    if len(direct_files) > 8:
        debug_logs.append(f"retrieve.direct_files_truncated={len(direct_files)}")

    for file_entry in direct_files[:8]:
        outline = format_file_outline(index, file_entry, max_chunks=6 if len(direct_files) <= 3 else 3)
        new_total = _append_raw_context(
            raw_code_chunks=raw_code_chunks,
            text=f"--- File Outline: {file_entry.path} ---\n{outline}",
            model_name=config.model_name,
            token_budget=token_budget_for_raw,
            tokens_used=tokens_used,
        )
        if new_total != tokens_used:
            tokens_used = new_total
            outline_loaded += 1

    direct_seed_chunks: list[ChunkEntry] = []
    for file_entry in direct_files[:4]:
        local_hits = search_chunks(
            index,
            user_input,
            max_results=2,
            allowed_files={file_entry.path},
            use_embedding_retrieval=config.use_embedding_retrieval,
        )
        if not local_hits:
            local_hits = get_file_chunks(index, file_entry.path)[:2]
        direct_seed_chunks.extend(local_hits[:2])

    for chunk in expand_chunk_window(index, _dedupe_chunks(direct_seed_chunks), neighbor_depth=1, max_chunks=10):
        chunk_key = (chunk.file_path, chunk.start_line, chunk.end_line)
        if chunk_key in seen_chunk_keys:
            continue
        new_total = _append_raw_context(
            raw_code_chunks=raw_code_chunks,
            text=_format_chunk_context(chunk, file_lookup.get(chunk.file_path)),
            model_name=config.model_name,
            token_budget=token_budget_for_raw,
            tokens_used=tokens_used,
        )
        if new_total == tokens_used:
            continue
        seen_chunk_keys.add(chunk_key)
        tokens_used = new_total
        direct_chunk_loaded += 1

    if tokens_used < token_budget_for_raw - 200:
        keyword_chunks = search_chunks(
            index,
            user_input,
            max_results=8,
            use_embedding_retrieval=config.use_embedding_retrieval,
        )
        if not keyword_chunks:
            keyword_chunks = _chunks_from_search_results(index, search_index(index, user_input, max_results=8))

        expanded_keyword_chunks = expand_chunk_window(
            index,
            [chunk for chunk in keyword_chunks if chunk.file_path not in direct_paths],
            neighbor_depth=1,
            max_chunks=12,
        )

        for chunk in expanded_keyword_chunks:
            chunk_key = (chunk.file_path, chunk.start_line, chunk.end_line)
            if chunk_key in seen_chunk_keys:
                continue
            new_total = _append_raw_context(
                raw_code_chunks=raw_code_chunks,
                text=_format_chunk_context(chunk, file_lookup.get(chunk.file_path)),
                model_name=config.model_name,
                token_budget=token_budget_for_raw,
                tokens_used=tokens_used,
            )
            if new_total == tokens_used:
                continue
            seen_chunk_keys.add(chunk_key)
            tokens_used = new_total
            search_loaded += 1

    if not raw_code_chunks:
        manifest = format_file_manifest_summary(index, max_entries=20)
        manifest = trim_text_to_token_budget(manifest, config.model_name, 800)
        raw_code_chunks.append(f"--- File Manifest ---\n{manifest}")
        debug_logs.append("retrieve.fallback=file_manifest")

    debug_logs.append(
        "retrieve.selection "
        f"outlines={outline_loaded} direct_chunks={direct_chunk_loaded} keyword_chunks={search_loaded} "
        f"items={len(raw_code_chunks)} "
        f"tokens={tokens_used}/{token_budget_for_raw}"
    )

    # ---------------------------------------------------------------
    # Subagent compression: compress raw chunks into dense digest
    # Uses multi-hop decomposer for complex questions, or simple
    # retrieval subagent for straightforward ones.
    # ---------------------------------------------------------------
    if config.use_retrieval_subagent and raw_code_chunks:
        if config.use_multi_hop and is_complex_question(user_input):
            def _index_search_for_subquery(sub_query: str) -> list[str]:
                sub_results = search_chunks(
                    index,
                    sub_query,
                    max_results=5,
                    use_embedding_retrieval=config.use_embedding_retrieval,
                )
                if not sub_results:
                    sub_results = _chunks_from_search_results(index, search_index(index, sub_query, max_results=5))
                formatted = [
                    _format_chunk_context(chunk, file_lookup.get(chunk.file_path))
                    for chunk in expand_chunk_window(index, sub_results, neighbor_depth=1, max_chunks=8)
                ]
                return formatted if formatted else raw_code_chunks

            multi_hop_result = multi_hop_decomposer_sync(
                user_query=user_input,
                raw_code_chunks=raw_code_chunks,
                index_search_fn=_index_search_for_subquery,
                model_name=config.subagent_model,
                max_output_tokens=config.retrieval_digest_tokens,
                return_trace=True,
            )
            if isinstance(multi_hop_result, tuple):
                context_text, trace = multi_hop_result
            else:
                context_text, trace = multi_hop_result, {}
            mh_used = int(trace.get("total_subagents_used", 0))
            turn_subagent_count += mh_used
            sub_queries = trace.get("sub_queries", [])
            if isinstance(sub_queries, list) and sub_queries:
                plan = " | ".join(str(q) for q in sub_queries[:3])
                debug_logs.append(f"subagent.multi_hop.plan={plan[:280]}")
            debug_logs.append(
                "subagent.multi_hop "
                f"used={mh_used} parallel={trace.get('parallel_subagents', 0)} "
                f"merge={trace.get('merge_used', False)} fallback={trace.get('fallback_used', False)}"
            )
        else:
            # Simple compression: one retrieval subagent
            context_text = retrieval_subagent_sync(
                user_query=user_input,
                raw_code_chunks=raw_code_chunks,
                model_name=config.subagent_model,
                max_input_tokens=3500,
                max_output_tokens=config.retrieval_digest_tokens,
            )
            turn_subagent_count += 1
            debug_logs.append(
                f"subagent.retrieval_compressor used=1 chunks={len(raw_code_chunks)}"
            )
    else:
        # No subagent: use raw chunks directly (original behavior)
        context_text = "\n\n".join(raw_code_chunks)
        context_text = trim_text_to_token_budget(
            context_text, config.model_name, 1800
        )
        debug_logs.append("subagent.retrieval_compressor used=0")

    result = {
        "_retrieved_context": context_text,
        "_turn_subagent_count": turn_subagent_count,
        "_turn_debug_logs": debug_logs,
    }
    if index_updated:
        result["codebase_index"] = index
    return result


def _append_raw_context(
    raw_code_chunks: list[str],
    text: str,
    model_name: str,
    token_budget: int,
    tokens_used: int,
) -> int:
    """Append retrieval context while respecting the raw retrieval token budget."""
    remaining = token_budget - tokens_used
    if remaining <= 80:
        return tokens_used

    item_tokens = estimate_text_tokens(text, model_name)
    if item_tokens <= remaining:
        raw_code_chunks.append(text)
        return tokens_used + item_tokens

    trimmed = trim_text_to_token_budget(text, model_name, remaining)
    if not trimmed.strip():
        return tokens_used

    trimmed_tokens = estimate_text_tokens(trimmed, model_name)
    if trimmed_tokens <= 0:
        return tokens_used

    raw_code_chunks.append(trimmed)
    return tokens_used + trimmed_tokens


def _format_chunk_context(chunk: ChunkEntry, file_entry: FileEntry | None) -> str:
    purpose_tag = f" [{file_entry.purpose}]" if file_entry and file_entry.purpose else ""
    heading_tag = f" | heading={chunk.heading}" if chunk.heading else ""
    symbols = ", ".join(chunk.symbol_names[:5])
    symbols_tag = f" | symbols={symbols}" if symbols else ""
    return (
        f"--- {chunk.file_path} (lines {chunk.start_line}-{chunk.end_line})"
        f"{purpose_tag}{heading_tag}{symbols_tag} ---\n"
        f"{chunk.text}"
    )


def _chunks_from_search_results(
    index: CodebaseIndex,
    results: list[FileEntry | SymbolEntry],
) -> list[ChunkEntry]:
    chunks: list[ChunkEntry] = []
    for item in results:
        if isinstance(item, SymbolEntry):
            chunk = _find_chunk_for_line(index, item.file, item.line)
            if chunk is not None:
                chunks.append(chunk)
        elif isinstance(item, FileEntry):
            file_chunks = get_file_chunks(index, item.path)
            if file_chunks:
                chunks.append(file_chunks[0])
    return _dedupe_chunks(chunks)


def _find_chunk_for_line(
    index: CodebaseIndex,
    rel_path: str,
    line: int,
) -> ChunkEntry | None:
    for chunk in get_file_chunks(index, rel_path):
        if chunk.start_line <= line <= chunk.end_line:
            return chunk
    file_chunks = get_file_chunks(index, rel_path)
    return file_chunks[0] if file_chunks else None


def _dedupe_chunks(chunks: list[ChunkEntry]) -> list[ChunkEntry]:
    deduped: list[ChunkEntry] = []
    seen: set[tuple[str, int, int]] = set()
    for chunk in chunks:
        key = (chunk.file_path, chunk.start_line, chunk.end_line)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(chunk)
    return deduped


# ===================================================================
# Node: answer_question — handle QUESTION intent
# ===================================================================

def answer_question(state: AgentState, config: AgentConfig) -> dict[str, Any]:
    """Generate an answer using retrieved context + conversation history.

    Tool-augmented: the LLM can call read_file_chunk/list_directory/search_codebase
    on-demand when pre-retrieved context is insufficient. It will simply choose
    not to call tools when the context already answers the question.
    """
    return _invoke_llm_with_context(state, config, use_tools=True)


# ===================================================================
# Node: run_build — handle COMPILE intent
# ===================================================================

def run_build(state: AgentState, config: AgentConfig) -> dict[str, Any]:
    """Handle build requests — LLM decides which build commands to run via tools."""
    return _invoke_llm_with_context(state, config, use_tools=True)


# ===================================================================
# Node: run_tests — handle RUN intent
# ===================================================================

def run_tests(state: AgentState, config: AgentConfig) -> dict[str, Any]:
    """Handle test execution requests — LLM uses tools to run and interpret tests."""
    return _invoke_llm_with_context(state, config, use_tools=True)


# ===================================================================
# Node: explore_codebase — handle EXPLORE intent
# ===================================================================

def explore_codebase(state: AgentState, config: AgentConfig) -> dict[str, Any]:
    """Handle codebase exploration requests — LLM uses tools to browse/search."""
    return _invoke_llm_with_context(state, config, use_tools=True)


# ===================================================================
# Node: execute_tools_node — execute tool calls from LLM
# ===================================================================

def handle_tool_result(state: AgentState, config: AgentConfig) -> dict[str, Any]:
    """Post-process tool results: update build state and compress large outputs.

    When the tool output summarizer is enabled, large shell command outputs
    are compressed by a subagent into ~200 tokens. This prevents build logs
    and test outputs from consuming the main 5000-token budget.

    Returns compressed messages via add_messages reducer to replace originals.
    """
    messages = list(state.get("messages", []))
    # Increment tool iteration counter
    iteration_count = state.get("_tool_iteration_count", 0) + 1

    # Update build state based on tool output
    build_state = state.get("build_state", BuildState())
    build_state = _update_build_state(messages, build_state)

    # --- Tool Output Summarizer Subagent ---
    # Compress large ToolMessage contents so they don't bloat the main context
    compressed_messages: list[BaseMessage] = []
    summarizer_calls = 0
    debug_logs = list(state.get("_turn_debug_logs", []))
    turn_subagent_count = int(state.get("_turn_subagent_count", 0))
    if config.use_tool_summarizer:
        for msg in reversed(messages):
            if not isinstance(msg, ToolMessage):
                break
            content = str(msg.content)
            if should_summarize_tool_output(content):
                # Extract command from output for context
                cmd = _extract_cmd(content) or "unknown"
                compressed = tool_output_summarizer_sync(
                    tool_output=content,
                    command=cmd,
                    model_name=config.subagent_model,
                    max_input_tokens=3500,
                    max_output_tokens=config.tool_summary_tokens,
                )
                # Create a replacement ToolMessage with compressed content
                compressed_messages.append(
                    ToolMessage(
                        content=compressed,
                        tool_call_id=msg.tool_call_id,
                        id=msg.id,
                    )
                )
                summarizer_calls += 1

    if summarizer_calls:
        turn_subagent_count += summarizer_calls
        debug_logs.append(f"subagent.tool_summarizer calls={summarizer_calls}")

    result: dict[str, Any] = {
        "build_state": build_state,
        "_tool_iteration_count": iteration_count,
        "_turn_subagent_count": turn_subagent_count,
        "_turn_debug_logs": debug_logs,
    }
    if compressed_messages:
        # Return compressed messages — add_messages reducer will merge by ID
        result["messages"] = compressed_messages
    return result


# ===================================================================
# Node: continue_or_respond — after tools, decide: more tools or final text?
# ===================================================================

def continue_or_respond(state: AgentState, config: AgentConfig) -> dict[str, Any]:
    """After tool execution, call LLM to either make more tool calls or produce a text response.

    Enforces max_tool_iterations: once the limit is reached, the LLM is
    invoked without tool bindings so it MUST produce a text response.
    """
    iteration_count = state.get("_tool_iteration_count", 0)
    allow_tools = iteration_count < config.max_tool_iterations
    return _invoke_llm_with_context(state, config, use_tools=allow_tools)


# ===================================================================
# Router functions
# ===================================================================

def route_by_intent(state: AgentState) -> str:
    """Route to the appropriate handler based on classified intent."""
    intent = state.get("current_intent", "QUESTION")
    if intent == "EXIT":
        return "exit"
    if intent == "COMPILE":
        return "run_build"
    if intent == "RUN":
        return "run_tests"
    if intent == "EXPLORE":
        return "explore_codebase"
    return "answer_question"


# ===================================================================
# Node: handle_exit — inject a farewell message so display isn't stale
# ===================================================================

def handle_exit(state: AgentState, config: AgentConfig) -> dict[str, Any]:
    """Produce a short farewell AIMessage before the graph terminates.

    Without this, routing to END on EXIT would leave no new AIMessage in
    state, causing ``_display_response`` in main.py to replay the *previous*
    turn's answer — confusing the user.
    """
    stop_background_reindexing(config.workspace_path)
    return {"messages": [AIMessage(content="Goodbye!")]}


def route_after_llm(state: AgentState) -> str:
    """Check if the LLM wants to call tools or has produced a final text response."""
    messages = state.get("messages", [])
    last = _last_ai_message(messages)
    if last and getattr(last, "tool_calls", None):
        return "execute_tools"
    return "respond_to_user"


# ===================================================================
# Shared LLM invocation helper
# ===================================================================

def _invoke_llm_with_context(
    state: AgentState,
    config: AgentConfig,
    use_tools: bool,
) -> dict[str, Any]:
    """Build messages under token budget and invoke the main LLM."""
    intent = state.get("current_intent", "QUESTION")
    system_text = INTENT_PROMPT_MAP.get(intent, INTENT_PROMPT_MAP["QUESTION"])
    system = SystemMessage(content=system_text)

    # Build summary context message (SystemMessage is semantically correct —
    # this is injected context, not user input)
    summary = state.get("summary_of_knowledge", "")
    retrieved = state.get("_retrieved_context", "")
    summary_content = f"Knowledge: {summary}"
    if retrieved:
        summary_content += f"\n\nRetrieved code:\n{retrieved}"
    summary_msg = SystemMessage(content=summary_content)

    # Conversation history
    history = list(state.get("messages", []))

    # Build candidate message list
    candidate = [system, summary_msg] + history

    # Fit to input budget (reserve some for tool schemas if using tools)
    effective_budget = config.input_token_budget
    if use_tools:
        effective_budget -= 300  # reserve for tool schemas

    candidate = fit_messages_to_budget(candidate, config.model_name, effective_budget)
    candidate = sanitize_tool_message_sequence(candidate)

    # Use the effective output budget from config (caps at 800 for headroom)
    output_budget = config.effective_output_budget

    # Build model — use shared build_chat_model which auto-detects Responses API
    model = build_chat_model(config.model_name, temperature=0, max_tokens=output_budget)
    if use_tools:
        model = model.bind_tools(ALL_TOOLS)

    try:
        response = model.invoke(candidate)
        # Normalize content: Responses API may return list-of-blocks instead of str
        response = normalize_ai_message(response)
    except Exception as exc:
        response = AIMessage(
            content=f"LLM error: {type(exc).__name__}: {str(exc)[:200]}. "
            "Please rephrase or try again."
        )

    return {"messages": [response]}


# ===================================================================
# Environment probing
# ===================================================================

def _probe_environment(workspace_path: Path) -> list[str]:
    """Detect OS, build tools, and recommended cmake generator."""
    facts: list[str] = []
    facts.append(f"os={platform.system()}")

    tool_checks = [
        ("cmake --version", "cmake"),
        ("ninja --version", "ninja"),
    ]
    if os.name == "nt":
        tool_checks.extend([
            ("g++ --version", "gxx"),
            ("mingw32-make --version", "mingw32_make"),
        ])
    else:
        tool_checks.extend([
            ("g++ --version", "gxx"),
            ("make --version", "make"),
        ])

    for cmd, label in tool_checks:
        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                first_line = (result.stdout or "").strip().split("\n")[0][:80]
                facts.append(f"{label}={first_line}")
            else:
                facts.append(f"{label}=not_found")
        except Exception:
            facts.append(f"{label}=check_failed")

    # Recommended generator
    if os.name == "nt":
        if shutil.which("ninja"):
            facts.append("generator=Ninja")
        elif shutil.which("mingw32-make"):
            facts.append("generator=MinGW Makefiles")
        else:
            facts.append("generator=default")
    else:
        facts.append("generator=Ninja" if shutil.which("ninja") else "generator=Unix Makefiles")

    return facts


# ===================================================================
# Build state tracking
# ===================================================================

def _update_build_state(
    messages: list[BaseMessage], current: BuildState
) -> BuildState:
    """Update build state based on recent tool outputs.

    Returns a new frozen BuildState instance (never mutates the input).
    """
    for msg in reversed(messages):
        if not isinstance(msg, ToolMessage):
            continue
        text = str(msg.content)
        if "[cmd]=" not in text or "[exit_code]=" not in text:
            continue

        cmd = _extract_cmd(text) or ""
        exit_code = _extract_exit_code(text)
        cmd_lower = cmd.lower()

        # Start with inherited values
        status = current.status
        configured = current.configured
        built = current.built
        tested = current.tested
        last_error = current.last_error
        consecutive_errors = current.consecutive_errors

        if exit_code is not None and exit_code != 0:
            error_line = _first_error_line(text) or f"exit code {exit_code}"
            status = "FAILED"
            last_error = error_line[:200]
            consecutive_errors = current.consecutive_errors + 1
        else:
            consecutive_errors = 0
            last_error = ""

        # Detect lifecycle stage
        if re.search(r"\bcmake\b", cmd_lower) and "--build" not in cmd_lower:
            if exit_code == 0:
                configured = True
                status = "CONFIGURING"
        elif re.search(r"--build|\bmake\b|\bninja\b|\bmingw32-make\b", cmd_lower):
            if exit_code == 0:
                built = True
                status = "BUILDING"
        elif re.search(r"\bctest\b|\btest\b", cmd_lower):
            if exit_code == 0:
                tested = True
                status = "SUCCESS"
            elif exit_code is not None:
                status = "FAILED"

        return BuildState(
            status=status,
            configured=configured,
            built=built,
            tested=tested,
            last_exit_code=exit_code,
            last_error=last_error,
            consecutive_errors=consecutive_errors,
        )
    return current


# ===================================================================
# Text extraction helpers
# ===================================================================

def _extract_cmd(text: str) -> str | None:
    match = re.search(r"^\[cmd\]=(.*)$", text, flags=re.MULTILINE)
    return match.group(1).strip() if match else None


def _extract_exit_code(text: str) -> int | None:
    match = re.search(r"\[exit_code\]\s*=\s*(\d+)|\[exit_code\]=(\d+)", text)
    if not match:
        return None
    value = match.group(1) or match.group(2)
    return int(value)


def _first_error_line(text: str) -> str | None:
    for line in text.splitlines():
        if re.search(r"\berror\b|\bfatal\b|\bfailed\b", line, flags=re.IGNORECASE):
            return line.strip()
    return None


def _last_ai_message(messages: list[BaseMessage]) -> AIMessage | None:
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            return msg
    return None


def _last_ai_text(messages: list[BaseMessage]) -> str:
    last_ai = _last_ai_message(messages)
    if last_ai is None:
        return ""
    content = last_ai.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                parts.append(str(block.get("text", "")))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(p for p in parts if p)
    return str(content)


__all__ = [
    "index_workspace",
    "classify_and_prepare",
    "retrieve_context",
    "answer_question",
    "run_build",
    "run_tests",
    "explore_codebase",
    "handle_exit",
    "handle_tool_result",
    "continue_or_respond",
    "route_by_intent",
    "route_after_llm",
]
