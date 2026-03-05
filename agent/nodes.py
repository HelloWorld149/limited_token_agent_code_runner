from __future__ import annotations

import os
import platform
import re
import shutil
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

from agent.config import AgentConfig
from agent.indexer import (
    build_codebase_index,
    detect_directory_references,
    detect_file_references,
    format_file_manifest_summary,
    search_index,
)
from agent.intent import classify_intent_sync
from agent.prompts import INTENT_PROMPT_MAP
from agent.state import AgentState, BuildState, CodebaseIndex, FileEntry, SymbolEntry
from agent.token_utils import (
    estimate_text_tokens,
    fit_messages_to_budget,
    sanitize_tool_message_sequence,
    trim_text_to_token_budget,
)
from agent.subagents import (
    retrieval_subagent_sync,
    should_summarize_tool_output,
    tool_output_summarizer_sync,
)
from agent.tools import ALL_TOOLS


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

    # Change working directory to the workspace
    os.chdir(ws)

    # Build index (use resolved absolute path)
    index = build_codebase_index(ws)

    # Detect environment
    env_facts = _probe_environment(ws)

    summary = (
        f"Workspace: {ws.resolve()} | "
        f"Files: {len(index.files)} | Symbols: {len(index.symbols)} | "
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
    """Classify user intent (via lightweight LLM) and prepare retrieval context."""
    user_input = state.get("last_user_input", "")

    # Classify intent using separate lightweight LLM call (~110 tokens)
    intent = classify_intent_sync(user_input, config.classifier_model)

    return {
        "current_intent": intent,
        "turn_count": state.get("turn_count", 0) + 1,
    }


# ===================================================================
# Node: retrieve_context — search index and inject relevant snippets
# ===================================================================

def retrieve_context(state: AgentState, config: AgentConfig) -> dict[str, Any]:
    """Three-layer retrieval: path-aware direct load -> keyword search -> (tool fallback via ReAct).

    Layer 1: Detect explicit file paths/names in user input and load directly.
    Layer 2: Enhanced keyword + symbol search with fuzzy path matching.
    Layer 3 is handled by the ReAct tool loop in the LLM nodes (all intents have tools).

    When the retrieval subagent is enabled, raw code chunks are read with a LARGER
    budget (~3000 tokens) and then compressed by a subagent into a dense digest
    (~400 tokens). This frees up ~1400 tokens in the main context for reasoning.
    """
    user_input = state.get("last_user_input", "")
    index = state.get("codebase_index", CodebaseIndex())
    intent = state.get("current_intent", "QUESTION")

    raw_code_chunks: list[str] = []
    ws = config.workspace_path

    # When using retrieval subagent, read MORE raw code because the subagent
    # will compress it. Without subagent, use the original tight budget.
    if config.use_retrieval_subagent:
        token_budget_for_raw = 3000  # subagent compresses to ~400 tokens
    else:
        token_budget_for_raw = 1800  # goes directly into main context

    tokens_used = 0
    seen_files: set[str] = set()

    # ---------------------------------------------------------------
    # Layer 1: Path-aware direct retrieval
    # ---------------------------------------------------------------
    referenced_files = detect_file_references(user_input, index)
    dir_files = detect_directory_references(user_input, index)
    direct_files = referenced_files + [f for f in dir_files if f.path not in {r.path for r in referenced_files}]

    for fe in direct_files:
        if fe.path in seen_files:
            continue
        seen_files.add(fe.path)

        filepath = ws / fe.path
        if not filepath.exists() or not filepath.is_file():
            continue

        try:
            content = filepath.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        lines = content.splitlines()
        max_lines = min(80, len(lines))
        chunk = "\n".join(lines[:max_lines])

        chunk_tokens = estimate_text_tokens(chunk, config.model_name)
        if tokens_used + chunk_tokens > token_budget_for_raw:
            max_lines = min(30, len(lines))
            chunk = "\n".join(lines[:max_lines])
            chunk_tokens = estimate_text_tokens(chunk, config.model_name)
            if tokens_used + chunk_tokens > token_budget_for_raw:
                continue

        purpose_tag = f" [{fe.purpose}]" if fe.purpose else ""
        raw_code_chunks.append(
            f"--- {fe.path} (lines 1-{max_lines}){purpose_tag} ---\n{chunk}"
        )
        tokens_used += chunk_tokens

    # ---------------------------------------------------------------
    # Layer 2: Enhanced keyword + symbol search (fills remaining budget)
    # ---------------------------------------------------------------
    if tokens_used < token_budget_for_raw - 200:
        results = search_index(index, user_input, max_results=10)

        files_to_read: list[tuple[str, int]] = []
        for item in results:
            if isinstance(item, SymbolEntry):
                files_to_read.append((item.file, item.line))
            elif isinstance(item, FileEntry):
                files_to_read.append((item.path, 1))

        for rel_path, line in files_to_read:
            if rel_path in seen_files:
                continue
            seen_files.add(rel_path)

            filepath = ws / rel_path
            if not filepath.exists() or not filepath.is_file():
                continue

            try:
                content = filepath.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue

            lines_list = content.splitlines()
            start = max(0, line - 5)
            end = min(len(lines_list), line + 40)
            chunk = "\n".join(lines_list[start:end])

            chunk_tokens = estimate_text_tokens(chunk, config.model_name)
            if tokens_used + chunk_tokens > token_budget_for_raw:
                end = min(len(lines_list), line + 15)
                chunk = "\n".join(lines_list[start:end])
                chunk_tokens = estimate_text_tokens(chunk, config.model_name)
                if tokens_used + chunk_tokens > token_budget_for_raw:
                    continue

            raw_code_chunks.append(f"--- {rel_path} (lines {start+1}-{end}) ---\n{chunk}")
            tokens_used += chunk_tokens

    # ---------------------------------------------------------------
    # Fallback: if nothing retrieved, show file manifest
    # ---------------------------------------------------------------
    if not raw_code_chunks:
        manifest = format_file_manifest_summary(index, max_entries=20)
        manifest = trim_text_to_token_budget(manifest, config.model_name, 800)
        raw_code_chunks.append(f"--- File Manifest ---\n{manifest}")

    # ---------------------------------------------------------------
    # Subagent compression: compress raw chunks into dense digest
    # ---------------------------------------------------------------
    if config.use_retrieval_subagent and raw_code_chunks:
        context_text = retrieval_subagent_sync(
            user_query=user_input,
            raw_code_chunks=raw_code_chunks,
            model_name=config.subagent_model,
            max_input_tokens=3500,
            max_output_tokens=config.retrieval_digest_tokens,
        )
    else:
        # No subagent: use raw chunks directly (original behavior)
        context_text = "\n\n".join(raw_code_chunks)
        context_text = trim_text_to_token_budget(
            context_text, config.model_name, 1800
        )

    return {
        "_retrieved_context": context_text,
    }


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

    # Update build state based on tool output
    build_state = state.get("build_state", BuildState())
    build_state = _update_build_state(messages, build_state)

    # --- Tool Output Summarizer Subagent ---
    # Compress large ToolMessage contents so they don't bloat the main context
    compressed_messages: list[BaseMessage] = []
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

    result: dict[str, Any] = {"build_state": build_state}
    if compressed_messages:
        # Return compressed messages — add_messages reducer will merge by ID
        result["messages"] = compressed_messages
    return result


# ===================================================================
# Node: continue_or_respond — after tools, decide: more tools or final text?
# ===================================================================

def continue_or_respond(state: AgentState, config: AgentConfig) -> dict[str, Any]:
    """After tool execution, call LLM to either make more tool calls or produce a text response."""
    return _invoke_llm_with_context(state, config, use_tools=True)


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

    # Build summary context message
    summary = state.get("summary_of_knowledge", "")
    retrieved = state.get("_retrieved_context", "")
    summary_content = f"Knowledge: {summary}"
    if retrieved:
        summary_content += f"\n\nRetrieved code:\n{retrieved}"
    summary_msg = HumanMessage(content=summary_content)

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

    output_budget = min(config.output_token_budget, 800)

    # Build model — use Responses API for codex models, chat completions otherwise
    use_responses = _is_responses_model(config.model_name)
    model = ChatOpenAI(
        model=config.model_name,
        temperature=0,
        max_tokens=output_budget,
        **({"use_responses_api": True} if use_responses else {}),
    )
    if use_tools:
        model = model.bind_tools(ALL_TOOLS)

    try:
        response = model.invoke(candidate)
        # Normalize content: Responses API may return list-of-blocks instead of str
        response = _normalize_ai_message(response)
    except Exception as exc:
        response = AIMessage(
            content=f"LLM error: {type(exc).__name__}: {str(exc)[:200]}. "
            "Please rephrase or try again."
        )

    return {"messages": [response]}


# ===================================================================
# Model type detection
# ===================================================================

_RESPONSES_API_PATTERNS = ("codex", "o1", "o3", "o4")
_CHAT_ONLY_PREFIXES = ("gpt-4o", "gpt-4-", "gpt-3.5")


def _is_responses_model(model_name: str) -> bool:
    """Return True if the model should use the Responses API.

    Codex and reasoning models (o1/o3/o4) use the Responses API.
    Standard chat models (gpt-4o, gpt-3.5) use the chat completions API.
    """
    name_lower = model_name.lower()
    # Explicitly chat-only models first
    if any(name_lower.startswith(p) for p in _CHAT_ONLY_PREFIXES):
        return False
    return any(p in name_lower for p in _RESPONSES_API_PATTERNS)


def _normalize_ai_message(msg: AIMessage) -> AIMessage:
    """Normalize AIMessage content from list-of-blocks (Responses API) to str."""
    content = msg.content
    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, dict):
                text_parts.append(block.get("text", ""))
            elif isinstance(block, str):
                text_parts.append(block)
        content = "\n".join(p for p in text_parts if p)
    if not isinstance(content, str):
        content = str(content)
    return AIMessage(
        content=content,
        tool_calls=msg.tool_calls if msg.tool_calls else [],
        id=msg.id,
    )


# ===================================================================
# Environment probing
# ===================================================================

def _probe_environment(workspace_path: Path) -> list[str]:
    """Detect OS, build tools, and recommended cmake generator."""
    import subprocess

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
    """Update build state based on recent tool outputs."""
    for msg in reversed(messages):
        if not isinstance(msg, ToolMessage):
            continue
        text = str(msg.content)
        if "[cmd]=" not in text or "[exit_code]=" not in text:
            continue

        cmd = _extract_cmd(text) or ""
        exit_code = _extract_exit_code(text)
        cmd_lower = cmd.lower()

        new_state = BuildState(
            status=current.status,
            configured=current.configured,
            built=current.built,
            tested=current.tested,
            last_exit_code=exit_code,
            last_error=current.last_error,
            consecutive_errors=current.consecutive_errors,
        )

        if exit_code is not None and exit_code != 0:
            error_line = _first_error_line(text) or f"exit code {exit_code}"
            new_state.status = "FAILED"
            new_state.last_error = error_line[:200]
            new_state.consecutive_errors = current.consecutive_errors + 1
        else:
            new_state.consecutive_errors = 0
            new_state.last_error = ""

        # Detect lifecycle stage
        if re.search(r"\bcmake\b", cmd_lower) and "--build" not in cmd_lower:
            if exit_code == 0:
                new_state.configured = True
                new_state.status = "CONFIGURING"
        elif re.search(r"--build|\bmake\b|\bninja\b|\bmingw32-make\b", cmd_lower):
            if exit_code == 0:
                new_state.built = True
                new_state.status = "BUILDING"
        elif re.search(r"\bctest\b|\btest\b", cmd_lower):
            if exit_code == 0:
                new_state.tested = True
                new_state.status = "SUCCESS"
            elif exit_code is not None:
                new_state.status = "FAILED"

        return new_state
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
