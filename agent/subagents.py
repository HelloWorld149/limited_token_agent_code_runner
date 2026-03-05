"""Subagent modules that run as independent LLM calls with their own token budgets.

Each subagent runs as a separate LLM call capped at 5000 tokens, allowing the
system to expand its reasoning capacity by distributing work across multiple
focused LLM calls.

Subagents:
    1. Retrieval Subagent — reads raw code files, produces compressed digests
    2. Tool Output Summarizer — condenses large tool outputs (build logs, test results)
    3. Conversation Compressor — re-summarizes rolling conversation history
    4. Multi-Hop Decomposer — breaks complex questions into sub-queries, each
       investigated by a separate subagent, then merges findings
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from agent.model_utils import build_chat_model, extract_text, run_async
from agent.token_utils import estimate_text_tokens, trim_text_to_token_budget


# ---------------------------------------------------------------------------
# Budget enforcement: every LLM call (including subagents) stays <= 5000 tokens
# ---------------------------------------------------------------------------

_SUBAGENT_TOKEN_CAP = 5000
_MAX_QUERY_TOKENS = 300  # user queries trimmed before subagent processing


def _enforce_budget(max_input: int, max_output: int) -> tuple[int, int]:
    """Clamp input + output to never exceed the 5000-token cap."""
    total = max_input + max_output
    if total <= _SUBAGENT_TOKEN_CAP:
        return max_input, max_output
    # Shrink input budget first to preserve output quality
    max_input = _SUBAGENT_TOKEN_CAP - max_output
    if max_input < 200:
        max_input = 200
        max_output = _SUBAGENT_TOKEN_CAP - max_input
    return max_input, max_output


# ===================================================================
# 1. Retrieval Subagent
# ===================================================================

_RETRIEVAL_SYSTEM_PROMPT = """\
You are a code retrieval assistant. Your ONLY job is to read the provided code \
snippets and produce a compressed, information-dense summary that another AI \
can use to answer the user's question.

Rules:
1. Focus on what is RELEVANT to the user's question — skip irrelevant code.
2. Preserve key facts: file paths, function/class names, line numbers, \
   signatures, return types, important logic.
3. Use compact notation — abbreviate obvious patterns, skip boilerplate.
4. Your output MUST be under {max_output_tokens} tokens.
5. If code is insufficient to answer the question, say "INSUFFICIENT: <reason>".
6. Never answer the question yourself — just summarize the relevant code.
""".strip()


async def retrieval_subagent_async(
    user_query: str,
    raw_code_chunks: list[str],
    model_name: str = "gpt-4o-mini",
    max_input_tokens: int = 3500,
    max_output_tokens: int = 400,
) -> str:
    """Compress raw code chunks into a dense retrieval digest."""
    if not raw_code_chunks:
        return ""

    max_input_tokens, max_output_tokens = _enforce_budget(max_input_tokens, max_output_tokens)
    user_query = trim_text_to_token_budget(user_query, model_name, _MAX_QUERY_TOKENS)

    system_prompt = _RETRIEVAL_SYSTEM_PROMPT.format(max_output_tokens=max_output_tokens)
    combined_code = "\n\n".join(raw_code_chunks)
    code_budget = max_input_tokens - 200
    combined_code = trim_text_to_token_budget(combined_code, model_name, code_budget)

    user_content = (
        f"USER QUESTION: {user_query}\n\n"
        f"CODE SNIPPETS:\n{combined_code}\n\n"
        "Produce a compressed summary of the relevant code for answering this question."
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_content),
    ]

    model = build_chat_model(model_name, temperature=0, max_tokens=max_output_tokens)

    try:
        response = await model.ainvoke(messages)
        return extract_text(response.content)
    except Exception as exc:
        fallback = trim_text_to_token_budget(combined_code, model_name, max_output_tokens)
        return f"[retrieval subagent failed: {type(exc).__name__}]\n{fallback}"


def retrieval_subagent_sync(
    user_query: str,
    raw_code_chunks: list[str],
    model_name: str = "gpt-4o-mini",
    max_input_tokens: int = 3500,
    max_output_tokens: int = 400,
) -> str:
    """Synchronous wrapper for retrieval_subagent_async."""
    return run_async(
        retrieval_subagent_async(
            user_query, raw_code_chunks, model_name,
            max_input_tokens, max_output_tokens,
        )
    )


# ===================================================================
# 2. Tool Output Summarizer Subagent
# ===================================================================

_TOOL_SUMMARIZER_SYSTEM_PROMPT = """\
You are a tool output summarizer. Your ONLY job is to condense the raw output \
from a shell command into a compact, actionable summary.

Rules:
1. Preserve ALL critical information: error messages, file paths, line numbers, \
   pass/fail counts, exit codes.
2. Remove repetitive/boilerplate output (compiler flags, progress bars, etc.).
3. For build output: focus on errors, warnings, and success/failure status.
4. For test output: report total/passed/failed counts and list failing test names.
5. Your output MUST be under {max_output_tokens} tokens.
6. Use structured format: status line first, then key details.
""".strip()

_LONG_OUTPUT_PATTERNS = [
    re.compile(r"\[stdout\]", re.IGNORECASE),
    re.compile(r"\[stderr\]", re.IGNORECASE),
]


def should_summarize_tool_output(tool_output: str, min_tokens: int = 200) -> bool:
    """Determine if a tool output is large enough to benefit from summarization."""
    if not any(p.search(tool_output) for p in _LONG_OUTPUT_PATTERNS):
        return False
    return len(tool_output) > min_tokens * 4


async def tool_output_summarizer_async(
    tool_output: str,
    command: str = "",
    model_name: str = "gpt-4o-mini",
    max_input_tokens: int = 3500,
    max_output_tokens: int = 200,
) -> str:
    """Compress a large tool output into a compact summary."""
    max_input_tokens, max_output_tokens = _enforce_budget(max_input_tokens, max_output_tokens)

    system_prompt = _TOOL_SUMMARIZER_SYSTEM_PROMPT.format(
        max_output_tokens=max_output_tokens
    )

    trimmed_output = trim_text_to_token_budget(
        tool_output, model_name, max_input_tokens - 150
    )

    user_content = f"COMMAND: {command}\n\nRAW OUTPUT:\n{trimmed_output}"

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_content),
    ]

    model = build_chat_model(model_name, temperature=0, max_tokens=max_output_tokens)

    try:
        response = await model.ainvoke(messages)
        summary = extract_text(response.content)
        return f"[compressed output]\n{summary}"
    except Exception as exc:
        fallback = trim_text_to_token_budget(tool_output, model_name, max_output_tokens)
        return f"[summarizer failed: {type(exc).__name__}]\n{fallback}"


def tool_output_summarizer_sync(
    tool_output: str,
    command: str = "",
    model_name: str = "gpt-4o-mini",
    max_input_tokens: int = 3500,
    max_output_tokens: int = 200,
) -> str:
    """Synchronous wrapper for tool_output_summarizer_async."""
    return run_async(
        tool_output_summarizer_async(
            tool_output, command, model_name,
            max_input_tokens, max_output_tokens,
        )
    )


# ===================================================================
# 3. Conversation Compressor Subagent
# ===================================================================

_CONVERSATION_COMPRESSOR_SYSTEM_PROMPT = """\
You are a conversation compressor. Your ONLY job is to produce a dense, \
factual summary of the conversation history so far.

Rules:
1. Preserve ALL discovered facts: file purposes, architecture decisions, \
   build results, test outcomes, error diagnostics, file paths.
2. Track what the user has asked and what was answered.
3. Note the current build state (configured? built? tested? errors?).
4. Drop conversational fluff — keep only information-dense facts.
5. Your output MUST be under {max_output_tokens} tokens.
6. Use bullet points for individual facts. Group by topic.
""".strip()


async def conversation_compressor_async(
    old_summary: str,
    recent_messages_text: str,
    model_name: str = "gpt-4o-mini",
    max_input_tokens: int = 3000,
    max_output_tokens: int = 400,
) -> str:
    """Re-summarize conversation history into a tight factual digest.

    Takes the previous rolling summary + recent message pairs and produces
    an updated summary. The LLM can merge, deduplicate, and prioritize facts.

    Runs as a separate LLM call with its own 5000-token cap.
    """
    max_input_tokens, max_output_tokens = _enforce_budget(max_input_tokens, max_output_tokens)

    system_prompt = _CONVERSATION_COMPRESSOR_SYSTEM_PROMPT.format(
        max_output_tokens=max_output_tokens
    )

    input_text = ""
    if old_summary:
        input_text += f"PREVIOUS SUMMARY:\n{old_summary}\n\n"
    input_text += f"RECENT CONVERSATION:\n{recent_messages_text}"
    input_text = trim_text_to_token_budget(input_text, model_name, max_input_tokens - 150)

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=input_text),
    ]

    model = build_chat_model(model_name, temperature=0, max_tokens=max_output_tokens)

    try:
        response = await model.ainvoke(messages)
        return extract_text(response.content)
    except Exception as exc:
        if old_summary:
            return trim_text_to_token_budget(
                f"{old_summary}\n[compression failed: {type(exc).__name__}]",
                model_name, max_output_tokens,
            )
        return trim_text_to_token_budget(
            recent_messages_text, model_name, max_output_tokens
        )


def conversation_compressor_sync(
    old_summary: str,
    recent_messages_text: str,
    model_name: str = "gpt-4o-mini",
    max_input_tokens: int = 3000,
    max_output_tokens: int = 400,
) -> str:
    """Synchronous wrapper for conversation_compressor_async."""
    return run_async(
        conversation_compressor_async(
            old_summary, recent_messages_text, model_name,
            max_input_tokens, max_output_tokens,
        )
    )


# ===================================================================
# 4. Multi-Hop Decomposer Subagent
# ===================================================================

_DECOMPOSER_SYSTEM_PROMPT = """\
You are a question decomposer. Your ONLY job is to analyze a complex question \
about a codebase and break it into 2-3 simpler, independent sub-queries that \
can each be answered by reading specific files.

Rules:
1. Each sub-query should target a SPECIFIC file, directory, or concept.
2. Sub-queries must be independent — answerable in parallel.
3. Output ONLY a JSON array of strings, e.g.: ["sub-query 1", "sub-query 2"]
4. If the question is already simple enough, output: ["<original question>"]
5. Max 3 sub-queries to stay within token budget.
""".strip()

_MERGER_SYSTEM_PROMPT = """\
You are a findings merger. You receive sub-findings from independent research \
about a codebase. Your job is to merge them into a single coherent, compressed \
digest that another AI can use to answer the original question.

Rules:
1. Eliminate duplicates across sub-findings.
2. Preserve all unique facts: file paths, function names, line numbers.
3. Organize by relevance to the original question.
4. Your output MUST be under {max_output_tokens} tokens.
""".strip()


async def _decompose_question(
    user_query: str,
    model_name: str,
    max_output_tokens: int = 150,
) -> list[str]:
    """Break a complex question into 2-3 independent sub-queries."""
    user_query = trim_text_to_token_budget(
        user_query, model_name,
        _SUBAGENT_TOKEN_CAP - max_output_tokens - 200,  # reserve for system prompt
    )
    messages = [
        SystemMessage(content=_DECOMPOSER_SYSTEM_PROMPT),
        HumanMessage(content=user_query),
    ]

    model = build_chat_model(model_name, temperature=0, max_tokens=max_output_tokens)

    try:
        response = await model.ainvoke(messages)
        text = extract_text(response.content)
        json_match = re.search(r"\[.*\]", text, re.DOTALL)
        if json_match:
            queries = json.loads(json_match.group(0))
            if isinstance(queries, list) and all(isinstance(q, str) for q in queries):
                return queries[:3]
        return [user_query]
    except Exception:
        return [user_query]


async def _merge_findings(
    user_query: str,
    sub_findings: list[str],
    model_name: str,
    max_output_tokens: int = 500,
) -> str:
    """Merge sub-findings from multiple parallel subagents into one digest."""
    system_prompt = _MERGER_SYSTEM_PROMPT.format(max_output_tokens=max_output_tokens)

    findings_text = "\n\n".join(
        f"--- Finding {i+1} ---\n{f}" for i, f in enumerate(sub_findings)
    )
    user_content = (
        f"ORIGINAL QUESTION: {user_query}\n\n"
        f"SUB-FINDINGS:\n{findings_text}"
    )
    # Enforce 5000-token cap on combined input
    user_content = trim_text_to_token_budget(
        user_content, model_name,
        _SUBAGENT_TOKEN_CAP - max_output_tokens - 200,  # reserve for system prompt
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_content),
    ]

    model = build_chat_model(model_name, temperature=0, max_tokens=max_output_tokens)

    try:
        response = await model.ainvoke(messages)
        return extract_text(response.content)
    except Exception:
        return "\n".join(sub_findings)


async def multi_hop_decomposer_async(
    user_query: str,
    raw_code_chunks: list[str],
    index_search_fn: Any = None,
    model_name: str = "gpt-4o-mini",
    max_output_tokens: int = 500,
    return_trace: bool = False,
) -> str | tuple[str, dict[str, Any]]:
    """Handle complex multi-aspect questions by decomposing, parallel-investigating, and merging.

    Flow:
        1. Decompose question into 2-3 sub-queries (1 LLM call)
        2. For each sub-query, run a retrieval subagent in parallel (N LLM calls)
        3. Merge all sub-findings into one compressed digest (1 LLM call)

    Each step is a separate LLM call with its own token budget.
    Total effective capacity: ~(3-5) x 5000 tokens worth of processing.

    Args:
        user_query: The complex user question.
        raw_code_chunks: Pre-retrieved code chunks (from index search).
        index_search_fn: Optional callable(query) -> list[str] that searches
            the index and reads file chunks for a sub-query.
        model_name: Model to use for all subagent calls.
        max_output_tokens: Token budget for the final merged output.
    """
    trace: dict[str, Any] = {
        "stage": "multi_hop",
        "sub_queries": [],
        "parallel_subagents": 0,
        "merge_used": False,
        "fallback_used": False,
        "total_subagents_used": 0,
    }

    sub_queries = await _decompose_question(user_query, model_name)
    trace["sub_queries"] = sub_queries
    trace["parallel_subagents"] = len(sub_queries)

    # Step 2: Parallel retrieval subagent per sub-query
    async def investigate(sub_query: str) -> str:
        if index_search_fn is not None:
            try:
                chunks = index_search_fn(sub_query)
            except Exception:
                chunks = raw_code_chunks
        else:
            chunks = raw_code_chunks

        return await retrieval_subagent_async(
            user_query=sub_query,
            raw_code_chunks=chunks,
            model_name=model_name,
            max_input_tokens=3500,
            max_output_tokens=350,
        )

    sub_findings = await asyncio.gather(
        *(investigate(sq) for sq in sub_queries),
        return_exceptions=True,
    )

    findings: list[str] = []
    for i, result in enumerate(sub_findings):
        if isinstance(result, Exception):
            findings.append(f"[sub-query {i+1} failed: {type(result).__name__}]")
        elif isinstance(result, str) and result.strip():
            findings.append(result)

    if not findings:
        trace["fallback_used"] = True
        trace["total_subagents_used"] = 2
        fallback = await retrieval_subagent_async(
            user_query=user_query,
            raw_code_chunks=raw_code_chunks,
            model_name=model_name,
        )
        if return_trace:
            return fallback, trace
        return fallback

    if len(findings) == 1:
        trace["total_subagents_used"] = 1 + len(sub_queries)
        if return_trace:
            return findings[0], trace
        return findings[0]

    trace["merge_used"] = True
    trace["total_subagents_used"] = 2 + len(sub_queries)
    merged = await _merge_findings(
        user_query, findings, model_name, max_output_tokens
    )
    if return_trace:
        return merged, trace
    return merged


def multi_hop_decomposer_sync(
    user_query: str,
    raw_code_chunks: list[str],
    index_search_fn: Any = None,
    model_name: str = "gpt-4o-mini",
    max_output_tokens: int = 500,
    return_trace: bool = False,
) -> str | tuple[str, dict[str, Any]]:
    """Synchronous wrapper for multi_hop_decomposer_async."""
    return run_async(
        multi_hop_decomposer_async(
            user_query, raw_code_chunks, index_search_fn,
            model_name, max_output_tokens, return_trace,
        )
    )


def is_complex_question(user_query: str) -> bool:
    """Heuristic: detect if a question is complex enough for multi-hop decomposition.

    Complex indicators:
    - Multiple question marks
    - Multiple files referenced
    - Comparison/relationship keywords
    - Long queries with multiple clauses
    """
    q_lower = user_query.lower()

    if user_query.count("?") >= 2:
        return True

    multi_keywords = [
        "and also", "compare", "difference between", "relationship between",
        "how does .* relate to", "both", "as well as", "versus", "vs",
    ]
    if any(re.search(kw, q_lower) for kw in multi_keywords):
        return True

    file_refs = re.findall(r"\b[\w-]+\.(?:cpp|hpp|h|cppm|cmake|txt|py)\b", q_lower)
    if len(file_refs) >= 2:
        return True

    if len(user_query) > 120 and ("," in user_query or " and " in q_lower):
        return True

    return False


__all__ = [
    "retrieval_subagent_sync",
    "tool_output_summarizer_sync",
    "conversation_compressor_sync",
    "multi_hop_decomposer_sync",
    "should_summarize_tool_output",
    "is_complex_question",
]
