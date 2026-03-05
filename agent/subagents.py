"""Subagent modules that run as independent LLM calls with their own token budgets.

Each subagent operates outside the main 5000-token budget, allowing the system
to effectively expand its reasoning capacity by distributing work across multiple
focused LLM calls.

Subagents:
    1. Retrieval Subagent — reads raw code files, produces compressed digests
    2. Tool Output Summarizer — condenses large tool outputs (build logs, test results)
"""

from __future__ import annotations

import re
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from agent.token_utils import estimate_text_tokens, trim_text_to_token_budget


# ---------------------------------------------------------------------------
# Retrieval Subagent
# ---------------------------------------------------------------------------

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
    """Compress raw code chunks into a dense retrieval digest.

    This runs as a SEPARATE LLM call with its own token budget, outside
    the main 5000-token constraint.

    Args:
        user_query: The user's original question/request.
        raw_code_chunks: List of raw code snippets (file path + content).
        model_name: Model to use for compression (cheap/fast model).
        max_input_tokens: Max tokens for the subagent's input.
        max_output_tokens: Max tokens for the subagent's output.

    Returns:
        Compressed code digest string, ready for injection into main context.
    """
    if not raw_code_chunks:
        return ""

    system_prompt = _RETRIEVAL_SYSTEM_PROMPT.format(max_output_tokens=max_output_tokens)

    # Combine code chunks, trimming to budget
    combined_code = "\n\n".join(raw_code_chunks)
    # Reserve tokens for system prompt (~100) + user query (~50) + overhead
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

    model = ChatOpenAI(
        model=model_name,
        temperature=0,
        max_tokens=max_output_tokens,
    )

    try:
        response = await model.ainvoke(messages)
        content = response.content
        if isinstance(content, list):
            content = "\n".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in content
            )
        return str(content).strip()
    except Exception as exc:
        # Fallback: return truncated raw code if subagent fails
        fallback = trim_text_to_token_budget(
            combined_code, model_name, max_output_tokens
        )
        return f"[retrieval subagent failed: {type(exc).__name__}]\n{fallback}"


def retrieval_subagent_sync(
    user_query: str,
    raw_code_chunks: list[str],
    model_name: str = "gpt-4o-mini",
    max_input_tokens: int = 3500,
    max_output_tokens: int = 400,
) -> str:
    """Synchronous wrapper for retrieval_subagent_async."""
    import asyncio

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(
                    asyncio.run,
                    retrieval_subagent_async(
                        user_query, raw_code_chunks, model_name,
                        max_input_tokens, max_output_tokens,
                    ),
                ).result(timeout=30)
        return loop.run_until_complete(
            retrieval_subagent_async(
                user_query, raw_code_chunks, model_name,
                max_input_tokens, max_output_tokens,
            )
        )
    except RuntimeError:
        return asyncio.run(
            retrieval_subagent_async(
                user_query, raw_code_chunks, model_name,
                max_input_tokens, max_output_tokens,
            )
        )


# ---------------------------------------------------------------------------
# Tool Output Summarizer Subagent
# ---------------------------------------------------------------------------

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

# Patterns indicating output that benefits from summarization
_LONG_OUTPUT_PATTERNS = [
    re.compile(r"\[stdout\]", re.IGNORECASE),
    re.compile(r"\[stderr\]", re.IGNORECASE),
]


def should_summarize_tool_output(tool_output: str, min_tokens: int = 200) -> bool:
    """Determine if a tool output is large enough to benefit from summarization.

    Small outputs or non-command outputs are returned as-is.
    """
    if not any(p.search(tool_output) for p in _LONG_OUTPUT_PATTERNS):
        return False
    # Only summarize if the output is large enough
    # Use a rough char estimate (1 token ≈ 4 chars) to avoid expensive tokenization
    return len(tool_output) > min_tokens * 4


async def tool_output_summarizer_async(
    tool_output: str,
    command: str = "",
    model_name: str = "gpt-4o-mini",
    max_input_tokens: int = 3500,
    max_output_tokens: int = 200,
) -> str:
    """Compress a large tool output into a compact summary.

    This runs as a SEPARATE LLM call with its own token budget, outside
    the main 5000-token constraint.

    Args:
        tool_output: Raw output from execute_shell_command.
        command: The command that was run (for context).
        model_name: Model to use for summarization.
        max_input_tokens: Max tokens for the subagent's input.
        max_output_tokens: Max tokens for the subagent's output.

    Returns:
        Compressed summary of the tool output.
    """
    system_prompt = _TOOL_SUMMARIZER_SYSTEM_PROMPT.format(
        max_output_tokens=max_output_tokens
    )

    # Trim output to fit input budget (reserve ~100 for system + overhead)
    trimmed_output = trim_text_to_token_budget(
        tool_output, model_name, max_input_tokens - 150
    )

    user_content = f"COMMAND: {command}\n\nRAW OUTPUT:\n{trimmed_output}"

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_content),
    ]

    model = ChatOpenAI(
        model=model_name,
        temperature=0,
        max_tokens=max_output_tokens,
    )

    try:
        response = await model.ainvoke(messages)
        content = response.content
        if isinstance(content, list):
            content = "\n".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in content
            )
        summary = str(content).strip()
        # Prepend a marker so the main LLM knows this is a compressed summary
        return f"[compressed output]\n{summary}"
    except Exception as exc:
        # Fallback: return the original truncated output
        fallback = trim_text_to_token_budget(
            tool_output, model_name, max_output_tokens
        )
        return f"[summarizer failed: {type(exc).__name__}]\n{fallback}"


def tool_output_summarizer_sync(
    tool_output: str,
    command: str = "",
    model_name: str = "gpt-4o-mini",
    max_input_tokens: int = 3500,
    max_output_tokens: int = 200,
) -> str:
    """Synchronous wrapper for tool_output_summarizer_async."""
    import asyncio

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(
                    asyncio.run,
                    tool_output_summarizer_async(
                        tool_output, command, model_name,
                        max_input_tokens, max_output_tokens,
                    ),
                ).result(timeout=30)
        return loop.run_until_complete(
            tool_output_summarizer_async(
                tool_output, command, model_name,
                max_input_tokens, max_output_tokens,
            )
        )
    except RuntimeError:
        return asyncio.run(
            tool_output_summarizer_async(
                tool_output, command, model_name,
                max_input_tokens, max_output_tokens,
            )
        )
