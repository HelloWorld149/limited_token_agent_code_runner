"""Shared model utilities used across the entire agent package.

This is the SINGLE SOURCE OF TRUTH for:
    - Model type detection (Responses API vs Chat Completions)
    - ChatOpenAI construction with correct API mode
    - Response content normalization (list-of-blocks -> str)
    - AIMessage normalization
    - Async-to-sync execution helper
"""

from __future__ import annotations

import asyncio
import re
from typing import Any

from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI


# ---------------------------------------------------------------------------
# Model type detection (Responses API vs Chat Completions)
# ---------------------------------------------------------------------------

# Regex patterns for models that use the Responses API.
# Using compiled regexes instead of simple substring matching so we can handle
# versioned model names (e.g. "o3-mini-2025-01-31") and future naming changes.
_RESPONSES_API_REGEXES: list[re.Pattern[str]] = [
    re.compile(r"\bcodex\b", re.IGNORECASE),          # codex family
    re.compile(r"\bo[134]\b", re.IGNORECASE),          # o1, o3, o4 reasoning models
    re.compile(r"\bo[134]-", re.IGNORECASE),            # o1-mini, o3-mini-2025-01-31, etc.
    re.compile(r"\bo[134]p\b", re.IGNORECASE),          # o1p, o3p, o4p (preview variants)
]

# Explicit chat completions models — checked FIRST so they are never
# accidentally matched by a Responses API pattern.
_CHAT_ONLY_REGEXES: list[re.Pattern[str]] = [
    re.compile(r"^gpt-4o", re.IGNORECASE),             # gpt-4o, gpt-4o-mini
    re.compile(r"^gpt-4-", re.IGNORECASE),             # gpt-4-turbo, gpt-4-0613 …
    re.compile(r"^gpt-3\.5", re.IGNORECASE),           # gpt-3.5-turbo …
    re.compile(r"^gpt-4\b", re.IGNORECASE),            # gpt-4 (exact)
]


def is_responses_model(model_name: str) -> bool:
    """Return True if the model should use the Responses API.

    Codex and reasoning models (o1/o3/o4) use the Responses API.
    Standard chat models (gpt-4o, gpt-3.5) use the chat completions API.

    This function uses compiled regex patterns instead of simple substring
    matching so it handles versioned suffixes and future naming changes
    without needing constant updates.
    """
    name = model_name.strip()
    # Chat-only models take priority
    if any(p.search(name) for p in _CHAT_ONLY_REGEXES):
        return False
    return any(p.search(name) for p in _RESPONSES_API_REGEXES)


# ---------------------------------------------------------------------------
# Unified ChatOpenAI construction
# ---------------------------------------------------------------------------

def build_chat_model(
    model_name: str,
    temperature: float = 0,
    max_tokens: int = 800,
) -> ChatOpenAI:
    """Build a ChatOpenAI instance with correct API mode for the model.

    Automatically enables Responses API for codex/reasoning models.
    ALL code that needs a ChatOpenAI instance should use this function
    instead of constructing ChatOpenAI directly.
    """
    use_responses = is_responses_model(model_name)
    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        **({"use_responses_api": True} if use_responses else {}),
    )


# ---------------------------------------------------------------------------
# Response content normalization
# ---------------------------------------------------------------------------

def extract_text(content: Any) -> str:
    """Extract plain text from LLM response content.

    The Responses API may return a list-of-blocks instead of a plain string.
    This function handles both formats uniformly.
    """
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(p for p in parts if p).strip()
    return str(content).strip()


def normalize_ai_message(msg: AIMessage) -> AIMessage:
    """Normalize AIMessage content from list-of-blocks (Responses API) to str.

    Returns a new AIMessage with string content, preserving tool_calls and id.
    """
    content = extract_text(msg.content)
    return AIMessage(
        content=content,
        tool_calls=msg.tool_calls if msg.tool_calls else [],
        id=msg.id,
    )


# ---------------------------------------------------------------------------
# Async-to-sync execution helper
# ---------------------------------------------------------------------------

def run_async(coro: Any) -> Any:
    """Run an async coroutine from synchronous code, handling event-loop edge cases.

    Handles three scenarios:
    1. No event loop exists -> asyncio.run()
    2. Event loop exists but not running -> loop.run_until_complete()
    3. Event loop is already running -> ThreadPoolExecutor + asyncio.run()
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run, coro).result(timeout=60)
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)
