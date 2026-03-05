from __future__ import annotations

import asyncio

from langchain_core.messages import HumanMessage, SystemMessage

from agent.model_utils import build_chat_model, extract_text, run_async
from agent.state import Intent
from agent.token_utils import trim_text_to_token_budget


_CLASSIFIER_SYSTEM = (
    "Classify the user's intent into exactly one category. "
    "Reply with ONLY the category name, nothing else.\n"
    "Categories:\n"
    "- QUESTION: asking about code, architecture, explanations, how something works\n"
    "- COMPILE: wants to build or compile the project (cmake, make, build)\n"
    "- RUN: wants to run tests or execute specific commands\n"
    "- EXPLORE: wants to browse, search, or navigate the codebase\n"
    "- EXIT: wants to end the session (quit, exit, bye, done)\n"
)

_VALID_INTENTS: set[str] = {"QUESTION", "COMPILE", "RUN", "EXPLORE", "EXIT"}


async def classify_intent_async(
    user_input: str,
    model_name: str = "gpt-4o-mini",
) -> Intent:
    """Classify user intent via a lightweight LLM call (~110 tokens).

    This runs as a separate async task concurrently with context preparation.
    It does NOT consume main 5000-token budget.
    """
    # Trim input — classifier only needs intent, not the full question (~200 tokens max)
    user_input = trim_text_to_token_budget(user_input, model_name, 200)

    model = build_chat_model(model_name, temperature=0, max_tokens=10)
    messages = [
        SystemMessage(content=_CLASSIFIER_SYSTEM),
        HumanMessage(content=user_input),
    ]
    try:
        response = await model.ainvoke(messages)
        raw = extract_text(response.content).upper()
        # Parse — take first valid intent word found
        for token in raw.split():
            cleaned = token.strip(".:,;!\"'")
            if cleaned in _VALID_INTENTS:
                return cleaned  # type: ignore[return-value]
        # Fallback heuristics
        return _fallback_classify(user_input)
    except Exception:
        return _fallback_classify(user_input)


def classify_intent_sync(
    user_input: str,
    model_name: str = "gpt-4o-mini",
) -> Intent:
    """Synchronous wrapper for intent classification."""
    return run_async(classify_intent_async(user_input, model_name))


def _fallback_classify(user_input: str) -> Intent:
    """Keyword-based fallback when the LLM classifier fails."""
    lowered = user_input.lower().strip()

    if lowered in ("exit", "quit", "bye", "done", "q"):
        return "EXIT"

    build_kw = ("build", "compile", "cmake", "make", "configure", "ninja")
    if any(kw in lowered for kw in build_kw):
        return "COMPILE"

    run_kw = ("test", "run", "ctest", "execute")
    if any(kw in lowered for kw in run_kw):
        return "RUN"

    explore_kw = ("list", "ls", "dir", "tree", "browse", "search", "find", "grep", "show files")
    if any(kw in lowered for kw in explore_kw):
        return "EXPLORE"

    return "QUESTION"
