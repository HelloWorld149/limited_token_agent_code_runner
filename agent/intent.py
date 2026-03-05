from __future__ import annotations

import asyncio
import difflib
import re
from typing import Literal

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
    "- EXIT: wants to end the session (quit, exit, bye, done)\n\n"
    "IMPORTANT: You will be given optional dialog context (the previous intent "
    "and a summary of the last assistant message). Use this context to correctly "
    "resolve ambiguous follow-ups:\n"
    "- If the user says 'yes', 'sure', 'do it', 'go ahead', 'please do' etc. "
    "and the assistant just offered an action, classify as the intent matching "
    "that action (EXPLORE, COMPILE, RUN, etc.), NOT as EXIT or QUESTION.\n"
    "- Only classify as EXIT when the user clearly wants to end the session.\n"
)

_VALID_INTENTS: set[str] = {"QUESTION", "COMPILE", "RUN", "EXPLORE", "EXIT"}
_FOLLOWUP_DECISIONS: set[str] = {"CONFIRM", "CANCEL", "EXIT", "NEW_REQUEST"}

FollowupDecision = Literal["CONFIRM", "CANCEL", "EXIT", "NEW_REQUEST"]

_FOLLOWUP_SYSTEM = (
    "You classify a short follow-up user reply in dialog context. "
    "Return EXACTLY one token: CONFIRM, CANCEL, EXIT, or NEW_REQUEST.\n"
    "Definitions:\n"
    "- CONFIRM: user agrees to proceed with the previous assistant action.\n"
    "- CANCEL: user declines, stops, or asks not to proceed.\n"
    "- EXIT: user wants to end the whole session.\n"
    "- NEW_REQUEST: user is asking something else.\n"
    "Treat typos and informal language robustly."
)


async def classify_intent_async(
    user_input: str,
    model_name: str = "gpt-4o-mini",
    *,
    previous_intent: str = "",
    last_ai_summary: str = "",
) -> Intent:
    """Classify user intent via a lightweight LLM call (~200 tokens).

    This runs as a separate async task concurrently with context preparation.
    It does NOT consume main 5000-token budget.

    When *previous_intent* and *last_ai_summary* are provided the classifier
    can resolve ambiguous follow-ups ("yes please", "do it") in a single
    pass — no second-stage follow-up classifier needed.
    """
    # Trim input — classifier only needs intent, not the full question (~200 tokens max)
    user_input = trim_text_to_token_budget(user_input, model_name, 200)

    # Build the user payload.  When dialog context is available, prepend it
    # so the LLM can resolve confirmations like "yes" or "do it".
    if previous_intent and last_ai_summary:
        last_ai_summary = trim_text_to_token_budget(
            last_ai_summary, model_name, 150
        )
        payload = (
            f"DIALOG CONTEXT (use to resolve ambiguous follow-ups):\n"
            f"Previous intent: {previous_intent}\n"
            f"Last assistant message summary: {last_ai_summary}\n\n"
            f"USER MESSAGE:\n{user_input}"
        )
    else:
        payload = user_input

    model = build_chat_model(model_name, temperature=0, max_tokens=10)
    messages = [
        SystemMessage(content=_CLASSIFIER_SYSTEM),
        HumanMessage(content=payload),
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
    *,
    previous_intent: str = "",
    last_ai_summary: str = "",
) -> Intent:
    """Synchronous wrapper for intent classification."""
    return run_async(
        classify_intent_async(
            user_input,
            model_name,
            previous_intent=previous_intent,
            last_ai_summary=last_ai_summary,
        )
    )


async def classify_followup_async(
    user_input: str,
    previous_intent: Intent,
    last_ai_message: str,
    model_name: str = "gpt-4o-mini",
) -> FollowupDecision:
    """Classify short follow-up replies using prior-turn context.

    This is designed for ambiguous confirmations like "go ahead", including typos.
    """
    user_input = trim_text_to_token_budget(user_input, model_name, 120)
    last_ai_message = trim_text_to_token_budget(last_ai_message, model_name, 240)

    model = build_chat_model(model_name, temperature=0, max_tokens=8)
    user_payload = (
        f"PREVIOUS_INTENT: {previous_intent}\n"
        f"LAST_ASSISTANT_MESSAGE:\n{last_ai_message}\n\n"
        f"USER_REPLY:\n{user_input}"
    )

    messages = [
        SystemMessage(content=_FOLLOWUP_SYSTEM),
        HumanMessage(content=user_payload),
    ]

    try:
        response = await model.ainvoke(messages)
        raw = extract_text(response.content).upper()
        for token in raw.split():
            cleaned = token.strip(".:,;!\"'")
            if cleaned in _FOLLOWUP_DECISIONS:
                return cleaned  # type: ignore[return-value]
        return _fallback_followup(user_input)
    except Exception:
        return _fallback_followup(user_input)


def classify_followup_sync(
    user_input: str,
    previous_intent: Intent,
    last_ai_message: str,
    model_name: str = "gpt-4o-mini",
) -> FollowupDecision:
    """Synchronous wrapper for follow-up classification."""
    return run_async(
        classify_followup_async(
            user_input=user_input,
            previous_intent=previous_intent,
            last_ai_message=last_ai_message,
            model_name=model_name,
        )
    )


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


def _normalize_text(text: str) -> str:
    lowered = text.lower().strip()
    lowered = re.sub(r"\s+", " ", lowered)
    lowered = re.sub(r"[^a-z0-9\s]", "", lowered)
    return lowered.strip()


def _fuzzy_contains(tokens: list[str], lexicon: set[str], cutoff: float = 0.82) -> bool:
    for token in tokens:
        if token in lexicon:
            return True
        if difflib.get_close_matches(token, lexicon, n=1, cutoff=cutoff):
            return True
    return False


def _fallback_followup(user_input: str) -> FollowupDecision:
    """Typo-tolerant heuristic fallback for follow-up classification failures."""
    normalized = _normalize_text(user_input)
    if not normalized:
        return "NEW_REQUEST"

    tokens = normalized.split()

    exit_words = {"exit", "quit", "bye", "goodbye", "q"}
    cancel_words = {"no", "stop", "cancel", "dont", "nevermind", "nvm", "nah"}
    confirm_words = {
        "yes", "y", "ok", "okay", "sure", "proceed", "continue", "go", "ahead", "do", "it"
    }

    if _fuzzy_contains(tokens, exit_words, cutoff=0.9):
        return "EXIT"
    if _fuzzy_contains(tokens, cancel_words):
        return "CANCEL"
    if _fuzzy_contains(tokens, confirm_words):
        return "CONFIRM"
    return "NEW_REQUEST"
