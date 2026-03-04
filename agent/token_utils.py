from __future__ import annotations

from typing import Iterable

import tiktoken
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage


def _get_encoder(model_name: str) -> tiktoken.Encoding:
    """Get a tiktoken encoder, falling back to cl100k_base."""
    try:
        return tiktoken.encoding_for_model(model_name)
    except Exception:
        return tiktoken.get_encoding("cl100k_base")


def estimate_token_count(messages: Iterable[BaseMessage], model_name: str) -> int:
    """Estimate total tokens for a list of messages (content + overhead)."""
    encoder = _get_encoder(model_name)
    total = 0
    for message in messages:
        content = _message_text(message)
        total += len(encoder.encode(content)) + 4  # per-message overhead
        # Account for tool_calls serialization
        if isinstance(message, AIMessage) and getattr(message, "tool_calls", None):
            tool_text = str(message.tool_calls)
            total += len(encoder.encode(tool_text))
    return total


def estimate_text_tokens(text: str, model_name: str) -> int:
    """Estimate tokens for a plain text string."""
    encoder = _get_encoder(model_name)
    return len(encoder.encode(text))


def trim_text_to_token_budget(text: str, model_name: str, max_tokens: int) -> str:
    """Trim text to fit within a token budget using binary-ish shrink."""
    if max_tokens <= 0:
        return ""
    if estimate_text_tokens(text, model_name) <= max_tokens:
        return text
    # Binary-ish reduction
    reduced = text
    while estimate_text_tokens(reduced, model_name) > max_tokens and len(reduced) > 32:
        reduced = reduced[: int(len(reduced) * 0.8)]
    return reduced


def fit_messages_to_budget(
    messages: list[BaseMessage],
    model_name: str,
    input_budget: int,
) -> list[BaseMessage]:
    """Drop oldest tool-observation pairs, then oldest messages, until under budget."""
    current = list(messages)
    while estimate_token_count(current, model_name) > input_budget and len(current) > 2:
        if not _pop_oldest_tool_observation_pair(current):
            current.pop(1 if len(current) > 2 else 0)  # keep system prompt at [0]
    return current


def _pop_oldest_tool_observation_pair(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Remove the oldest AI(tool_calls) + ToolMessage(s) pair."""
    for i, msg in enumerate(messages):
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            # Collect following ToolMessages
            indexes = [i]
            for j in range(i + 1, len(messages)):
                if isinstance(messages[j], ToolMessage):
                    indexes.append(j)
                else:
                    break
            removed = []
            for idx in sorted(indexes, reverse=True):
                removed.append(messages.pop(idx))
            removed.reverse()
            return removed
    return []


def sanitize_tool_message_sequence(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Remove orphaned ToolMessages whose tool_call_id has no matching AIMessage."""
    allowed_ids: set[str] = set()
    sanitized: list[BaseMessage] = []

    for msg in messages:
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls or []:
                cid = str(tc.get("id", "")).strip()
                if cid:
                    allowed_ids.add(cid)
            sanitized.append(msg)
        elif isinstance(msg, ToolMessage):
            cid = str(getattr(msg, "tool_call_id", "") or "").strip()
            if cid and cid in allowed_ids:
                sanitized.append(msg)
        else:
            sanitized.append(msg)
    return sanitized


def _message_text(message: BaseMessage) -> str:
    """Extract text content from a message."""
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(str(part) for part in content)
    return str(content)
