from __future__ import annotations

from typing import Iterable

import tiktoken
from langchain_core.messages import BaseMessage


def _get_encoder(model_name: str) -> tiktoken.Encoding:
    """Execute function `_get_encoder`.

    This routine is part of the agent workflow and keeps its existing runtime behavior.

    Args:
        model_name (str): Input value used by this routine.

    Returns:
        tiktoken.Encoding: Result produced by this routine.
    """
    try:
        return tiktoken.encoding_for_model(model_name)
    except Exception:
        return tiktoken.get_encoding("cl100k_base")


def estimate_token_count(messages: Iterable[BaseMessage], model_name: str) -> int:
    """Execute function `estimate_token_count`.

    This routine is part of the agent workflow and keeps its existing runtime behavior.

    Args:
        messages (Iterable[BaseMessage]): Input value used by this routine.
        model_name (str): Input value used by this routine.

    Returns:
        int: Result produced by this routine.
    """
    encoder = _get_encoder(model_name)
    total = 0
    for message in messages:
        content = _message_text(message)
        total += len(encoder.encode(content)) + 4
    return total


def estimate_text_tokens(text: str, model_name: str) -> int:
    """Execute function `estimate_text_tokens`.

    This routine is part of the agent workflow and keeps its existing runtime behavior.

    Args:
        text (str): Input value used by this routine.
        model_name (str): Input value used by this routine.

    Returns:
        int: Result produced by this routine.
    """
    encoder = _get_encoder(model_name)
    return len(encoder.encode(text))


def trim_text_to_token_budget(text: str, model_name: str, max_tokens: int) -> str:
    """Execute function `trim_text_to_token_budget`.

    This routine is part of the agent workflow and keeps its existing runtime behavior.

    Args:
        text (str): Input value used by this routine.
        model_name (str): Input value used by this routine.
        max_tokens (int): Input value used by this routine.

    Returns:
        str: Result produced by this routine.
    """
    if max_tokens <= 0:
        return ""
    if estimate_text_tokens(text, model_name) <= max_tokens:
        return text

    reduced = text
    while estimate_text_tokens(reduced, model_name) > max_tokens and len(reduced) > 32:
        reduced = reduced[: int(len(reduced) * 0.8)]
    return reduced


def _message_text(message: BaseMessage) -> str:
    """Execute function `_message_text`.

    This routine is part of the agent workflow and keeps its existing runtime behavior.

    Args:
        message (BaseMessage): Input value used by this routine.

    Returns:
        str: Result produced by this routine.
    """
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(str(part) for part in content)
    return str(content)
