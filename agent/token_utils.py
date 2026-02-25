from __future__ import annotations

from typing import Iterable
from langchain_core.messages import BaseMessage

try:
    import tiktoken
except Exception:  # pragma: no cover
    tiktoken = None


def estimate_token_count(messages: Iterable[BaseMessage], model_name: str) -> int:
    if tiktoken is not None:
        try:
            encoder = tiktoken.encoding_for_model(model_name)
        except Exception:
            encoder = tiktoken.get_encoding("cl100k_base")

        total = 0
        for message in messages:
            content = _message_text(message)
            total += len(encoder.encode(content)) + 4
        return total

    total_chars = sum(len(_message_text(message)) for message in messages)
    return total_chars // 4


def estimate_text_tokens(text: str, model_name: str) -> int:
    if tiktoken is not None:
        try:
            encoder = tiktoken.encoding_for_model(model_name)
        except Exception:
            encoder = tiktoken.get_encoding("cl100k_base")
        return len(encoder.encode(text))
    return len(text) // 4


def trim_text_to_token_budget(text: str, model_name: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return ""
    if estimate_text_tokens(text, model_name) <= max_tokens:
        return text

    reduced = text
    while estimate_text_tokens(reduced, model_name) > max_tokens and len(reduced) > 32:
        reduced = reduced[: int(len(reduced) * 0.8)]
    return reduced


def _message_text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(str(part) for part in content)
    return str(content)
