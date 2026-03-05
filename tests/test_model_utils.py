"""Tests for agent/model_utils.py — model detection, build_chat_model, normalization."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage

from agent.model_utils import extract_text, is_responses_model, normalize_ai_message


# ---------------------------------------------------------------------------
# is_responses_model — regex-based detection
# ---------------------------------------------------------------------------

class TestIsResponsesModel:
    """Verify Responses API detection with versioned and unversioned model names."""

    # --- Chat completions models (should return False) ---

    @pytest.mark.parametrize(
        "model",
        [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4o-2024-08-06",
            "gpt-4-turbo",
            "gpt-4-0613",
            "gpt-4",
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-0125",
        ],
    )
    def test_chat_models_not_responses(self, model: str) -> None:
        assert is_responses_model(model) is False

    # --- Responses API models (should return True) ---

    @pytest.mark.parametrize(
        "model",
        [
            "codex",
            "gpt-5.3-codex",
            "o1",
            "o1-mini",
            "o1-mini-2025-01-31",
            "o3",
            "o3-mini",
            "o3-mini-2025-01-31",
            "o4-mini",
            "o4",
        ],
    )
    def test_responses_models(self, model: str) -> None:
        assert is_responses_model(model) is True

    # --- Edge cases ---

    def test_empty_string(self) -> None:
        assert is_responses_model("") is False

    def test_unknown_model(self) -> None:
        assert is_responses_model("claude-3-opus") is False

    def test_whitespace_stripped(self) -> None:
        assert is_responses_model("  o3-mini  ") is True


# ---------------------------------------------------------------------------
# extract_text
# ---------------------------------------------------------------------------

class TestExtractText:
    def test_string_passthrough(self) -> None:
        assert extract_text("hello") == "hello"

    def test_strips_whitespace(self) -> None:
        assert extract_text("  hello  ") == "hello"

    def test_list_of_dicts(self) -> None:
        content = [{"text": "Hello"}, {"text": " world"}]
        assert extract_text(content) == "Hello\n world"

    def test_list_of_strings(self) -> None:
        content = ["Hello", "world"]
        assert extract_text(content) == "Hello\nworld"

    def test_mixed_list(self) -> None:
        content = [{"text": "Hello"}, "world"]
        assert extract_text(content) == "Hello\nworld"

    def test_empty_list(self) -> None:
        assert extract_text([]) == ""

    def test_other_type(self) -> None:
        assert extract_text(42) == "42"


# ---------------------------------------------------------------------------
# normalize_ai_message
# ---------------------------------------------------------------------------

class TestNormalizeAiMessage:
    def test_string_content_preserved(self) -> None:
        msg = AIMessage(content="hello", id="1")
        result = normalize_ai_message(msg)
        assert result.content == "hello"
        assert result.id == "1"

    def test_list_content_normalized(self) -> None:
        msg = AIMessage(content=[{"text": "Hello"}, {"text": "world"}], id="2")
        result = normalize_ai_message(msg)
        assert isinstance(result.content, str)
        assert "Hello" in result.content
        assert "world" in result.content

    def test_tool_calls_preserved(self) -> None:
        tc = [{"id": "tc1", "name": "test", "args": {}}]
        msg = AIMessage(content="hi", tool_calls=tc, id="3")
        result = normalize_ai_message(msg)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["id"] == "tc1"
        assert result.tool_calls[0]["name"] == "test"
