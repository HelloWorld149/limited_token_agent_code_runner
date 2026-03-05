"""Tests for agent/token_utils.py — token counting, trimming, budget fitting, sanitization."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from agent.token_utils import (
    estimate_text_tokens,
    estimate_token_count,
    fit_messages_to_budget,
    sanitize_tool_message_sequence,
    trim_text_to_token_budget,
)

MODEL = "gpt-4o-mini"  # uses cl100k_base encoder


# ---------------------------------------------------------------------------
# estimate_text_tokens
# ---------------------------------------------------------------------------

class TestEstimateTextTokens:
    def test_empty_string(self) -> None:
        assert estimate_text_tokens("", MODEL) == 0

    def test_short_string(self) -> None:
        tokens = estimate_text_tokens("hello world", MODEL)
        assert 1 <= tokens <= 5

    def test_longer_string(self) -> None:
        text = "The quick brown fox jumps over the lazy dog. " * 20
        tokens = estimate_text_tokens(text, MODEL)
        assert tokens > 50


# ---------------------------------------------------------------------------
# trim_text_to_token_budget  (binary search implementation)
# ---------------------------------------------------------------------------

class TestTrimTextToTokenBudget:
    def test_empty_budget_returns_empty(self) -> None:
        assert trim_text_to_token_budget("hello world", MODEL, 0) == ""

    def test_negative_budget_returns_empty(self) -> None:
        assert trim_text_to_token_budget("hello world", MODEL, -5) == ""

    def test_text_within_budget_unchanged(self) -> None:
        text = "short text"
        result = trim_text_to_token_budget(text, MODEL, 1000)
        assert result == text

    def test_long_text_trimmed_to_budget(self) -> None:
        text = "word " * 2000  # ~2000 tokens
        result = trim_text_to_token_budget(text, MODEL, 100)
        result_tokens = estimate_text_tokens(result, MODEL)
        assert result_tokens <= 100
        assert len(result) > 0  # not empty

    def test_trimmed_result_is_prefix(self) -> None:
        text = "ABCDEFGHIJ " * 500
        result = trim_text_to_token_budget(text, MODEL, 50)
        assert text.startswith(result)

    def test_binary_search_is_tight(self) -> None:
        """The binary search should find a result close to the budget, not too small."""
        text = "sample text with tokens " * 200
        budget = 100
        result = trim_text_to_token_budget(text, MODEL, budget)
        result_tokens = estimate_text_tokens(result, MODEL)
        # Should be within budget but not wastefully small
        assert result_tokens <= budget
        assert result_tokens >= budget * 0.5  # at least half used


# ---------------------------------------------------------------------------
# estimate_token_count (message list)
# ---------------------------------------------------------------------------

class TestEstimateTokenCount:
    def test_empty_list(self) -> None:
        assert estimate_token_count([], MODEL) == 0

    def test_single_message(self) -> None:
        msgs = [HumanMessage(content="hello")]
        count = estimate_token_count(msgs, MODEL)
        assert count > 0

    def test_multiple_messages_increase_count(self) -> None:
        msgs1 = [HumanMessage(content="hello")]
        msgs2 = [HumanMessage(content="hello"), AIMessage(content="world")]
        assert estimate_token_count(msgs2, MODEL) > estimate_token_count(msgs1, MODEL)

    def test_tool_calls_counted(self) -> None:
        no_tools = [AIMessage(content="hi")]
        with_tools = [
            AIMessage(
                content="hi",
                tool_calls=[{"id": "1", "name": "test", "args": {"x": 1}}],
            )
        ]
        assert estimate_token_count(with_tools, MODEL) > estimate_token_count(
            no_tools, MODEL
        )


# ---------------------------------------------------------------------------
# fit_messages_to_budget
# ---------------------------------------------------------------------------

class TestFitMessagesToBudget:
    def test_already_within_budget(self) -> None:
        msgs = [SystemMessage(content="sys"), HumanMessage(content="hi")]
        result = fit_messages_to_budget(msgs, MODEL, 5000)
        assert len(result) == 2

    def test_drops_oldest_non_system(self) -> None:
        msgs = [
            SystemMessage(content="system prompt"),
            HumanMessage(content="old message " * 200),
            HumanMessage(content="new message"),
        ]
        result = fit_messages_to_budget(msgs, MODEL, 50)
        # System message should be preserved
        assert isinstance(result[0], SystemMessage)
        assert len(result) < len(msgs)

    def test_drops_tool_observation_pairs_first(self) -> None:
        msgs = [
            SystemMessage(content="sys"),
            AIMessage(
                content="",
                tool_calls=[{"id": "tc1", "name": "test", "args": {}}],
            ),
            ToolMessage(content="big output " * 100, tool_call_id="tc1"),
            HumanMessage(content="follow up"),
        ]
        result = fit_messages_to_budget(msgs, MODEL, 100)
        # The tool pair should be dropped before the human message
        tool_msgs = [m for m in result if isinstance(m, ToolMessage)]
        assert len(tool_msgs) == 0

    def test_preserves_minimum_messages(self) -> None:
        msgs = [SystemMessage(content="s"), HumanMessage(content="h")]
        result = fit_messages_to_budget(msgs, MODEL, 1)
        assert len(result) >= 2  # can't drop below 2


# ---------------------------------------------------------------------------
# sanitize_tool_message_sequence
# ---------------------------------------------------------------------------

class TestSanitizeToolMessageSequence:
    def test_orphaned_tool_message_removed(self) -> None:
        msgs = [
            HumanMessage(content="hi"),
            ToolMessage(content="orphan", tool_call_id="missing_id"),
            AIMessage(content="response"),
        ]
        result = sanitize_tool_message_sequence(msgs)
        tool_msgs = [m for m in result if isinstance(m, ToolMessage)]
        assert len(tool_msgs) == 0

    def test_matched_tool_message_kept(self) -> None:
        msgs = [
            AIMessage(
                content="",
                tool_calls=[{"id": "tc1", "name": "test", "args": {}}],
            ),
            ToolMessage(content="result", tool_call_id="tc1"),
        ]
        result = sanitize_tool_message_sequence(msgs)
        tool_msgs = [m for m in result if isinstance(m, ToolMessage)]
        assert len(tool_msgs) == 1

    def test_non_tool_messages_preserved(self) -> None:
        msgs = [
            SystemMessage(content="sys"),
            HumanMessage(content="hi"),
            AIMessage(content="hello"),
        ]
        result = sanitize_tool_message_sequence(msgs)
        assert len(result) == 3
