"""Tests for agent/intent.py — keyword fallback classification,
and agent/nodes.py — _is_ambiguous_followup heuristic.
"""

from __future__ import annotations

import pytest

from agent.intent import _fallback_classify, _fallback_followup
from agent.nodes import _is_ambiguous_followup


# ---------------------------------------------------------------------------
# _fallback_classify — keyword-based intent classification
# ---------------------------------------------------------------------------

class TestFallbackClassify:
    @pytest.mark.parametrize("text", ["exit", "quit", "bye", "done", "q"])
    def test_exit_keywords(self, text: str) -> None:
        assert _fallback_classify(text) == "EXIT"

    @pytest.mark.parametrize(
        "text",
        ["build the project", "compile it", "run cmake", "make", "configure cmake", "ninja build"],
    )
    def test_build_keywords(self, text: str) -> None:
        assert _fallback_classify(text) == "COMPILE"

    @pytest.mark.parametrize(
        "text",
        ["run tests", "run ctest", "test the library", "execute the tests"],
    )
    def test_run_keywords(self, text: str) -> None:
        assert _fallback_classify(text) == "RUN"

    @pytest.mark.parametrize(
        "text",
        ["list files", "ls src", "show files", "search for json.hpp", "find the header", "grep for parse"],
    )
    def test_explore_keywords(self, text: str) -> None:
        assert _fallback_classify(text) == "EXPLORE"

    @pytest.mark.parametrize(
        "text",
        [
            "what does this function do",
            "explain the architecture",
            "how does parsing work",
            "hello",
        ],
    )
    def test_question_fallback(self, text: str) -> None:
        assert _fallback_classify(text) == "QUESTION"


# ---------------------------------------------------------------------------
# _fallback_followup — typo-tolerant follow-up classification
# ---------------------------------------------------------------------------

class TestFallbackFollowup:
    @pytest.mark.parametrize("text", ["yes", "y", "ok", "sure", "go ahead", "proceed"])
    def test_confirm(self, text: str) -> None:
        assert _fallback_followup(text) == "CONFIRM"

    @pytest.mark.parametrize("text", ["no", "stop", "cancel", "nevermind", "nah"])
    def test_cancel(self, text: str) -> None:
        assert _fallback_followup(text) == "CANCEL"

    @pytest.mark.parametrize("text", ["exit", "quit", "bye"])
    def test_exit(self, text: str) -> None:
        assert _fallback_followup(text) == "EXIT"

    def test_new_request(self) -> None:
        assert _fallback_followup("what is the file structure?") == "NEW_REQUEST"

    def test_empty_string(self) -> None:
        assert _fallback_followup("") == "NEW_REQUEST"


# ---------------------------------------------------------------------------
# _is_ambiguous_followup — heuristic gate for follow-up classifier
# ---------------------------------------------------------------------------

class TestIsAmbiguousFollowup:
    """Ensure the heuristic correctly triggers the follow-up classifier."""

    @pytest.mark.parametrize("text", ["yes please", "please do it!", "go ahead", "sure", "ok"])
    def test_short_confirmations_are_ambiguous(self, text: str) -> None:
        """Short confirmations must be treated as ambiguous, regardless of raw_intent."""
        assert _is_ambiguous_followup(text, "EXIT") is True
        assert _is_ambiguous_followup(text, "QUESTION") is True
        assert _is_ambiguous_followup(text, "COMPILE") is True

    @pytest.mark.parametrize("text", ["exit", "quit", "bye", "done", "q", "goodbye"])
    def test_hard_exit_phrases_not_ambiguous(self, text: str) -> None:
        """Unambiguous exit phrases should never trigger the followup classifier."""
        assert _is_ambiguous_followup(text, "EXIT") is False

    def test_empty_not_ambiguous(self) -> None:
        assert _is_ambiguous_followup("", "QUESTION") is False
        assert _is_ambiguous_followup("   ", "QUESTION") is False

    @pytest.mark.parametrize(
        "text",
        [
            "yes please do it",
            "no don't do that",
            "I said go",
            "do it!!!",
            "please do it!!!",
        ],
    )
    def test_short_messages_any_intent_are_ambiguous(self, text: str) -> None:
        """Messages ≤ 8 words should be ambiguous regardless of raw intent."""
        assert _is_ambiguous_followup(text, "EXIT") is True

    def test_long_question_is_ambiguous(self) -> None:
        """Longer messages classified as QUESTION are still ambiguous."""
        long_q = "can you explain what the function does in more detail than before please now"
        assert _is_ambiguous_followup(long_q, "QUESTION") is True

    def test_long_non_question_not_ambiguous(self) -> None:
        """Long messages with a non-QUESTION intent are not ambiguous."""
        long_msg = "I want to understand the entire build system and the cmake configuration in great detail"
        assert _is_ambiguous_followup(long_msg, "COMPILE") is False

    def test_exit_with_punctuation_not_ambiguous(self) -> None:
        """Hard exit phrases with punctuation should still be recognized."""
        assert _is_ambiguous_followup("exit!", "EXIT") is False
        assert _is_ambiguous_followup("quit.", "EXIT") is False
        assert _is_ambiguous_followup("bye!!", "EXIT") is False
