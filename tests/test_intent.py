"""Tests for agent/intent.py — keyword fallback classification."""

from __future__ import annotations

import pytest

from agent.intent import _fallback_classify, _fallback_followup


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
