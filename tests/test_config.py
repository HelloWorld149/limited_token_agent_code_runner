"""Tests for agent/config.py — AgentConfig validation and immutability."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from agent.config import AgentConfig


class TestAgentConfig:
    def test_default_construction(self) -> None:
        config = AgentConfig()
        assert config.token_budget == 5000
        assert config.input_token_budget + config.output_token_budget <= config.token_budget

    def test_frozen_immutability(self) -> None:
        config = AgentConfig()
        with pytest.raises(FrozenInstanceError):
            config.token_budget = 9999  # type: ignore[misc]

    def test_budget_invariant_violation(self) -> None:
        with pytest.raises(ValueError, match="input_token_budget \\+ output_token_budget"):
            AgentConfig(
                input_token_budget=4500,
                output_token_budget=1000,
                token_budget=5000,
            )

    def test_token_budget_must_be_5000(self) -> None:
        with pytest.raises(ValueError, match="token_budget must remain 5000"):
            AgentConfig(token_budget=3000, input_token_budget=2000, output_token_budget=1000)

    def test_effective_output_budget(self) -> None:
        config = AgentConfig()
        assert config.effective_output_budget <= config.output_token_budget
        assert config.effective_output_budget == min(config.output_token_budget, 800)
