"""Tests for agent/config.py — AgentConfig validation and immutability."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

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

    def test_new_production_defaults(self) -> None:
        config = AgentConfig()
        assert config.index_cache_enabled is True
        assert config.use_embedding_retrieval is False
        assert config.embedding_provider == "openai"
        assert config.embedding_model == "text-embedding-3-large"
        assert config.embedding_dimensions == 256
        assert config.background_reindex_enabled is False
        assert config.background_reindex_interval_seconds == 60.0
        assert config.cache_directory.name == ".cache"
        assert config.shell_timeout_seconds > 0
        assert config.allow_dangerous_shell_commands is False

    def test_embedding_provider_must_be_supported(self) -> None:
        with pytest.raises(ValueError, match="embedding_provider must be one of"):
            AgentConfig(embedding_provider="unsupported")

    def test_embedding_dimensions_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="embedding_dimensions must be > 0"):
            AgentConfig(embedding_dimensions=0)

    def test_background_reindex_interval_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="background_reindex_interval_seconds must be > 0"):
            AgentConfig(background_reindex_interval_seconds=0)

    def test_shell_timeout_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="shell_timeout_seconds must be > 0"):
            AgentConfig(shell_timeout_seconds=0)

    def test_cache_directory_must_be_outside_workspace(self, tmp_path: Path) -> None:
        workspace_path = tmp_path / "workspace" / "json"
        cache_directory = workspace_path / ".cache"
        with pytest.raises(ValueError, match="cache_directory must be outside workspace_path"):
            AgentConfig(
                workspace_path=workspace_path,
                cache_directory=cache_directory,
            )
