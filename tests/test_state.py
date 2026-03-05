"""Tests for agent/state.py — BuildState immutability, AgentState structure."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from agent.state import BuildState


class TestBuildStateImmutability:
    """Verify BuildState is frozen and cannot be mutated after construction."""

    def test_default_values(self) -> None:
        bs = BuildState()
        assert bs.status == "IDLE"
        assert bs.configured is False
        assert bs.built is False
        assert bs.tested is False
        assert bs.last_exit_code is None
        assert bs.last_error == ""
        assert bs.consecutive_errors == 0

    def test_frozen_cannot_set_attribute(self) -> None:
        bs = BuildState()
        with pytest.raises(FrozenInstanceError):
            bs.status = "FAILED"  # type: ignore[misc]

    def test_frozen_cannot_set_built(self) -> None:
        bs = BuildState()
        with pytest.raises(FrozenInstanceError):
            bs.built = True  # type: ignore[misc]

    def test_construction_with_values(self) -> None:
        bs = BuildState(
            status="SUCCESS",
            configured=True,
            built=True,
            tested=True,
            last_exit_code=0,
            last_error="",
            consecutive_errors=0,
        )
        assert bs.status == "SUCCESS"
        assert bs.configured is True
        assert bs.built is True

    def test_equality(self) -> None:
        a = BuildState(status="IDLE")
        b = BuildState(status="IDLE")
        assert a == b

    def test_inequality(self) -> None:
        a = BuildState(status="IDLE")
        b = BuildState(status="FAILED")
        assert a != b
