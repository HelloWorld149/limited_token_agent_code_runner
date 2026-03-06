"""Tests for agent/nodes.py helpers."""

from __future__ import annotations

from pathlib import Path
import subprocess

from agent.nodes import _probe_environment


def test_probe_environment_uses_shell_false(
    monkeypatch,
    tmp_path: Path,
) -> None:
    observed_calls: list[tuple[object, object]] = []

    monkeypatch.setattr("agent.nodes.shutil.which", lambda cmd: str(tmp_path / f"{cmd}.exe"))

    def _fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        observed_calls.append((args[0], kwargs["shell"]))
        return subprocess.CompletedProcess(args=args[0], returncode=0, stdout="tool 1.0\n", stderr="")

    monkeypatch.setattr(subprocess, "run", _fake_run)

    facts = _probe_environment(tmp_path)

    assert facts
    assert observed_calls
    assert all(shell is False for _, shell in observed_calls)
    assert all(isinstance(args, list) for args, _ in observed_calls)